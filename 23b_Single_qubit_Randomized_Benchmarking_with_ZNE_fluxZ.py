"""
        SINGLE QUBIT RANDOMIZED BENCHMARKING WITH ZERO-NOISE EXTRAPOLATION

The program consists in playing random sequences of Clifford gates and measuring the state of the resonator afterward.
Each random sequence is derived on the FPGA for the maximum depth (specified as an input) and played for each depth
asked by the user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate,
found at each step thanks to a preloaded lookup table (Cayley table), that will bring the qubit back to its ground state.

To enable Zero-Noise Extrapolation (ZNE), each Clifford sequence is additionally executed under multiple effective noise scales.
This is achieved by modifying the primitive gate decompositions—such as by controlled gate folding or duration scaling—so that the 
implemented unitary remains unchanged while the noise strength is systematically increased. The same random Clifford sequences are 
reused across all noise scales to ensure consistent extrapolation.

If the readout has been calibrated and is good enough, state discrimination is applied to return only the qubit state.
Otherwise, the in-phase (I) and quadrature (Q) quadratures are returned.
Each sequence is played n_avg times for averaging. A second averaging is performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per gate

.
Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR and state discrimination.
    - Set the desired flux bias.


The gates are chosen based on : https://doi.org/10.1103/PhysRevX.9.021011

"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from quam_libs.lib.RB_z_rotations import c1_table_XZ
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    use_state_discrimination: bool = True
    use_strict_timing: bool = False
    num_random_sequences: int = 2  # Number of random sequences
    num_averages: int = 1
    max_circuit_depth: int = 20  # Maximum circuit depth
    delta_clifford: int = 10
    seed: int = 345324
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="23b_Single_Qubit_Randomized_Benchmarking_with_ZNE_fluxZ", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations

config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program_parameters}
num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
# Number of averaging loops for each random sequence
n_avg = node.parameters.num_averages
max_circuit_depth = node.parameters.max_circuit_depth  # Maximum circuit depth
if node.parameters.delta_clifford < 1:
    raise NotImplementedError("Delta clifford < 2 is not supported.")
#  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
delta_clifford = node.parameters.delta_clifford
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
num_depths = max_circuit_depth // delta_clifford + 1
seed = node.parameters.seed  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
strict_timing = node.parameters.use_strict_timing
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table_XZ[i, :] == 0)[0][0]) for i in range(24)]

# ZNE noise scales λ  (odd folding factors: λ = 2k + 1)
ZNE_SCALES = [1, 3, 5]


# %% {Utility functions}
def power_law(power, a, b, p):
    return a * (p**power) + b


def generate_sequence():
    cayley = declare(int, value=c1_table_XZ.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def _invert_z_label(label: str) -> str:
    # 180° Z rotations: inverse is identical up to global phase
    if "180" in label:
        return label

    # normal invert for ±90° or other asymmetric pulses
    if label.startswith("-"):
        return label[1:]
    return "-" + label


def z_pulse(qubit: Transmon, label: str, scale: int) -> None:
    """
    For noise scale λ = 2k + 1, implement:
        Z(θ) -> Z(θ)[Z(-θ) Z(θ)]^k

    where Z(θ) is the original flux pulse (label),
    and Z(-θ) is implemented by the inverse label.

    Only physical Flux-Z pulses are folded. Virtual Z and XY pulses
    are left unchanged elsewhere in the sequence.
    """
    if scale <= 1:
        # λ = 1: original behavior (no folding)
        qubit.z.play(label)
        return

    # Require odd λ = 2k + 1
    if scale % 2 != 1:
        raise ValueError(f"ZNE scale must be odd, got {scale}")

    inv_label = _invert_z_label(label)
    k = (scale - 1) // 2

    # First logical Z(θ)
    qubit.z.play(label)
    # Then k repetitions of Z(-θ) Z(θ)
    for _ in range(k):
        qubit.z.play(inv_label)
        qubit.z.play(label)


def play_sequence(sequence_list, depth, qubit: Transmon, zne_scale: int):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=False):
            with case_(0):  # I
                pass
            with case_(1):  # X180
                qubit.xy.play("x180")
            with case_(2):  # Y180
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x180")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
            with case_(3):  # Y180,X180
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z180", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
            with case_(4):  # x90, y90
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
            with case_(5):  # x90, -y90
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
            with case_(6):  # -x90, y90
                qubit.xy.play("-x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
            with case_(7):  # -x90, -y90
                qubit.xy.play("-x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
            with case_(8):  # y90 x90
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
            with case_(9):  # y90 -x90
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("-x90")
            with case_(10):  # -y90 x90
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
            with case_(11):  # -y90 -x90
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("-x90")
            with case_(12):  # x90
                qubit.xy.play("x90")
            with case_(13):  # -x90
                qubit.xy.play("-x90")
            with case_(14):  # y90
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
            with case_(15):  # -y90
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
            with case_(16):  # Z90
                z_pulse(qubit, "z90", zne_scale)
            with case_(17):  # -Z90
                z_pulse(qubit, "-z90", zne_scale)
            with case_(18):  # x180 y90
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
            with case_(19):  # x180 -y90
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
            with case_(20):  # Y180 X90
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
            with case_(21):  # Y180 -X90
                z_pulse(qubit, "-z90", zne_scale)
                qubit.align()
                qubit.xy.play("x90")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
                qubit.align()
                qubit.xy.play("-x90")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)
            with case_(22):  # x90 Y90 X90
                qubit.xy.play("x180")
                qubit.align()
                z_pulse(qubit, "z90", zne_scale)
            with case_(23):  # -x90 Y90 -X90
                qubit.xy.play("x180")
                qubit.align()
                z_pulse(qubit, "-z90", zne_scale)


# %% {QUA_program}
def build_randomized_benchmarking_program(zne_scale: int):
    with program() as randomized_benchmarking:
        depth = declare(int)  # QUA variable for the varying depth
        # QUA variable for the current depth (changes in steps of delta_clifford)
        depth_target = declare(int)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        state = [declare(int) for _ in range(num_qubits)]
        # The relevant streams
        m_st = declare_stream()
        # state_st = declare_stream()
        state_st = [declare_stream() for _ in range(num_qubits)]

        for i, qubit in enumerate(qubits):
            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

            # QUA for_ loop over the random sequences
            with for_(m, 0, m < num_of_sequences, m + 1):
                # Generate the random sequence of length max_circuit_depth
                sequence_list, inv_gate_list = generate_sequence()
                assign(depth_target, 0)  # Initialize the current depth to 0

                with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
                    # Replacing the last gate in the sequence with the sequence's inverse gate
                    # The original gate is saved in 'saved_gate' and is being restored at the end
                    assign(saved_gate, sequence_list[depth])
                    assign(sequence_list[depth], inv_gate_list[depth - 1])
                    # Only played the depth corresponding to target_depth
                    with if_((depth == 1) | (depth == depth_target)):
                        with for_(n, 0, n < n_avg, n + 1):
                            # Initialize the qubits
                            if reset_type == "active":
                                active_reset(qubit, "readout")
                            else:
                                qubit.resonator.wait(qubit.thermalization_time * u.ns)
                            # Align the two elements to play the sequence after qubit initialization
                            qubit.align()
                            # The strict_timing ensures that the sequence will be played without gaps
                            if strict_timing:
                                with strict_timing_():
                                    # Play the random sequence of desired depth
                                    # ONLY CHANGE vs 23: pass zne_scale
                                    play_sequence(sequence_list, depth, qubit, zne_scale)
                            else:
                                # ONLY CHANGE vs 23: pass zne_scale
                                play_sequence(sequence_list, depth, qubit, zne_scale)
                            # Align the two elements to measure after playing the circuit.
                            qubit.align()
                            readout_state(qubit, state[i])

                            save(state[i], state_st[i])

                        # Go to the next depth
                        assign(depth_target, depth_target + delta_clifford)
                    # Reset the last gate of the sequence back to the original Clifford gate
                    # (that was replaced by the recovery gate at the beginning)
                    assign(sequence_list[depth], saved_gate)
                # Save the counter for the progress bar
                save(m, m_st)

        with stream_processing():
            m_st.save("iteration")
            for i in range(num_qubits):
                state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                    f"state{i + 1}"
                )

    return randomized_benchmarking


# Container for per-λ ZNE analysis results (kept in memory, not JSON)
zne_results: dict[int, dict] = {}


# %% {Simulate_or_execute}
if node.parameters.simulate:
    randomized_benchmarking = build_randomized_benchmarking_program(zne_scale=1)
    simulation_config = SimulationConfig(duration=100_000)
    job = qmm.simulate(config, randomized_benchmarking, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results["figure"] = plt.gcf()
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    node.results = {}
    zne_summary: dict[str, dict] = {}

    for zne_scale in ZNE_SCALES:
        print(f"Running RB for λ = {zne_scale}")
        # Build QUA program for this λ
        randomized_benchmarking = build_randomized_benchmarking_program(zne_scale=zne_scale)

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(randomized_benchmarking)
            results = fetching_tool(job, ["iteration"], mode="live")
            while results.is_processing():
                m = results.fetch_all()[0]
                progress_counter(m, num_of_sequences, start_time=results.start_time)

        depths = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
        depths[0] = 1
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubits,
            {"depths": depths, "sequence": np.arange(num_of_sequences)},
        )

        da_state = 1 - ds["state"].mean(dim="sequence")
        da_state: xr.DataArray
        da_state.attrs = {"long_name": "p(0)"}
        da_state = da_state.assign_coords(depths=da_state.depths - 1)
        da_state = da_state.rename(depths="m")
        da_state.m.attrs = {"long_name": "no. of Cliffords"}
        da_fit = fit_decay_exp(da_state, "m")
        alpha = np.exp(da_fit.sel(fit_vals="decay"))

        num_of_x_gates = 0*3 + 1*8 + 2*13
        num_of_z_gates = 0*4 + 1*7 + 2*11 + 3*2
        average_gate_per_clifford = (num_of_x_gates + num_of_z_gates) / 24
        EPC = (1 - alpha) - (1 - alpha) / 2
        EPG = EPC / average_gate_per_clifford

        if zne_scale == 1:
            node.results["ds"] = ds
            node.results["fit_results"] = {}
            for q in qubits:
                node.results["fit_results"][q.name] = {}
                node.results["fit_results"][q.name]["EPC"] = EPC.sel(qubit=q.name).values
                node.results["fit_results"][q.name]["EPG"] = EPG.sel(qubit=q.name).values
                print(f"λ={zne_scale}, {q.name}: EPC={EPC.sel(qubit=q.name).values}")
                print(f"λ={zne_scale}, {q.name}: EPG={EPG.sel(qubit=q.name).values}")
        else:
            for q in qubits:
                print(f"λ={zne_scale}, {q.name}: EPC={EPC.sel(qubit=q.name).values}")
                print(f"λ={zne_scale}, {q.name}: EPG={EPG.sel(qubit=q.name).values}")

        # JSON-safe summary for this λ
        lam_key = f"lambda_{zne_scale}"
        zne_summary[lam_key] = {}
        for q in qubits:
            qname = q.name
            zne_summary[lam_key][qname] = {
                "EPC": float(EPC.sel(qubit=qname).values),
                "EPG": float(EPG.sel(qubit=qname).values),
            }

        # Also keep full data for plotting in-memory if you want:
        zne_results[zne_scale] = {
            "ds": ds,
            "da_state": da_state,
            "da_fit": da_fit,
            "EPC": EPC,
            "EPG": EPG,
        }

    node.results["zne_results"] = zne_summary

else:
    # load_data_id path unchanged from original RB
    ds, machine, json_data, qubits, node.parameters = load_dataset(
        node.parameters.load_data_id, parameters=node.parameters
    )
    node.results = {"ds": ds}
    da_state = 1 - ds["state"].mean(dim="sequence")
    da_state: xr.DataArray
    da_state.attrs = {"long_name": "p(0)"}
    da_state = da_state.assign_coords(depths=da_state.depths - 1)
    da_state = da_state.rename(depths="m")
    da_state.m.attrs = {"long_name": "no. of Cliffords"}
    da_fit = fit_decay_exp(da_state, "m")
    alpha = np.exp(da_fit.sel(fit_vals="decay"))
    num_of_x_gates = 0 * 3 + 1 * 8 + 2 * 13
    num_of_z_gates = 0 * 4 + 1 * 7 + 2 * 11 + 3 * 2
    average_gate_per_clifford = (num_of_x_gates + num_of_z_gates) / 24
    EPC = (1 - alpha) - (1 - alpha) / 2
    EPG = EPC / average_gate_per_clifford
    node.results["fit_results"] = {}
    for q in qubits:
        node.results["fit_results"][q.name] = {}
        node.results["fit_results"][q.name]["EPC"] = EPC.sel(qubit=q.name).values
        node.results["fit_results"][q.name]["EPG"] = EPG.sel(qubit=q.name).values
        print(f"{q.name}: EPC={EPC.sel(qubit=q.name).values}")
        print(f"{q.name}: EPG={EPG.sel(qubit=q.name).values}")


# %% {Plotting}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        grid_names = [q.grid_location for q in qubits]

        ref_scale = ZNE_SCALES[0]
        ref_ds = zne_results[ref_scale]["ds"]

        grid = QubitGrid(ref_ds, [q.grid_location for q in qubits])

        for ax, qubit in grid_iter(grid):
            qname = qubit["qubit"]

            for idx, zne_scale in enumerate(ZNE_SCALES):
                res = zne_results[zne_scale]
                ds = res["ds"]
                da_state = res["da_state"]
                da_fit = res["da_fit"]
                EPC = res["EPC"]
                EPG = res["EPG"]

                # Mean survival and std for this λ and qubit
                da_state_qubit = da_state.sel(qubit=qname)
                da_state_std = (
                    ds["state"].std(dim="sequence").sel(qubit=qname)
                    / np.sqrt(ds.sequence.size)
                )

                m_vals = da_state_qubit.m.values

                # Color per λ (C0, C1, C2, ...)
                color = f"C{idx}"

                # Data points with error bars
                ax.errorbar(
                    m_vals,
                    da_state_qubit,
                    yerr=da_state_std,
                    fmt="o",
                    capsize=2,
                    elinewidth=0.5,
                    markersize=3,
                    label=f"λ={zne_scale} data",
                    color=color,
                )

                # Fitted decay curve
                fit_dict = {
                    k: da_fit.sel(qubit=qname).sel(fit_vals=k).values
                    for k in da_fit.fit_vals.values
                }
                ax.plot(
                    m_vals,
                    decay_exp(m_vals, **fit_dict),
                    "--",
                    color=color,
                    label=f"λ={zne_scale} fit (f={1 - EPG.sel(qubit=qname).values:.5f})",
                )

            ax.set_title(f"{qname}", pad=22)
            ax.set_xlabel("Circuit depth")
            ax.set_ylabel("p(0)")
            ax.grid("all")
            ax.legend(fontsize=6)

        plt.tight_layout()
        plt.suptitle("RB with ZNE - flux Z rotations", y=1.02)

        # Store this combined figure as the node's primary figure
        node.results["figure"] = grid.fig

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

    else:
        # load_data_id plotting path (single RB curve, no ZNE sweep)
        grid_names = [q.grid_location for q in qubits]
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            da_state_qubit = da_state.sel(qubit=qubit["qubit"])
            da_state_std = ds["state"].std(dim="sequence").sel(qubit=qubit["qubit"]) / np.sqrt(ds.sequence.size)
            ax.errorbar(
                da_state_qubit.m,
                da_state_qubit,
                yerr=da_state_std,
                fmt=".",
                capsize=2,
                elinewidth=0.5,
            )
            ax.grid("all")
            m = da_state.m.values
            ax.set_title(qubit["qubit"], pad=22)
            ax.set_xlabel("Circuit depth")
            fit_dict = {k: da_fit.sel(**qubit).sel(fit_vals=k).values for k in da_fit.fit_vals.values}
            ax.plot(m, decay_exp(m, **fit_dict), "r--", label="fit")
            ax.text(
                0.0,
                1.07,
                f"RB fidelity = {1 - EPG.sel(**qubit).values:.5f}",
                transform=ax.transAxes,
            )
        plt.tight_layout()
        plt.suptitle("RB with ZNE - flux Z rotations")
        plt.show()
        node.results["figure"] = grid.fig

        if not node.parameters.simulate:
            node.outcomes = {q.name: "successful" for q in qubits}
            node.results["initial_parameters"] = node.parameters.model_dump()
            node.machine = machine
            node.save()

# %%is
