# Randomized Benchmarking with Zero-Noise Extrapolation (RB+ZNE)

This directory contains an implementation of single-qubit randomized benchmarking (RB) extended with Zero-Noise Extrapolation (ZNE) for superconducting qubits.  
The workflow closely follows standard RB protocols, while introducing controlled noise scaling at the gate level to estimate intrinsic error rates in the zero-noise limit.
The implementation is designed to integrate seamlessly with Qolab’s existing **QUA-based benchmarking infrastructure** and to remain compatible with standard RB analysis pipelines within the broader [`qolab-start`](https://github.com/Qolaboration/qolab-start) framework.


---

## Overview

Randomized Benchmarking characterizes average single-qubit gate performance by applying random sequences of Clifford gates followed by a recovery operation that ideally returns the qubit to its ground state.  
In the presence of noise, the survival probability decays exponentially with sequence length, allowing extraction of the error per Clifford (EPC) and error per gate (EPG).

In this implementation, ZNE is incorporated by coherently scaling gate noise while preserving the logical Clifford operation. This enables extrapolation of RB decay parameters to the zero-noise limit, providing a refined estimate of intrinsic gate errors beyond raw hardware noise.

---

## Protocol Description

### 1. Random Clifford Sequence Generation
- Random single-qubit Clifford sequences are generated on the FPGA up to a user-specified maximum depth.
- For each requested RB length, the sequence is truncated accordingly.
- A recovery Clifford is appended using a preloaded Cayley table, ensuring the ideal final state is \|0⟩.

### 2. Noise Scaling for ZNE
- Each Clifford is decomposed into primitive hardware gates (e.g. X/Y rotations and Flux-Z operations).
- Noise scaling is applied by gate folding:
  - Virtual-Z and Flux-Z rotations are extended while preserving the target rotation angle.
  - X and Y rotations are replaced by folded sequences that implement the same unitary but accumulate additional noise.
- Multiple noise-scaled realizations are generated for each sequence length.

### 3. Execution and Averaging
- Each sequence is executed `n_avg` times to suppress shot noise.
- Additional averaging is performed over multiple random sequences per depth.
- The protocol supports both:
  - State discrimination, when readout calibration is reliable.
  - Raw I/Q quadrature readout, when discrimination is unavailable.

### 4. Post-Processing and Extrapolation
- Survival probabilities are fitted to exponential decay curves for each noise scale.
- ZNE is applied to extrapolate the decay parameter to the zero-noise limit.
- Final outputs include:
  - Zero-noise EPC
  - Zero-noise EPG
  - Comparison against standard RB estimates
