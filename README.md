# wavevector-surface-response

Reproducible Monte Carlo + response-analysis pipeline for a one-component liquid-vapor interface under a spatially periodic external field.

Primary response observable:

\[
\gamma(A,\mathbf q)=\gamma_0+\frac12\chi_\gamma(\mathbf q)A^2+O(A^4),\qquad
\chi_\gamma(\mathbf q)=\left.\frac{\partial^2\gamma}{\partial A^2}\right|_{A=0}.
\]

The first target is to test a **Goldstone-displacement -> microscopic-packing crossover** by decomposing density fluctuations

\[
\delta\rho_q(z)=h_q[-\rho_0'(z)]+\delta\rho_q^\perp(z)
\]

and measuring

\[
\chi_{\rm total}=\chi_{hh}+2\chi_{h\perp}+\chi_{\perp\perp}.
\]

## Pipeline

1. Generate an NVT Lennard-Jones liquid-vapor slab with Metropolis MC.
2. Sample lateral density modes \(\rho_q(z,t)\) at PBC-compatible physical wavevectors.
3. Build the covariance kernel \(C(z,z';q)\).
4. Diagonalize \(C\) and compute overlap with the translational template \(-\rho_0'(z)\).
5. Project each snapshot into displacement and orthogonal sectors.
6. Compute sector susceptibilities and closure error.
7. Perform finite-size and sampling audits.

## Repository layout

```text
config/                 simulation/analysis configuration
src/wvsr/               MC and analysis package
scripts/                 command-line entry points
results/                 generated outputs (ignored except small summaries)
.github/workflows/       reproducible CI smoke tests
```

## First-stage scope

The initial implementation deliberately focuses on \(q_\perp=0\). This removes the extra absolute-interface-phase ambiguity of normal/oblique fields and lets us establish the lateral response before treating \(q_\perp\neq0\) as a separate phase/pinning problem.

## Core diagnostics

For each physical \(q\), save at least:

- covariance eigenvalues \(\lambda_n(q)\)
- translational overlap \(O_n(q)\)
- \(\chi_{hh}\), \(2\chi_{h\perp}\), \(\chi_{\perp\perp}\)
- closure error
- integrated autocorrelation estimates / effective sample count
- box size and exact reciprocal-lattice mode index

A convincing Goldstone-to-packing crossover would show a low-q mode with strong overlap with \(-\rho_0'(z)\), capillary-like enhancement at small q, and progressive hybridization with non-translational density modes near the liquid packing scale \(q_*\).
