# Koopman-MPC for activated-sludge aeration

Real-time Koopman-operator model predictive control (MPC) of activated-sludge aeration, including nitrous oxide (N₂O). Code accompanying the manuscript:

> Kuş, B. A. *Convex, real-time Koopman model predictive control of activated-sludge aeration: constraint-aware operation with an explicit energy–nitrous-oxide operating frontier.* *Water Environment Research* (under revision, ms. 3395422).

A data-driven Koopman linear predictor of the nonlinear activated-sludge dynamics, N₂O included, is identified by Extended Dynamic Mode Decomposition (EDMD) and embedded in a receding-horizon problem with explicit effluent-ammonia and total-nitrogen constraints. The economic objective retains one bilinear term, the fugitive N₂O emission (aeration × dissolved N₂O); successive linear programming (SLP) reduces **each control step to a strictly convex quadratic program**. Each QP is solved to global optimality and returns a certificate; the bilinear problem the QPs approximate is not convex, and no global-optimality certificate is claimed for it. The multi-zone controller manipulates both aeration (`K_La`) and internal recirculation (`Q_a`).

## Repository contents

```
plant_model.py            Reduced single-reactor ASM + N2O plant (7 states)
koopman_mpc.py            EDMD identification, emission-SLP Koopman-MPC, PI/ABAC baselines
nmpc_baseline.py          Nonlinear-MPC comparator (single-shooting SLSQP)

plant_model_cascade.py    Five-tank ASM1 cascade with N2O and hydrolysis (40 states)
koopman_mpc_cascade.py    MIMO emission-SLP Koopman-MPC + conventional cascade-PI baseline
eval_cascade.py           Multi-seed variance, model-plant mismatch, energy-compliance frontier

plant_model_eval.py       Independently structured EVALUATION plant (8 states) for the
                          cross-model validation: explicit NH2OH intermediate, Haldane
                          oxygen law on the AOB denitrification pathway, N2O reductase sink
crossmodel.py             Runs the unchanged controller and both baselines on that plant
crossmodel_extra.py       Steady-state N2O characteristic of both plants; repeated-seed
                          statistics on the evaluation plant

revision_analyses.py      VAF and time-domain predictor validation, component ablation,
                          design-parameter sweep, move-suppression / solver-invariance
                          study, repeated-seed statistics, control-interval study
revision_figures.py       Earlier versions of the control-interval and cascade figures
figdata.py                Regenerates and caches the trajectories the figures need, and
                          ASSERTS that the recomputed indices match the archived JSON
figstyle.py               One house style for every figure (sizes, colours, panel letters)
figures_final.py          Figures 2-10 as submitted at revision, 400-dpi PNG + vector PDF
figure1_architecture.py   Figure 1, the control architecture
figS1_layout.py           Supporting-information figure S1, the cascade layout
graphical_abstract.py     Graphical abstract (bar values read from results_revision.json)

figures/                  The figures as submitted: 400-dpi PNG and vector PDF

results/                  Numerical result files (results*.json)
requirements.txt          Pinned Python dependencies
CHANGELOG.md              What changed at revision, and why
LICENSE                   MIT License
CITATION.cff              Citation metadata (also used by Zenodo)
```

The five-tank files were named `*_bsm2.py` in the first release. They were renamed to `*_cascade.py` because the plant is a benchmark-inspired reduced surrogate and is explicitly **not** the IWA BSM2; the old names invited the opposite reading.

## Installation

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
```

Requires Python 3.9+ with NumPy, SciPy, OSQP and Matplotlib.

## Reproducing the results

Single-reactor study:

```bash
python koopman_mpc.py ident            # identify the Koopman predictor
python koopman_mpc.py control          # closed loop vs PI and ABAC, plus the emission-weight sweep
python koopman_mpc.py robust           # multiple seeds and parameter mismatch
python nmpc_baseline.py                # nonlinear-MPC solve-time comparison
```

Analyses added at revision:

```bash
python revision_analyses.py vaf        # VAF and the time-domain predictor validation
python revision_analyses.py ablation   # component-wise ablation of the controller
python revision_analyses.py tuning     # horizon, Tikhonov parameter, dictionary size
python revision_analyses.py rdu        # move-suppression weight and solver invariance
python revision_analyses.py seeds      # ten realisations, all three controllers noise-matched
python revision_analyses.py dt         # control-interval study, 5 to 60 minutes
python figure1_architecture.py         # Figure 1
```

Figures as submitted. `figdata.py` re-runs the three simulations that produce
trajectories and refuses to write its cache if any recomputed index disagrees
with the archived JSON, so a figure can never be redrawn from different numbers
than the ones the paper reports:

Run the identification step above first: `figdata.py` loads `koop.npz` and
`x_ss.npy`, which are build products and are not tracked.

```bash
python figdata.py                      # cache_*.npz, with the numbers checked
python figures_final.py                # Figures 2-10, out/FigureN.png and .pdf
python graphical_abstract.py           # out/graphical_abstract.png and .pdf
```

Five-tank multi-zone study:

```bash
python koopman_mpc_cascade.py ident    # identify the MIMO predictor
python koopman_mpc_cascade.py control  # MIMO Koopman-MPC vs cascade-PI
python eval_cascade.py variance        # multi-seed variance
python eval_cascade.py mismatch        # +/-15-30% parameter mismatch
python eval_cascade.py frontier        # energy-compliance frontier
```

Cross-model validation:

```bash
python plant_model_eval.py             # steady-state characteristic of the evaluation plant
python crossmodel.py                   # unchanged controller + baselines on that plant
python crossmodel_extra.py             # both characteristics, and five noise realisations
```

Metrics are written as `results*.json`. `figures_final.py` and
`graphical_abstract.py` write to `out/`; the older scripts write PNGs beside
themselves.

## Computational environment and reproducibility

The closed loop of a receding-horizon controller is deterministic but sensitive: a difference in the last places of one solve propagates through the warm start and, in the SLP step, through the linearisation point of the next iteration. Three things are therefore fixed rather than left to defaults, in `osqp_settings()` in `koopman_mpc.py`:

* primal and dual tolerances of `1e-7` with solution polishing enabled, so the returned point is the exact solution of the identified active set;
* a **fixed adaptive-ρ interval** — OSQP derives this from a wall-clock measurement of its own setup time by default, which makes the iterate sequence depend on machine load;
* a move-suppression weight `r_du` large enough that the Hessian's curvature is commensurate with the linear term (`R_DU = 3e-1` for the single reactor, `R_DU_BSM2 = (3e-2, 3e-2)` for the cascade), chosen as the smallest value on a half-decade grid at which every reported index is invariant across solver builds.

Both plants are integrated with a common sub-step (`SUBSTEP_MIN = 0.5` minutes in `koopman_mpc_cascade.py`) so that the controller and the baselines are compared on the same numerical experiment.

With these settings the study was verified in three environments — OSQP 0.6.3 / NumPy 1.26.4 / SciPy 1.11.4, OSQP 0.6.7 / NumPy 2.4.4, and OSQP 1.1.3 / NumPy 2.4.4. Aeration energy agrees across builds to within 0.16%, cumulative N₂O emission to within 0.28%, and the ammonia violation time and peak are identical. Repeated runs within one environment are bit-identical. The seeds of every reported run are fixed in the scripts.

## Note on scope

The plants use IWA Activated Sludge Model No. 1 kinetics and stoichiometry with the benchmark parameter set and a two-pathway AOB N₂O submodel. They are transparent reduced implementations (ideal point settler; no anaerobic-digester line), **not** the complete plant-wide BSM2, and `plant_model_eval.py` is an independently *structured* evaluation plant but our own *implementation*, not a third-party benchmark.

The N₂O submodel of `plant_model.py` is phenomenological and uncalibrated. The cross-model validation shows the emission magnitude changing by a factor of 2.8, and the ranking of the three controllers on N₂O reversing, when the submodel structure changes, while the compliance and energy results are unaffected. Emission magnitudes from these models should not be read as plant predictions.

## License

MIT License — see `LICENSE`.
