# Koopman-MPC for activated-sludge aeration

Convex, real-time Koopman-operator model predictive control (MPC) of activated-sludge
aeration, including nitrous oxide (N₂O). Code accompanying the manuscript:

> Arda, B. *Convex, real-time Koopman model predictive control of activated-sludge
> aeration: constraint-aware operation with an explicit energy–nitrous-oxide operating
> frontier.* Submitted to *Water Environment Research*.

A data-driven Koopman linear predictor of the nonlinear activated-sludge dynamics
(including N₂O) is identified by Extended Dynamic Mode Decomposition (EDMD) and embedded
in a convex quadratic program with explicit effluent-ammonia and total-nitrogen
constraints. The fugitive N₂O emission (bilinear in aeration and dissolved N₂O) is
optimized directly by successive linear programming (SLP), and the multi-zone controller
manipulates both aeration (K_La) and internal recirculation (Q_a).

## Repository contents

```
plant_model.py          Reduced single-reactor ASM + N2O plant
koopman_mpc.py          EDMD identification, emission-SLP Koopman-MPC, PI/ABAC baselines
nmpc_baseline.py        Nonlinear-MPC comparator (single-shooting SLSQP)
plant_model_bsm2.py     Five-tank ASM1 cascade with N2O and hydrolysis (40 states)
koopman_mpc_bsm2.py     MIMO emission-SLP Koopman-MPC + conventional cascade-PI baseline
eval_bsm2.py            Multi-seed variance, model-plant mismatch, energy-compliance frontier
results/                Numerical result files (results*.json)
requirements.txt        Python dependencies
LICENSE                 MIT License
CITATION.cff            Citation metadata (also used by Zenodo)
```

## Installation

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
```

Requires Python 3.9+ with NumPy, SciPy, OSQP, and Matplotlib.

## Reproducing the results

Single-reactor study (Sections 2–3):

```bash
python koopman_mpc.py ident        # identify the Koopman predictor
python koopman_mpc.py control       # closed-loop MPC vs PI vs ABAC + Pareto sweep
python nmpc_baseline.py             # nonlinear-MPC solve-time comparison
```

Five-tank multi-zone study (Section 4):

```bash
python koopman_mpc_bsm2.py ident    # identify the MIMO predictor
python koopman_mpc_bsm2.py control  # MIMO Koopman-MPC vs cascade-PI
python eval_bsm2.py variance        # 5-seed variance
python eval_bsm2.py mismatch        # +/-15-30% parameter mismatch
python eval_bsm2.py frontier        # energy-compliance frontier
```

Figures are written as PNG files and metrics as `results*.json`.

## Note on scope

The plants use IWA Activated Sludge Model No. 1 kinetics and stoichiometry with the
benchmark parameter set and a two-pathway AOB N₂O submodel. They are transparent reduced
implementations (ideal point-settler; no anaerobic-digester line), not the complete
plant-wide BSM2.

## License

MIT License — see `LICENSE`.
