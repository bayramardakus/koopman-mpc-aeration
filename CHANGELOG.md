# Changelog

## v2.0 — first revision of ms. 3395422 (Water Environment Research)

Everything below is a change to the code released with the first submission. Two of them are corrections of defects that changed reported numbers; the rest are additions demanded by the review, or renames.

---

### Corrections that changed reported numbers

**1. The closed loop was not reproducible across solver builds — or between two runs on the same machine.**

Two causes, both in the QP setup.

* OSQP derives its adaptive-ρ interval from a wall-clock measurement of its own setup time. The iterate sequence therefore depended on machine load, and the closed loop, which feeds each solution forward through the warm start and through the SLP linearisation point, amplified that into visibly different trajectories. The interval is now fixed at a constant number of iterations.
* The move-suppression weight was small enough (`r_du = 3e-3`) that the smallest eigenvalue of the Hessian lay three orders of magnitude below the norm of the linear term. The minimiser was unique, but the cost surface was nearly flat and a first-order solver returned a point that depended on its own build and tolerances. `r_du` is now `3e-1` (single reactor) and `(3e-2, 3e-2)` (cascade) — the smallest values on a half-decade grid at which every reported index is invariant across three solver builds.

Tolerances were also tightened to `eps_abs = eps_rel = 1e-7` with polishing enabled. All of this is collected in `osqp_settings()` in `koopman_mpc.py`, which `koopman_mpc_cascade.py` imports, so the two studies cannot drift apart.

*Effect.* Identification results are unchanged. Both single-reactor baselines are unchanged. The single-reactor MPC now records **0.0 h** of ammonia violation and a **3.31 mgN L⁻¹** peak over the 10-day window, against 4.5 h and 5.46 previously, at essentially unchanged aeration energy. The emission-weight frontier is now monotone and its span narrower.

**2. The cascade MPC and its baseline integrated the plant at different step sizes.**

`run_mpc` advanced the plant with a 3.75-minute Runge–Kutta sub-step (`nsub=4` at a 15-minute interval) while `run_baseline` used 0.5 minutes. At the aeration rates the controller commands, the oxygen mode has a time constant of about 5.5 minutes and the N₂O stripping term about 10, so the coarser sub-step did not resolve the emission integral: the two runs were not the same numerical experiment. A convergence study shows the indices converged at a 1-minute sub-step and unchanged down to 0.125. Both runners now take their sub-step from one module constant, `SUBSTEP_MIN = 0.5`.

*Effect on the cascade comparison against cascade-PI.* Ammonia violation 69.0 h → **0.0 h** (was reported as 1 h); nitrogen-load effluent index **−38%** (was −43%); N₂O **+2.3%**, i.e. parity (was reported as −27%); operational cost index **+16.3%** (was +30%).

**3. The cascade emission weight drove the aeration input onto its bound.**

With the integration corrected, sweeping `w_N` on the cascade shows that any positive weight up to 1 pins `K_La` at its 260 d⁻¹ bound for 99% of the window — buying about 5% emission reduction at the cost of 17% more aeration energy, 32% more effluent nitrogen load and the reintroduction of ammonia violation — and that `w_N ≥ 2` destabilises the loop. The reported operating point is now `W_N_OPERATING = 0.0`, which is the best point on every effluent and energy measure, and the sweep is reported in the paper rather than hidden.

---

### Additions required by the review

* `revision_analyses.py` — VAF and time-domain predictor validation against the nonlinear plant; component-wise ablation (no Koopman lift, no forecast, no offset-free correction, dissolved-N₂O proxy, myopic and short horizons); design-parameter sweeps over prediction horizon, Tikhonov parameter and dictionary size; the move-suppression / solver-invariance study; ten-realisation statistics with **all three controllers exposed to the same measurement noise**, which the first release did not do; and the control-interval study at 5, 10, 15, 30 and 60 minutes with the horizon held at three hours of physical time.
* `revision_figures.py` — the control-interval figure and the cascade closed-loop figure, which previously had no script.
* `figure1_architecture.py` — Figure 1, redrawn so the controller emits one input vector with both components and their units labelled.
* `plant_model_eval.py`, `crossmodel.py`, `crossmodel_extra.py` — the cross-model validation. The evaluation plant differs from the design model structurally, and in no other way: an explicit hydroxylamine intermediate (8 states rather than 7, invisible to the controller), a Haldane oxygen dependence on the AOB denitrification pathway with its maximum at 0.65 mg O₂ L⁻¹, and an N₂O reductase sink that the design model cannot represent. The controller, dictionary, weights, horizon and solver settings are unchanged.
* `KoopMPC` gained an `offset_free` switch so the offset-free correction can be ablated.

---

### Figures redrawn

All figures were redrawn for legibility and for Wiley's revision requirement that figures be supplied as separate high-resolution files. **No number changed.** `figdata.py` re-runs the three simulations that produce trajectories and aborts unless every recomputed index matches the archived JSON to the digits reported; `figures_final.py` then draws from those caches and from `results*.json`, never from a fresh simulation. The recomputation agrees exactly: VAF to four figures, and every closed-loop index on both plants to the last archived digit.

What changed in the drawing:

* One house style (`figstyle.py`): one type size, one colour per controller across every figure, panel letters outside the axes, 400-dpi PNG and a vector PDF for each figure.
* Legends and annotations that sat on top of data were moved off it — the legend in the closed-loop and VAF figures, the VAF values in the time-domain figure, the PI label in the frontier figure.
* Figure 5 became 2 x 2 instead of 1 x 4, and its abscissa equally spaced, because at journal width four panels in a row left the 5-, 10- and 15-minute tick labels overlapping.
* Figure 7 reports the cascade emission in kgN d⁻¹ rather than gN d⁻¹, matching the text, and its recirculation panel was given less height because both controllers hold that input constant.
* Figure 9 now carries both rows its caption has always described: the box-and-whisker distributions **and** the mean ± standard deviation bars. The released figure showed only the first; the second existed as a separate unused file.
* **Figure 8's ordinate was mislabelled.** `eval_cascade.frontier()` runs a five-day window, and the axis read `h / 7 d`. It now reads `h / 5 d`, and the manuscript sentence that quotes 49.7 h and 12.7 h now names the window. No frontier value changed.
* Graphical abstract: bar values are read from `results_revision.json` rather than typed in, the controller colours match the figures, and the zero-violation bar carries an explicit zero rule so it does not read as missing data.

---

### Renames and housekeeping

* `plant_model_bsm2.py` → `plant_model_cascade.py`, `koopman_mpc_bsm2.py` → `koopman_mpc_cascade.py`, `eval_bsm2.py` → `eval_cascade.py`. The plant is a benchmark-inspired reduced surrogate and is explicitly not the IWA BSM2; the old names invited the opposite reading, and a reviewer asked about BSM2G.
* `np.trapz` → `np.trapezoid` with a 1.x/2.x compatibility shim. NumPy 2 removed `trapz`, so the released code did not run at all on a current NumPy.
* `requirements.txt` now pins exact versions and lists the two additional builds the results were verified against.
* The cascade effluent-quality index keeps its equal nitrogen weights (β_TKN = β_NO = 20) but the code comment now states plainly that this is an index *of the form of* the benchmark EQI restricted to its nitrogen terms, and not the benchmark EQI: the solids, COD and BOD terms are absent because the ideal point settler does not resolve them, and the benchmark weights are 30 and 10.
* The cascade evaluation window is 12 days, matching what the paper reports. The released default was 8 days, which stops inside the rain period and never reaches the storm.
