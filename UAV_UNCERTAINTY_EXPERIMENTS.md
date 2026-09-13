# UAV uncertainty implementation and Route 11 experiments

Last updated: 2026-09-13

## Goal

Add simulated UAV assistance to the Gazebo autonomy stack while keeping the
counterfactual behavior close to Meridian Drive. A request returns a fixed
25 m x 25 m ground-truth map centered on the counterfactual ROI. The map
contains both obstacle occupancy and semantic traversability derived from
GOOSE labels. The practical experiment target is fewer than five UAV requests
over all of Route 11, without imposing an artificial request cap or suppressing
Meridian's decision-relevant occupancy counterfactual.

The field starting point is 20% uncertain swept-footprint exposure. Simulator
experiments may use a higher exposure threshold after calibrating the source
uncertainties.

## Current repository state

All work described here is present in the working tree but is not committed.
There were already related dirty changes when this calibration work began, so
do not discard or reset the working tree wholesale.

The current focused test command passes 25 tests:

```bash
conda run --no-capture-output --name rugged-ugv \
  python -m unittest discover -s autonomy -p 'test*.py'
```

The mature-counterfactual correction has now completed a full 1,247 s Route 11
test window at a 0.75 exposure threshold. It produced two UAV requests, both
decision-relevant occupancy counterfactuals, with one drag that resolved. This
meets the fewer-than-five request-count target for this seed and direction.

## Simulated UAV implementation

### Ground-truth response

`autonomy/meridian_drive/uav_ground_truth.py` reconstructs the seeded painted
vegetation from `maps/vegetation_paint.npz`, parses fixed rocks from
`worlds/hill_country.sdf`, and atomically writes the requested local NPZ.

The response is exactly 25 m x 25 m by default, at 0.25 m resolution
(100 x 100 cells), centered on the ROI selected by the counterfactual. It
contains:

- planner `cost`, `obstacle`, and `uncertainty` layers;
- raw `occupancy` values of 0 or 100;
- raw `semantic_label` GOOSE IDs;
- lower-left `origin_xy`, `resolution`, and `sequence` metadata.

The main labels used by the simulated world are soil 31, grass 50, brush 17,
tree 28, and rock 40.

### Assistance state machine

`autonomy/meridian_drive/assistance.py` implements a Meridian-like flow:

```text
DRIVING -> STOPPING -> REQUESTING -> WAITING -> FUSING -> DRIVING
```

It uses two seconds of per-source persistence, confirms a stopped vehicle,
waits for a new atomic map, gives the map a fusion settling interval, and then
resumes. It preserves Meridian's two-request limit for one source/5 m region.
Retries remain 25 m instead of Meridian's expanding ROI because the simulator
requirement calls for a fixed 25 m x 25 m map.

Stale UAV maps from previous trials are excluded. `ground_only` never loads a
stale result. `counterfactual_uav` is the intended experiment mode.

Every new request is now retained in a per-run
`uav_request_history.json`. Each entry records source, map type, fixed ROI,
exposure, and whether the request bypassed the exposure threshold because it
was `decision_relevant`. The latest request is still written to
`uav_request.json`, and overall state remains in `autonomy_status.json`.

## Source-uncertainty changes, in implemented order

### 1. Semantic traversal-cost variance

Before this change, assistance consumed the semantic viewer score:

```text
max(normalized entropy, exp(-support / 5), age uncertainty)
```

For one perfect Gazebo label, the weak-support term is about 0.819. At a 0.20
cell threshold, a cell needed more than eight perfect observations to become
"certain." In the inspected Route 11 map, about 99% of known semantic cells
therefore appeared uncertain.

The implementation now separates the viewer diagnostic from the assistance
product. Assistance uses the same posterior traversal-cost variance as
Meridian:

```text
Var(cost) = (E[cost^2] - E[cost]^2) / (support + 1)
```

The semantic cell-variance cutoff is 0.04. With the simulator's one-hot GOOSE
labels, a directly observed cell has zero class-cost variance, while an unknown
cell remains unknown. The old entropy/support/age composite is retained in
snapshots and the map viewer. `semantic_cost_variance` is also saved in
`ground_maps.npz`.

Primary files:

- `autonomy/meridian_drive/semantic_grids.py`
- `autonomy/meridian_drive/ground_mapping.py`
- `autonomy/meridian_drive/gazebo_node.py`
- `autonomy/meridian_drive/maps.py`

### 2. Simulator-specific occupancy confidence

The uncalibrated filter needed approximately three direct observations to fall
below Meridian's occupancy uncertainty tests:

- one ground observation: probability about 0.299, variance about 0.105;
- two: probability about 0.154, variance about 0.043;
- three: probability about 0.072, variance about 0.017.

Gazebo ray observations are deterministic, so assistance now interprets one
scan as three effective likelihood/evidence observations. The raw scan counts
used for persistent `SOLID` classification are unchanged; one scan cannot
prematurely establish a physical solid obstacle. `TALL` remains ambiguous.
The default confidence is still one in the shared filter, so this calibration
is enabled specifically by `GroundMapper` for the simulator.

Primary files:

- `autonomy/meridian_drive/obstacle_grid_logic.py`
- `autonomy/meridian_drive/ground_mapping.py`

### 3. Per-cell uncertainty maturity

Meridian's two-second source persistence can be satisfied by continuously
replacing one set of new frontier cells with another. No individual cell has
to remain unresolved for the full interval.

`MapStack` now tracks uncertainty maturity by source and world-grid cell. The
default simulator window is one second, configured with:

```text
--mapping-uncertainty-maturity 1.0
```

A cell must stay both uncertain and present in the swept rollout set for the
whole window. Dropping out of the current rollout population resets its age.
Aerial coverage or resolved ground evidence also clears it. Compatibility
callers that provide no clock retain immediate behavior.

An integration bug was subsequently found: exposure and the ROI used mature
cells, but the occupancy free/occupied hypotheses still resolved every raw
uncertain cell. This allowed one mature cell to make a large fresh frontier
look decision-relevant. The current code fixes this by applying the same
mature uncertainty mask to counterfactual substitutions. With maturity set to
zero, counterfactual behavior remains equivalent to Meridian's immediate
uncertainty set.

Primary files:

- `autonomy/meridian_drive/maps.py`
- `autonomy/meridian_drive/gazebo_node.py`
- `tools/run_experiment.py`

## Counterfactual behavior retained from Meridian

The occupancy source still uses:

- unknown evidence as uncertain;
- probability in [0.20, 0.80] as ambiguous;
- variance at least 0.04 as uncertain;
- free and occupied hypotheses of 0.05 and 0.95;
- collision probability 0.50;
- minimum useful progress 0.25 m;
- baseline viability below 0.20;
- best-hypothesis improvement of at least 0.15;
- free/occupied viability spread of at least 0.15.

A decision-relevant counterfactual intentionally bypasses the general exposure
threshold. This is why simply raising exposure from 0.50 to 0.75 did not reduce
the pre-fix run below seven requests: all seven were counterfactual requests.
No cooldown or global request cap has been added.

Canonical Meridian references used for comparison:

- `../meridian-drive/docs/autonomy/UAV_ASSISTANCE.md`
- `../meridian-drive/src/terrain_aware_mppi/include/terrain_aware_mppi/counterfactual.hpp`
- `../meridian-drive/src/terrain_aware_mppi/src/mppi_node.cpp`
- `../meridian-drive/src/ragnarhorn_remote_map/ragnarhorn_remote_map/assistance_policy.py`

## Route 11 experiment method

Every reported run used Route 11 forward, seed 7, a 25 m UAV result, Gazebo at
3x, and the drag harness. The harness declares the rover stuck after 30
simulator seconds without 0.5 m of motion and can drag it up to 20 times. Drags
are expected on this route and are independent of the UAV request count.

Common command shape:

```bash
./scripts/run_experiment.sh \
  --run-id RUN_ID \
  --rtf 3 \
  --routes Route-11 \
  --directions forward \
  --cycles 1 \
  --seed 7 \
  --assistance counterfactual_uav \
  --uav-uncertainty-threshold THRESHOLD \
  --mapping-uncertainty-maturity 1.0
```

Gazebo transport needs local sockets, so these integration commands may need
to be run outside the filesystem/network sandbox.

The trial deadline is route length / 0.5 m/s + 120 s, about 1,247 simulator
seconds. All completed calibration runs reached this deadline instead of the
final waypoint even though many traveled farther than the 563 m route length.
This makes the full-window request comparisons useful, but route completion
itself remains a separate harness/autonomy issue.

## Completed Route 11 results

| Run ID | Change under test | Exposure threshold | UAV requests | Drags | Resolved drags | Driven | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| `uavr11b` | Initial UAV/counterfactual baseline | 0.20 | 19 | 14 | 12 | 804.95 m | timeout |
| `uav_semvar_r11` | Semantic cost variance | 0.20 | 14 | 8 | 8 | 1,247.35 m | timeout |
| `uav_occconf_r11` | + occupancy confidence | 0.20 | 11 | 15 | 14 | 955.83 m | timeout |
| `uav_maturity_r11` | + 1 s cell maturity, before CF-mask fix | 0.20 | 10 | 17 | 17 | 859.00 m | timeout |
| `uav_threshold50_r11` | Pre-fix threshold calibration | 0.50 | 7 | 18 | 17 | 823.44 m | timeout |
| `uav_threshold75_r11` | Pre-fix threshold calibration | 0.75 | 7 | 19 | 18 | 803.08 m | timeout |
| `uav_maturecf_threshold75_r11_retry` | Correct mature counterfactual set | 0.75 | 2 | 1 | 1 | 1,913.32 m | timeout |

The progression before correcting the counterfactual maturity mask was:

```text
19 -> 14 -> 11 -> 10 -> 7
```

The 0.50 request exposures were:

```text
0.626, 0.018, 0.083, 0.726, 0.471, 0.574, 0.561
```

The last seven requests were in seven distinct route regions. At least the
0.018, 0.083, and 0.471 events bypassed the 0.50 exposure threshold through
the counterfactual.

For the pre-fix 0.75 run, all seven recorded requests were explicitly marked
decision-relevant, at mature exposures:

```text
0.317, 0.376, 0.729, 0.323, 0.544, 0.314, 0.618
```

This observation led directly to the mature-counterfactual mask correction.

Completed CSVs and logs are under:

```text
runtime/experiments/<run-id>/
```

## Interrupted corrected run

Run ID:

```text
uav_maturecf_threshold75_r11
```

Configuration: current code after the mature-counterfactual fix, threshold
0.75, cell maturity 1.0 s, seed 7, drag harness enabled.

Observed before interruption:

- at least simulator time 463 s;
- eight drags;
- zero UAV requests;
- latest recorded mature exposure about 0.0022;
- no UAV map had been loaded because no request had occurred.

The process was terminated by the interrupted agent turn. The run is not a
valid full-route result and has no `campaign.csv`.

## Completed corrected run

Run ID:

```text
uav_maturecf_threshold75_r11_retry
```

This retry completed the full fixed test window. Final result:

- 1,247.40 simulator seconds;
- 1,913.32 m driven;
- one drag, resolved;
- two UAV requests;
- both requests selected `lidar_occupancy`;
- both were explicitly `decision_relevant` counterfactuals;
- request exposures were 0.511 and 0.535;
- no semantic source-only request occurred;
- outcome was still `timeout`, not endpoint success.

The result meets the fewer-than-five request-count target for the tested seed
and forward direction. It does not establish multi-seed robustness, and the
timeout means it should be described as a complete fixed-duration Route 11
test window rather than a successful endpoint-to-endpoint traversal.

## Immediate next steps

1. Optionally rerun the corrected code at 0.50 while keeping maturity at one
   second. The current 0.75 result already retains two genuine counterfactual
   requests, so this comparison would determine whether a less conservative
   general threshold can also stay below five.
2. Repeat the successful 0.75 configuration over multiple vegetation/planner
   seeds and Route 11 reverse. Do not promote a new simulator default based on
   one stochastic forward run.
3. Add periodic exposure and counterfactual viability metrics. Request history
   records triggers only; it does not yet record near-misses.
4. If another seed produces five or more requests, increase cell
   maturity conservatively (for example 1.5 or 2.0 s) before weakening the
   counterfactual viability thresholds. Keep one variable fixed per Route 11
   comparison.
5. Add periodic shadow diagnostics containing per-source mature exposure,
   missing-cell fraction, ambiguous-probability fraction, high-variance
   fraction, and counterfactual baseline/free/occupied viability. Use a
   ground-only shadow run to detect false negatives without UAV maps changing
   the route.
6. Validate safety, not only request count: known occluded or obstacle-
   sensitive regions should still trigger, and reducing requests must not
   increase collision/stuck behavior materially.
7. Separately investigate why the drag-enabled Route 11 trials time out rather
   than reach the final waypoint. Preserve the drag code; it is expected and
   should remain enabled for these tests.

## Recommended next command

Use a new run ID because the interrupted run directory contains partial files:

```bash
./scripts/run_experiment.sh \
  --run-id uav_maturecf_threshold75_r11_retry \
  --rtf 3 \
  --routes Route-11 \
  --directions forward \
  --cycles 1 \
  --seed 7 \
  --assistance counterfactual_uav \
  --uav-uncertainty-threshold 0.75 \
  --mapping-uncertainty-maturity 1.0
```

## Defaults and cautions

- The CLI and manager defaults are still 0.20 exposure. The 0.75 value has
  only been an explicit experiment setting and should not become the default
  until the corrected full run and multi-seed validation are complete.
- Cell uncertainty and swept-footprint exposure are separate thresholds.
  Semantic and occupancy cell rules should stay at their Meridian-compatible
  values while the experiment-level exposure threshold is calibrated.
- Request counts vary with the physical trajectory, asynchronous sensor/map
  timing, and drag locations even with planner seed 7. Compare trends over
  multiple completed runs rather than treating one count as deterministic.
- Do not add a hard request cap merely to satisfy the fewer-than-five target.
  The existing two-request-per-source/region behavior is the Meridian policy
  mechanism and should remain the only regional exhaustion rule unless the
  experiment design explicitly changes.
