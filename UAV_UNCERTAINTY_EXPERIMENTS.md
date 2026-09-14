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

The current focused test command passes 43 tests:

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

### One request, one product

A request names the single evidence channel that crossed its uncertainty rule,
and the response now carries only that product:

- `lidar_occupancy` raises `canopy_obstacle`, and the NPZ holds `obstacle` and
  `occupancy` rasterized from physical bodies;
- `ground_semantic_cost` raises `semantic_traversability`, and the NPZ holds
  `cost`, `semantic_label`, and an untraversable mask derived from the GOOSE
  labels rather than from bodies.

`UavMap.map_types` records this, and `UavMap.provides()` gates it. Aerial
evidence supersedes ground uncertainty only on the channel it answers, in both
`_evidence_exposure` and the occupancy counterfactual substitution. A map that
declares no types is treated as a legacy or external product that speaks for
every channel, so Meridian rasters are unaffected.

Before this change the producer always wrote both products. An occupancy
request therefore also delivered a full GOOSE semantic raster whose zero
uncertainty layer marked every uncertain semantic cell in the 25 m window as
resolved, even though no semantic source had asked for anything. The two
products deliberately differ in extent: occupancy paints the body a wheel can
strike (tree trunk, 0.20 m), while semantic traversability paints the labeled
canopy (1.35 m), matching what the ground semantic layer already treats as
untraversable.

Primary files:

- `autonomy/meridian_drive/uav_ground_truth.py`
- `autonomy/meridian_drive/maps.py`
- `autonomy/meridian_drive/assistance.py`

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
| `r11_cf_rtf1` | + map-type gating, at 1.0x RTF | 0.75 | 5 | 7 | 7 | 865.55 m | success |
| `r11_ground_rtf1` | Ground-only control, at 1.0x RTF | n/a | 0 | 3 | 3 | 1,495.25 m | success |
| `r11_cf_rtf1_v2` | + tick-budget fixes | 0.75 | 3 | 8 | 8 | 848.36 m | success |
| `r11_ground_rtf1_v2` | Ground-only control, same code | n/a | 0 | 3 | 3 | 680.70 m | success |

The `_v2` pair supersedes the `_v1` pair. Every other row ran at 3x with no
ground-only control beside it, so those measure request count only.

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

## Route 11 at 1.0x real time, against a ground-only control

Run IDs `r11_cf_rtf1` and `r11_ground_rtf1`. Both drove Route 11 forward, seed
7, maturity 1.0 s, drag harness enabled, default vegetation, on the current
code including map-type gating. The only difference between them is
`--assistance`. This is the first paired comparison in this document, and the
first time either mode reached the final waypoint instead of the deadline.

| Metric | `counterfactual_uav` | `ground_only` |
|---|---:|---:|
| Outcome | success | success |
| Simulator time | 854.74 s | 1,095.39 s |
| Distance driven | 865.55 m | 1,495.25 m |
| Route progress | 562.90 m | 562.90 m |
| Path / route length | 1.54x | 2.66x |
| Drags (all resolved) | 7 | 3 |
| UAV requests | 5 | 0 |

Assistance shortened the trial by 240.65 simulator seconds (22%) and removed
629.70 m of driving (42%). Both runs covered the same 563 m route, so the
difference is entirely wandering that the ground-only run did and the assisted
run did not. Against that, the assisted run wedged more than twice as often.

All five requests were `lidar_occupancy` and `canopy_obstacle`, all five were
marked `decision_relevant`, and every exposure was below the configured 0.75
threshold:

```text
0.387, 0.300, 0.205, 0.012, 0.678
```

The exposure threshold therefore gated nothing this run; the counterfactual
admitted every request on its own. Requests 1 and 5 sit in the same ground
region because Route 11 is an out-and-back loop that passes it at arc position
87 m and again at 451 m, so the two-per-source/region rule did not apply. No
semantic request occurred, so map-type gating changed only what the occupancy
answers were allowed to resolve, not which product was chosen.

### Superseded

Read this pair together with the `_v2` pair below. Both `_v1` runs were
measured while the controller was shedding half its ticks, and the ground-only
control was the more damaged of the two once the tick budget was repaired. The
`_v1` numbers are kept because they are what motivated the profiling, not
because the comparison stands.

### What this run does and does not establish

It does not establish that assistance helps. Three things are unresolved:

1. The drag harness confounds the headline numbers in both directions. Each
   drag costs 30 simulator seconds of stalled progress before it fires, so the
   assisted run paid about 120 s more in stuck detection than the control and
   still finished 240 s sooner. But each drag also teleports the rover at least
   3 m along the route, so the assisted run was also handed more free progress.
2. The controller does not hold its 20 Hz tick even at 1.0x RTF. Averaged over
   the run it missed 62 of every 100 cycles under assistance and 53 under
   ground-only, and both degrade badly through the middle of the route:

   ```text
   counterfactual_uav:  12 36 37 32 76 81 44 80 89 133
   ground_only:         11 10 31 30 68 97 84 105 74 23
   ```

   Both modes are compute-starved, and the assisted mode is the heavier of the
   two because it samples the aerial grid in `MapStack.cost` and in both
   evaluators. Part of the extra drag count may be that penalty rather than the
   aerial evidence itself.
3. It is one seed in one direction. Request counts vary with the trajectory
   even at fixed seed, as noted below.

Five requests also misses the fewer-than-five target, though not by the
mechanism the earlier calibration was chasing: the threshold is no longer what
admits them.

## Why the controller was missing cycles

The missed cycles in `r11_cf_rtf1` and `r11_ground_rtf1` were not caused by
Gazebo, by running two trials at once, or by a shortage of cores. The pair used
about 3 of 12 cores, and `gazebo_node` itself sat at 87% CPU, which is the
signature of one saturated core rather than a loaded machine.

`gazebo_node` is a single Python process. The 20 Hz controller, the lidar
transport callback, and the mapping worker all contend for one GIL, so their
costs add on a single core regardless of how many cores exist.

Measured per-call, before any fix:

| Work | Cost | Rate | Per second |
|---|---:|---:|---:|
| `GroundMapper.update` | 37.3 ms | 10 Hz | 373 ms/s |
| `write_snapshot` | 11.9 ms | 5 Hz | 59 ms/s |
| `SemanticMapper.project_if_ready` | 11.4 ms | 2 Hz | 23 ms/s |
| `SemanticMapper.render_layers` | 5.3 ms | 2 Hz | 11 ms/s |
| Controller tick, late-route map | 21 ms | 20 Hz | 420 ms/s |
| Controller tick, unexplored frontier | 42 ms | 20 Hz | 842 ms/s |

On explored ground that totals about 89% of a core, which is survivable. On an
unexplored frontier it totals about 131% of a core, which is not, and the
controller sheds ticks until the frontier resolves. That is exactly the
observed shape: both modes ran clean at the start, spiked through the middle of
the route, and recovered near the end.

Two hot spots accounted for almost all of it, and both were per-cell Python
loops inside otherwise vectorised numpy code.

### `LocalGrid._ray_clear_mask`

31.3 ms of the 37.3 ms `GroundMapper.update`. It marched every lidar ray cell
by cell in Python: roughly 360 azimuth bins by up to 180 cells, calling
`np.floor` and `np.isfinite` on Python scalars at each step. Rewritten as a
run-length-indexed numpy march over all rays at once, `GroundMapper.update`
drops to 15.4 ms.

### `MapStack._mature_uncertainty` and `_mature_sample_uncertainty`

Both probed a `dict` keyed by `(cell_x, cell_y)` once per uncertain cell. The
occupancy counterfactual does this for nine footprint offsets over the whole
rollout population, so an unexplored frontier costs up to ~69,000 Python dict
lookups per tick. Maturity state is now a sorted `int64` key array plus its
timestamps, looked up with `searchsorted`. Worst-case `evaluate_assistance`
drops from 32.5 ms to 15.7 ms.

Both rewrites are behaviour-preserving, and were checked that way rather than
by inspection: `_ray_clear_mask` is cell-for-cell identical to the original
loop across randomised scans including non-finite returns, unknown ground, and
rays leaving the window; the maturity gates match the original dict
implementation on every field of `AssistanceEvaluation` over 120-tick replays
at maturity 0.0, 1.0, and 2.0, including a backwards clock jump.

Combined, the single-core demand on an unexplored frontier falls from about
131% to about 76%.

### Consequences for interpretation

The `_v1` pair was measured while the controller was shedding half its ticks,
which turned out to invalidate its central finding; see the `_v2` pair above.
Any future autonomy comparison should report the missed-cycle rate alongside
its result, because a mode that is merely more expensive per tick can look like
a mode that drives differently.

### Remaining, not yet changed

`write_snapshot` still costs 59 ms/s for diagnostics only. It is
`np.savez_compressed`; the same payload written with `np.savez` takes 1.3 ms
instead of 10.6 ms, at roughly 1 MB per write instead of 167 KiB to a file that
is overwritten rather than accumulated. Worth taking if tick budget gets tight
again.

Hardware note: because the GIL serialises these three consumers, this workload
is bound by single-thread speed. A machine with more cores will not raise the
controller's tick rate; a machine with a faster core will.

## Route 11 at 1.0x real time, after the tick-budget fixes

Run IDs `r11_cf_rtf1_v2` and `r11_ground_rtf1_v2`, same configuration as the
`_v1` pair, on code with the vectorised ray march and maturity gates, merged
with the 1.0 m arrival radius.

| Run | Outcome | Sim time | Driven | Path / route | Drags | UAV | Missed / 100 |
|---|---|---:|---:|---:|---:|---:|---:|
| `counterfactual_uav` v1 | success | 855 s | 866 m | 1.54x | 7 | 5 | 62.3 |
| `counterfactual_uav` v2 | success | 861 s | 848 m | 1.51x | 8 | 3 | 17.0 |
| `ground_only` v1 | success | 1,095 s | 1,495 m | 2.66x | 3 | 0 | 53.1 |
| `ground_only` v2 | success | **570 s** | **681 m** | **1.21x** | 3 | 0 | 7.1 |

### This reverses the earlier conclusion

Ground-only improved enormously from the tick-budget fixes: 1,095 s to 570 s
and 1,495 m to 681 m, driving the route at 1.21x its length instead of 2.66x.
The assisted mode barely moved: 855 s to 861 s, 866 m to 848 m.

So the `_v1` finding that assistance saved 22% of the time and 42% of the
driving was an artifact. It was not measuring assistance; it was measuring
which of the two modes was hurt more by controller starvation.

With the controller mostly healthy, on this seed, assistance is the worse
configuration: 51% more simulator time, 25% more driving, and 8 drags against
3.

### Why that is probably not just the remaining compute gap

The assisted mode still misses more ticks than the control, 17.0 against 7.1,
because the aerial layer is sampled in `MapStack.cost` and in both evaluators.
Measured directly on the `r11_cf_rtf1_v2` snapshot, loading the aerial map
takes the median tick from 17.6 ms to 20.9 ms, a 3.3 ms or 19% increase against
a 50 ms budget. That is too small to account for a 51% difference in completion
time, so the deficit is most likely behavioural rather than computational.

Two candidate mechanisms, neither tested:

1. Aerial obstacles enter cost at weight 20.0 with a hard collision at 0.65,
   and the ground-truth product is sharper than anything the lidar filter
   produces. That may over-constrain the planner into wedging, which is
   consistent with 8 drags against 3.
2. Every request stops the vehicle through `STOPPING -> REQUESTING -> WAITING
   -> FUSING`. Three requests is not obviously 290 s of holding, but the hold
   also discards planner state at each stop.

Request behaviour itself improved: three requests instead of five, all
`lidar_occupancy` and `canopy_obstacle`, all `decision_relevant`, at exposures
0.338, 0.326, and 0.254. This meets the fewer-than-five target. The faster
controller resolves frontier cells sooner, so fewer of them mature into a
request.

## Immediate next steps

## Drag-regression debugging after the `_v2` comparison

The `_v2` result established the actual regression: Route 11 forward seed 7
needed eight drags with `counterfactual_uav`, against three with
`ground_only`. Debugging found that this was not one tuning error. Several
interfaces made UAV assistance structurally unable to help reliably.

### UAV truth did not supersede planner evidence

The uncertainty evaluator correctly let an answered UAV product supersede its
matching ground uncertainty, but `MapStack.cost` did not. Planner costs were a
maximum over ground and aerial layers. Consequently, clear UAV occupancy could
clear an assistance trigger but could not remove a false-positive ground
occupancy cost or collision. Aerial evidence could only add constraints; it
could never open a route that ground sensing had incorrectly closed.

Planner fusion now applies product-specific supersession inside aerial
coverage:

- `canopy_obstacle` replaces classified and probabilistic lidar occupancy;
- `semantic_traversability` replaces ground semantic cost and obstacle
  probability;
- neither product replaces the other channel or evidence outside its 25 m
  footprint.

Two focused tests pin both directions of this rule.

### Local UAV products were being discarded

`MapStack` held only one `aerial` map. Every atomic result replaced it. Because
requests can alternate occupancy and semantics, a semantic response erased the
last occupancy answer; moving into another region also erased all earlier
coverage. The policy therefore repeatedly rediscovered uncertainty it had
already paid to resolve.

The stack now retains UAV products across channels and route regions. For
overlapping products of the same type, the newest sequence wins. Ground-only
initialization clears the full retained history, so stale maps still cannot
leak into a control trial.

### A hard raster supplied no clearance margin

The aerial obstacle body was sharp and binary. The local guide treated only the
body cells as blocked and could choose the immediately adjacent grid cell. Map
resolution, footprint discretization, and Gazebo contact then turned a
nominally clear detour into a physical wedge. UAV maps now derive a 0.75 m
linear soft-clearance field around confirmed bodies. The original hard body
and collision threshold are unchanged; the added field supplies a lateral
gradient without inventing a larger hard obstacle.

### Confirmed overlap had no executable escape

The existing overlap exception intended to choose the rollout that left a
known collision fastest. Two later rules defeated it: controls were
forward-only, and the generic minimum-forward-progress penalty overrode the
overlap cost. The planner can now sample bounded reverse maneuvers only while
its current footprint is confirmed occupied, and the generic forward-progress
penalty is disabled during that overlap. A unit test places the rover in a
forward-closed contact geometry and verifies that it backs into clear space.

### Loss-of-mobility requests and rejected experiment

The 0.75 population exposure threshold sometimes left UAV assistance dormant
through every drag region. Wheel-derived odometry was not a valid fallback
trigger because it remains high during wheel slip. The node now detects five
simulator seconds with less than 0.25 m of world-pose displacement. If the
rollout still contains a mature uncertain ROI, that one physical stall episode
may raise one request through the existing two-second source persistence. It
does not retrigger until real displacement resumes. Request history records
this as `mobility_relevant`.

An intermediate run, `r11_uav_world_stall_rtf1`, is explicitly rejected. It
completed in 752 s with five drags and 18 requests. That run reset its mobility
episode during every requested hold and replaced each preceding UAV product,
creating an occupancy/semantic request loop. The retained-map and edge-trigger
fixes were made from this trace.

### Reproducibility correction

`--seed` seeded MPPI but `scripts/run_experiment.sh` did not pass a Gazebo
physics seed. The launcher now passes deterministic seed 4207 by default,
overrideable with `GZ_SIM_SEED`. Callback scheduling can still perturb the
single-process Python controller, so one paired result remains insufficient,
but simulator physics is no longer silently unseeded.

### Current verification status

The focused suite now passes 43 tests:

```bash
conda run --no-capture-output --name rugged-ugv \
  python -m unittest discover -s autonomy -p 'test*.py'
```

The first end-to-end run with retained products and overlap escape was
`r11_uav_persistent_escape_full_rtf1`, Route 11 forward, planner seed 7,
Gazebo seed 4207, 1.0x RTF, threshold 0.75, and maturity 1.0 s. It completed:

| Run | Outcome | Sim time | Driven | Drags | UAV requests |
|---|---|---:|---:|---:|---:|
| assisted `_v2` regression | success | 861 s | 848 m | 8 | 3 |
| persistent fusion + escape | success | 645 s | 702 m | 4 | 6 |

This is a material improvement over the assisted regression: 216 fewer
simulator seconds, 146 m less driving, and half as many drags. It is not yet an
accepted result. Four drags does not beat the old three-drag ground-only
control, and six requests misses the fewer-than-five request goal.

The old control is no longer an exact comparator because it predates the fixed
Gazebo seed and bounded reverse-overlap behavior. A fresh ground-only run with
the same launcher and code is required before attributing the remaining
difference to assistance.

The acceptance priority was clarified during this debugging pass. UAV request
count is secondary; fewer than five is desirable but is not the main target.
The assisted endpoint run must beat a fresh paired ground-only run on all three
operational metrics:

1. less simulator time;
2. less distance driven;
3. fewer drag interventions.

Do not trade any of those three for a lower request count.

### Fresh paired result after retained fusion and overlap escape

The exact ground-only control is `r11_ground_persistent_escape_rtf1`. It uses
the same code, planner seed 7, Gazebo seed 4207, 1.0x RTF, route direction, and
drag harness as `r11_uav_persistent_escape_full_rtf1`.

| Metric | `counterfactual_uav` | `ground_only` | Assisted delta |
|---|---:|---:|---:|
| Outcome | success | success | — |
| Simulator time | 645 s | **539 s** | +106 s (+20%) |
| Distance driven | 702 m | **631 m** | +71 m (+11%) |
| Drags | 4 | 4 | 0 |
| UAV requests | 6 | 0 | +6 |

This pair fails all three operational acceptance criteria: assistance is
slower, drives farther, and does not reduce drags. It does show that the
infrastructure fixes improved the assisted mode relative to its own `_v2`
regression (8 drags to 4), but that is not sufficient.

The drag geography also changed rather than disappearing. Both modes wedged in
the first dense corridor and each finished with four interventions. The
remaining leading hypothesis is over-constrained semantic fusion: a semantic
request paints the full labeled bush/tree canopy, the planner treats that
raster as hard collision, and the new soft clearance ring expands its routing
influence further. This can add detour distance without removing contact with
the smaller physical bodies. The next comparison must separate semantic
traversability cost from physical occupancy collision/clearance.

### Semantic canopy is no longer physical collision

The product-separation hypothesis was confirmed in code: although
`semantic_traversability` and `canopy_obstacle` were requested separately,
`MapStack.cost` recombined both products' `obstacle` rasters and applied the
same physical collision threshold. A semantic tree or bush therefore regained
hard-collision authority over its full labeled canopy.

Planner fusion now uses:

- semantic UAV `cost` as traversability cost only;
- occupancy UAV `obstacle` as physical hard collision;
- occupancy UAV clearance as the only aerial body-clearance field.

Semantic ground cost and semantic obstacle probability are still superseded
inside semantic aerial coverage. The change therefore removes an incorrect
hard constraint rather than stacking aerial cost on top of the old ground
collision. Ground-only does not execute this path, so
`r11_ground_persistent_escape_rtf1` remains the exact control for the next
assisted run.

The first headless product-split run, `r11_uav_product_split_rtf1`, was stopped
at simulator time 142 s to switch to visual diagnosis. It had driven 140 m,
made three UAV requests, and needed one drag. It is an interrupted run and is
not an endpoint comparison.

The active visual diagnostic is `r11_uav_product_split_gui`, with the same
Route 11 forward, planner seed 7, Gazebo seed 4207, 1.0x RTF, threshold 0.75,
and maturity 1.0 s configuration. It was launched with `--gui` and the chase
camera so contact, stopping, detour, and post-fusion behavior can be inspected
directly. Its result must not be treated as final until `campaign.csv` records
endpoint success.

That GUI diagnostic was stopped at simulator time 602 s after six drags and
ten UAV requests. All ten requests were `mobility_relevant`; neither the
population exposure threshold nor the occupancy counterfactual predicted the
failures before loss of motion. This directly confirms a disconnect between
the original uncertainty population and the drag condition.

### Selected-trajectory trigger

The population evaluator pools distinct swept cells from all 192 sampled
rollouts. That is appropriate for comparing control alternatives, but it can
dilute uncertainty on the one trajectory MPPI actually selects and can center
an ROI on rejected controls. The mobility fallback then reused that unrelated
ROI after contact.

A second, independent evaluator now measures only `MPPI.best_trajectory()`.
It uses independent per-cell maturity state and raises
`selected_trajectory` when at least 20% of the selected swept footprint remains
uncertain. The original 0.75 population threshold and counterfactual remain in
place. If selected-path evidence does not cross 20%, the normal policy keeps
priority.

The mobility fallback is now permitted only when both conditions hold:

1. world-pose displacement has stayed below 0.25 m for five seconds; and
2. the selected trajectory still has a nonempty uncertain ROI.

It can no longer attach a physical stall to uncertainty found only on rejected
rollouts. A focused test verifies that uncertainty on the selected path
triggers while the same uncertainty on another rollout does not.

Request and trigger histories now record simulator time and vehicle position.
The harness writes `intervention_history.json` with simulator time, stuck
position, drag target, route progress, and path length. These make temporal and
spatial trigger/drag correlation measurable instead of inferred from console
line order.

The next experiment is a ground-only shadow run. It evaluates and records the
same trigger policy but does not request or fuse a UAV map, so trigger lead
time can be compared with drag events without the planner trajectory changing
under assistance.

The first shadow, `r11_ground_trigger_shadow_rtf1`, was stopped after its first
drag. The rover became stuck near `(20.65, 6.64)` and was dragged at simulator
time 121.83 s. The first trigger was at 98.63 s at `(20.64, 6.65)`, a 23.20 s
lead relative to the harness intervention, but it was `mobility`, not
`selected_trajectory`. In other words, the trigger occurred at the correct
place only after world displacement had already stopped.

The cause was applying one-second per-cell maturity to a moving three-second
selected trajectory. Upcoming cells continually enter and leave that short
horizon, so the selected-path gate inherited the same moving-frontier failure
it was intended to avoid. The selected path now uses immediate cell
uncertainty and retains the existing two-second source persistence. Population
exposure and its counterfactual keep the original per-cell maturity behavior.

`assistance_trace.jsonl` is now written every 0.5 simulator seconds with
population exposure, selected-path exposure, command speed, wheel speed,
world-mobility duration, trigger relevance, assistance state, position, and
retained-product count. A corrected shadow must show a
`selected_trajectory` episode before `mobility_stalled` becomes true at the
first failure location.

### The first corrected shadow never ran

`r11_ground_action_shadow_rtf1` exited 0.89 simulator seconds in:

```text
TypeError: Object of type bool is not JSON serializable
```

`decision_relevant` is built as `np.any(uncertain) and ...`. Python's `and`
returns its first falsy operand, so on the common path the value was
`np.bool_(False)`, which is not a `bool` and not serializable. Requests had
never hit it because a request only happens when the expression is true, and
then `and` returns the last operand, a real `bool`. Adding a trace line that
records the value on every cycle exposed it immediately. The value is now
coerced at its source, and a test asserts that every diagnostic field is a
`float` or `None`.

## Measuring which uncertainty variable actually predicts a drag

Rather than test one trigger hypothesis per run, the trace now records a broad
set of candidate stuck-predictors at 2 Hz and the correlation is done offline
against the drag times in `intervention_history.json`. There are 78 fields per
sample. The uncertainty, hazard, and extent variables are each measured over
three geometries:

| Geometry | Prefix | What it is |
|---|---|---|
| Population | `pop_` | the sampled rollouts, which is what the Meridian exposure policy consumes |
| Selected path | `path_` | the trajectory the controller chose, a fixed 3 s horizon |
| Forward probe | `probe_` | a fixed 8 m corridor along the intended steering |

The probe exists because the other two are fixed-*time* geometries. The
selected path reaches about 5 m at target speed but collapses below 1 m as the
controller slows, which is exactly when a warning is wanted, so its reach is
smallest at the moment of maximum risk. The probe replays the nominal steering
at target speed for as many steps as 8 m needs, so its warning distance is
constant. `--assistance-probe-m` sets it.

Per geometry: occupancy exposure split into its three causes separately
(`_occ_unknown_frac`, `_occ_ambiguous_frac`, `_occ_variance_frac`), obstacle
probability max/mean, blocked fraction, the semantic equivalents, fused
planner cost, and path extent. Alongside those: the counterfactual's own
internals (`cf_baseline_viability`, `cf_free_viability`,
`cf_occupied_viability`), what the maps say about the cells the vehicle is
standing on (`here_*`), planner cost spread, and wheel-versus-world slip.

Slip is included because wheel odometry stays high while the rover spins its
wheels against vegetation; the gap between wheel speed and world displacement
is a direct physical precursor to a wedge that no uncertainty channel sees.

`tools/analyze_stuck_correlation.py` joins the trace to the interventions and
scores every variable by AUC in three windows:

- `approach`, the 15 s before the stuck timer started. Only separation **here**
  can warn early enough to act on.
- `stalled`, the stuck timer running. Separation only here means the variable
  is a detector, not a predictor.
- `drag`, around the intervention itself.

It also reports the median lead time of the first sustained threshold crossing
before each drag, and how often the same crossing fires during ordinary
driving, because a variable that alarms constantly explains nothing however
well it separates.

## Retained UAV products made every planner tick cost more

Profiling the instrumentation found a larger pre-existing problem. Retaining
UAV products across channels and route regions was correct for evidence, but
`_sample_aerial` rasterised the full query against *every* retained product on
every call, and it was called from `MapStack.cost` and from both evaluators.
The per-tick cost therefore grew linearly with how much of the route had been
answered.

Measured on a 192 x 60 rollout population, products spread along the route:

| Retained products | `MapStack.cost` | `evaluate_assistance` |
|---:|---:|---:|
| 0 | 3.2 ms | 21.8 ms |
| 3 | 14.0 ms | 39.8 ms |
| 6 | 24.0 ms | 55.6 ms |

At six products `evaluate_assistance` alone exceeded the whole 50 ms tick
budget before the planner did any work. This is consistent with the first
attempt at these correlation runs, where the assisted half sat at 89-95% missed
cycles against 14-17% for ground-only.

`_sample_aerial` now rejects a product whose bounding box does not intersect
the query before sampling it, samples all four layers in one index pass instead
of two, and stops once every query point has an answer. After the fix:

| Retained products | `MapStack.cost` | `evaluate_assistance` |
|---:|---:|---:|
| 0 | 3.2 ms | 21.8 ms |
| 6 | 6.4 ms | 26.0 ms |
| 10 | 6.6 ms | 21.3 ms |

A test pins the new loop cell-for-cell against the original one over six
overlapping products and 500 random query points, in both map types.

The diagnostics themselves were also cut from 15.3 ms to 3.4 ms by subsampling
the rollout population to 24 trajectories and not recomputing fused planner
cost over the population, which `evaluate_assistance` already covers.

This fix is necessary but was not sufficient. See the assisted run below.

## Correlation runs: `r11_corr_ground_rtf1` and `r11_corr_uav_rtf1`

Route 11 forward, planner seed 7, Gazebo seed 4207, 1.0x RTF, threshold 0.75,
maturity 1.0 s, path threshold 0.20, probe 8 m, drag harness enabled. The only
difference between them is `--assistance`. Both were interrupted at about 260
simulator seconds and have empty `campaign.csv` files, so neither is an endpoint
comparison. The ground trace contains 493 samples and three interventions; the
assisted trace contains 486 samples, two interventions, and eleven retained UAV
products. They are useful only for selecting a candidate predictor.

### The selected-trajectory channel has no specificity at 0.20

The assisted half made seven requests in the first 149 simulator seconds, then
four more before it was stopped at 260 s:

```text
 #  source                exposure  sim_time_s
 1  ground_semantic_cost     0.250       5.6
 2  ground_semantic_cost     0.463      20.4
 3  ground_semantic_cost     0.338      37.7
 4  ground_semantic_cost     0.319      65.4
 5  ground_semantic_cost     0.317      93.3
 6  lidar_occupancy          0.580     115.9
 7  ground_semantic_cost     0.433     148.8
```

Every one was `action_relevant`. None was `decision_relevant`, none was
`mobility_relevant`. Eleven requests in 260 s is already incompatible with the
fewer-than-five target; no extrapolation is needed.

So the `selected_trajectory` gate did not fix the alignment problem, it
inverted it. The mobility fallback fired only after the rover had already
stopped; the action gate at a 20% threshold fires more or less continuously on
open ground, because a moving three-second horizon always has a frontier in it.
Neither is aligned with the drag condition. The first is too late and the
second is not selective at all.

That also explains why the assisted run sat at 93-95% missed cycles even after
the `_sample_aerial` fix: it accumulated seven retained products in 149 s, and
consecutive requests along the route overlap, so bounding-box rejection has
less to reject. The request rate is the root cause, not the sampling cost.

The ground-only half stayed at 11-19% missed cycles throughout, so the
instrumentation itself is affordable.

### Partial correlation result

The ground-only trace recorded three interventions in three spatially distinct
regions. The corrected offline ranking is:

```text
variable                    approach AUC  threshold  prestall  early  baseline active
probe_occ_exposure                 0.853      0.049      7.4 s    3/3             0.6%
probe_occ_unknown_frac             0.819      0.002      9.9 s    3/3             1.2%
probe_occ_probability_mean         0.859      0.179     -1.1 s    1/3             3.1%
pop_occ_exposure                   0.824      0.044      4.2 s    3/3             3.1%
```

`prestall` is measured relative to the beginning of the harness's 30 s stuck
timer; positive values are genuine warnings, negative values are detectors.
The cleanest interpretation is therefore **uncertain occupancy in the fixed
8 m forward probe**. Obstacle probability has a high AUC but crosses too late.
Semantic uncertainty does not transfer consistently across the traces and is
not the candidate channel for these wedges.

The candidate threshold was then treated as fixed and checked on an independent
ground-only retry, `r11_corr_ground_full_rtf1_retry`. That retry was stopped on
request at simulator time 441 s, so it is also partial. It contains three drag
interventions but only two local map opportunities: the first two stuck poses
are 7 m apart and should be covered by one 25 m UAV product. Against those two
regions, the fixed `probe_occ_exposure > 0.04895` threshold, sustained for two
trace samples, produced:

| Drag region | Stuck position | First warning position | Before intervention | Before stall |
|---|---|---|---:|---:|
| dense outbound corridor | `(25.91, 23.34)` | `(29.15, 14.24)` | 36.11 s | 6.11 s |
| return corridor | `(11.28, 68.64)` | `(9.35, 70.47)` | 32.77 s | 2.77 s |

The warnings occurred 9.66 m and 2.66 m from the eventual stuck poses, both
inside the useful scale of a 25 m local product. The fixed threshold was active
for 3.7% of ordinary samples in the retry and began 1.94 false-alarm episodes
per minute. That specificity is not yet good enough to promote directly to a
request policy, but it is a much better starting point than the continuously
firing selected-path semantic gate.

For comparison, the fixed `probe_occ_probability_mean > 0.1791` threshold did
not warn either region before loss of progress. It fired in only one region,
3.89 s after the stall began. Probability is a contact detector here; occupancy
uncertainty is the channel with prospective value.

### Corrections to the analysis tool

`tools/analyze_stuck_correlation.py` now:

- uses the low tail for variables that predict a drag by falling, such as
  counterfactual viability, instead of ranking them and then testing the wrong
  direction;
- uses a strict threshold so a zero-valued 95th percentile does not alarm on
  every ordinary zero;
- counts contiguous false-alarm episodes rather than every overlapping pair of
  samples in one plateau;
- reports lead relative to both the intervention and the start of the stall;
- groups interventions within 12.5 m into one map opportunity by default;
- adds uncertainty-times-hazard and uncertainty-times-cost diagnostics to test
  whether uncertainty is colocated with a planner constraint.

Run the analysis with:

```bash
conda run --no-capture-output --name rugged-ugv \
  python tools/analyze_stuck_correlation.py \
  runtime/experiments/r11_corr_ground_rtf1 --top 20 --timeline 4
```

### Timing fix pulled after these traces

Commit `88ceb86` was pulled before committing this work. It makes the assistance
state machine use simulator time and adds optional lockstep physics, in which
the controller explicitly advances one 50 ms control period at a time. Every
correlation trace above predates that commit and was free-running. The spatial
finding is still useful, but its exact AUC, lead time, and threshold must be
validated under lockstep before they drive live requests.

### Next validation

1. Complete a lockstep ground-only shadow over Route 11 for seeds 7 and 8. Do
   not fuse UAV data yet; preserve the control trajectory while validating the
   candidate.
2. Start with `probe_occ_exposure > 0.04895`, require at least two consecutive
   samples, and report drag-region recall, lead before the stall, baseline
   active fraction, and false-alarm episodes per minute. The threshold is a
   measured candidate, not a new default.
3. Reduce the roughly two false episodes per minute before enabling requests.
   Longer persistence or an occupancy-specific planner-pressure condition is
   preferable to mixing semantic uncertainty back in.
4. If the predictor survives, use its uncertain probe cells to center a
   `lidar_occupancy` request. Then compare assisted and ground-only lockstep runs
   on endpoint success, simulator time, distance, and drag regions. Request
   count remains secondary to progress and intervention reduction.

The next shadow command should include `--lockstep`:

```bash
./scripts/run_experiment.sh \
  --run-id r11_corr_ground_lockstep_s7 \
  --rtf 1 \
  --routes Route-11 \
  --directions forward \
  --cycles 1 \
  --seed 7 \
  --assistance ground_only \
  --lockstep \
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
