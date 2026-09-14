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

The current focused test command passes 27 tests:

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

The focused suite now passes 36 tests:

```bash
conda run --no-capture-output --name rugged-ugv \
  python -m unittest discover -s autonomy -p 'test*.py'
```

The current end-to-end run is `r11_uav_persistent_escape_full_rtf1`, Route 11
forward, planner seed 7, Gazebo seed 4207, 1.0x RTF, threshold 0.75, and
maturity 1.0 s. At simulator time 224 s it has made one retained semantic UAV
request and needed two drags. This is an in-progress observation, not a final
result. The ground-only `_v2` control completed with three drags.

The target for accepting this work is an assisted endpoint run with fewer than
three drags under the paired configuration, followed by a fresh deterministic
ground-only control on the same launcher and seeds.

0. Separate the planner's tick budget from the comparison before running more
   paired trials. At a 50%+ miss rate the assisted mode is being measured
   partly on its own evaluation cost. Try `--assistance-period` above zero, or
   profile `MapStack.cost`, and confirm the miss rate is comparable across
   modes before attributing a drag difference to the aerial evidence.
1. Repeat the `r11_cf_rtf1` / `r11_ground_rtf1` pair over several seeds. One
   paired run cannot separate a 22% time saving from trajectory noise, and the
   drag counts (7 against 3) are small enough that one extra wedge moves them.
2. Repeat the successful 0.75 configuration over multiple vegetation/planner
   seeds and Route 11 reverse. Do not promote a new simulator default based on
   one stochastic forward run.
3. Add periodic exposure and counterfactual viability metrics. Request history
   records triggers only; it does not yet record near-misses.
4. Request count is now set by the counterfactual viability thresholds, not by
   exposure: every `r11_cf_rtf1` request was admitted below the 0.75 gate.
   Raising the exposure threshold further cannot reduce the count. Increase
   cell maturity conservatively (for example 1.5 or 2.0 s) first, and only then
   consider the viability thresholds. Keep one variable fixed per comparison.
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

Repeat the paired comparison on a second seed. Run both halves; a single
assisted run has no control to be read against.

```bash
for MODE in counterfactual_uav ground_only; do
  ./scripts/run_experiment.sh \
    --run-id "r11_${MODE}_s8_rtf1" \
    --rtf 1 \
    --routes Route-11 \
    --directions forward \
    --cycles 1 \
    --seed 8 \
    --assistance "${MODE}" \
    --uav-uncertainty-threshold 0.75 \
    --mapping-uncertainty-maturity 1.0
done
```

The two halves may run concurrently; each takes its own `GZ_PARTITION` from its
run ID. On a 12-core machine the pair used about 3 cores total, so the
concurrency did not itself cause the missed control cycles.

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
