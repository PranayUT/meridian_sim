# Rugged UGV simulation

This project provides a Conda-managed **Gazebo Harmonic** environment, a four-wheel skid-steer UGV, procedural rugged terrain, IMU / 3D lidar / NavSat sensors, and a Gazebo-native waypoint-and-obstacle-avoidance demo.

When `scripts/run_autonomy.sh` is running, the Gazebo GUI shows the GPS route as
a green line, the source GPS points as yellow dots, and the current best MPPI
rollout as a magenta line with pink sample points. These markers are GUI-only,
so they do not contaminate the UAV camera image. The planner also reads the
terrain GeoTIFF and treats grades above 42% as impassable.

The autonomy launcher also opens **UGV live maps**. Lidar and front-camera
semantic maps remain vertically stacked; each row shows the map value and its
cell-level uncertainty side by side. Both use Meridian Drive's 40 m square,
0.25 m grid. The blue arrow is the rover; cyan rays on the semantic panels show
the camera field of view. Red semantic cells are brush, tree, or rock posterior
probabilities high enough to be physical collisions for MPPI. Close the map window without stopping autonomy, or
suppress it with `--no-map-viewer`.

## Install and run

```bash
./scripts/setup.sh
./scripts/run_sim.sh
```

`run_sim.sh` enters the `rugged-ugv` Conda environment automatically when
Gazebo is not already on `PATH`. Activating the environment yourself is still
useful for the inspection commands below.

The launcher prints and fixes the Gazebo random seed, defaulting to `4207`,
and starts the GUI camera in a chase view that follows `hill_rover`. Override
the seed with `./scripts/run_sim.sh --seed 123` or `GZ_SIM_SEED=123`.

In a second activated terminal, start the Gazebo-native Meridian Drive stack:

```bash
source scripts/env.sh
./scripts/run_autonomy.sh
```

To run the GP-Navigation baseline against the same route, localization, LiDAR,
semantic maps, and Ackermann rover, use:

```bash
./scripts/run_gp_navigation.sh
```

This is a ROS-free adaptation of the ICRA 2024 implementation. It retains the
sparse-GP elevation and uncertainty map, geometric traversability calculation,
local RRT* planner, 5 m rolling planning radius, and 2 Hz replanning rate. The
upstream ROS Noetic action servers and differential-drive waypoint follower are
replaced by this simulator's Gazebo Transport boundary and an Ackermann
pure-pursuit follower. Use `--gp-iterations`, `--gp-inducing-points`,
`--gp-radius`, and `--gp-traversability-limit` to change baseline parameters.
The traversability cutoff defaults to `0.6` here instead of the upstream `0.3`;
the upstream documentation identifies it as environment-dependent, and `0.3`
disconnects free space on this substantially rougher terrain. The shared hard
obstacle and 42% grade constraints remain active.

When `paths/` contains one KMZ, KML, or Meridian GPS JSON file, autonomy loads
it automatically. The supplied `paths/Route 11.kmz` projects into the Gazebo
world from the WGS84 datum in the world file. The rover starts at its first
point. Select a file explicitly when the directory contains more than one:

```bash
./scripts/run_autonomy.sh --route-file "paths/Route 11.kmz"
```

The simulation starts running immediately. The autonomy process runs the
Meridian Drive velocity-model MPPI controller. It follows a continuous local
ENU route and plans around lidar and aerial-map obstacles. Stop it with
Ctrl-C. It publishes a final zero velocity.

The lidar mapper is the ROS-free mapping core from Meridian Drive. It retains
ten sweeps, keeps longer ground-height history, and publishes unknown (`-1`),
ground (`0`), tall/porous (`50`), and solid (`100`) cells through
`runtime/ground_maps.npz`. MPPI consumes this retained grid.
The same snapshot contains the Bernoulli occupancy probability, posterior
variance, support, and age used by Meridian's `MapEvidence` representation.

The semantic mapper uses co-located Gazebo segmentation and depth cameras in
place of the GOOSE neural-network inference stage. Gazebo supplies labels only
for rendered, unoccluded pixels; the mapper back-projects pixels from 0.3 m to
8 m into the same grid and applies Meridian's GOOSE policy: soil `0.20`, grass
`0.40`, brush `0.70`, rocks `0.80`, and trees `0.95`. Points more than 0.5 m
above the rover base are filtered to keep overhead canopy from marking the
ground below. Camera topics are:

```text
/model/hill_rover/semantic/labels_map
/model/hill_rover/semantic/colored_map
/model/hill_rover/depth
```

The old reactive controller remains available for subsystem checks:

```bash
./scripts/run_waypoint_demo.sh
```

Useful inspection commands:

```bash
gz topic -l
gz topic -e -t /model/hill_rover/odometry
gz topic -e -t /model/hill_rover/imu
gz topic -e -t /model/hill_rover/navsat
gz topic -e -t /model/hill_rover/lidar
```

## Bringing in a real 3D scan

Use two representations of the same survey rather than one huge photogrammetry mesh:

1. Georeference, crop, remove transient objects, and decimate the scan in CloudCompare, PDAL, or Blender. Keep a local ENU origin near the test area so coordinates remain numerically small.
2. Export a **16-bit single-channel PNG heightmap** for driveable ground. Use `2^n + 1` pixels per side (for example 1025 or 2049). Update the visual `<uri>`, `<size>`, and `<pos>` in `models/hill_terrain/model.sdf`. For this Conda build, use a separately decimated ground mesh for collision; the supplied generator creates `collision.obj` from the same elevations because the bundled default physics backend does not reliably construct heightmap collision.
3. Export overhangs, banks, buildings, and rocks that cannot be represented by a 2.5D heightmap as decimated `.glb`, `.dae`, or `.obj` visual meshes. Give them separate, coarse convex or primitive collision geometry.
4. Keep visual mesh chunks moderate (roughly 50–100 m tiles). Use lower-detail collision meshes and disable collision entirely for grass, leaves, and distant scenery.
5. Add trees as explicit `<include>` instances in `worlds/hill_country.sdf`. The supplied tree uses trunk-only collision. For hundreds of plants, combine visuals into batches or use GPU instancing; thousands of separate SDF models will make startup and updates expensive.

`tools/generate_terrain.py` is only a deterministic placeholder. Re-running it restores the demo terrain.

The setup defaults `GZ_IP` to `127.0.0.1` for dependable same-machine topic discovery. Set `GZ_IP` to the appropriate interface address before sourcing `scripts/env.sh` if the simulator and autonomy node will run on different machines.

## Vehicle and autonomy tuning

- Vehicle dimensions, mass, friction, wheel torque, and sensor specifications are in `models/hill_rover/model.sdf`.
- Terrain friction is in `models/hill_terrain/model.sdf`. Use several surface patches or a custom system if wet soil, gravel, and grass need distinct traction.
- Paint dense grass, bushes, and trees directly over the orthophoto with:

  ```bash
  ./scripts/paint_vegetation.sh
  ```

  Left-drag paints the selected plant type, right-drag erases, the mouse wheel
  zooms around the cursor, and middle-drag pans. Ctrl+wheel changes brush
  radius. Density controls the spacing throughout every
  painted region. **Save vegetation to Gazebo** writes an editable mask to
  `maps/vegetation_paint.npz` and creates one static batched model under
  `models/painted_vegetation/`. Restart Gazebo after saving. Select the legacy
  removal checkbox if the old individually placed vegetation should be removed.
- Change the route by repeating `--waypoint X Y`, for example:

  ```bash
  ./scripts/run_autonomy.sh --waypoint 10 0 --waypoint 12 10 --waypoint 0 0
  ```

### UAV assistance

The interactive world publishes a fixed top-down camera at 2 Hz:

```text
/uav/top_down/image
/uav/top_down/camera_info
```

The camera is fixed above world `(0, 40)` at Z = 220 m and looks straight
down. Its horizontal field of view is 90 degrees. The terrain is near Z = 20 m,
so the image covers about 400 m by 400 m. This includes Route 11. Use the
camera-info topic and the model pose for the final pixel-to-world projection.

Select one of the Meridian Drive experiment policies when autonomy starts:

```bash
./scripts/run_autonomy.sh --assistance ground_only
./scripts/run_autonomy.sh --assistance greedy_uav
./scripts/run_autonomy.sh --assistance counterfactual_uav
./scripts/run_autonomy.sh --assistance explore_then_drive
./scripts/run_autonomy.sh --assistance always_on_uav
```

The UAV modes default to `--uav-source ground_truth`. This is the experiment
surrogate: after a counterfactual request, autonomy constructs an exact 25 m by
25 m local map centered on the selected uncertainty region. Its occupancy is
rasterized from the seeded Gazebo bush/tree bodies and fixed rocks. Its
semantic layer uses the same GOOSE IDs as the ground camera (`31` soil, `50`
grass, `17` brush, `28` tree, and `40` rock). The result is written atomically
to `runtime/uav_map.npz`; `occupancy` and `semantic_label` retain the two raw
ground-truth products, while `obstacle`, `cost`, and `uncertainty` are their
planner-ready forms.

Autonomy also writes the request atomically to `runtime/uav_request.json`. It
contains the selected source, matching requested product, world-frame
`roi_xy`, uncertainty exposure, and `hold_requested`. Use `--uav-source file`
to disable the simulator producer and attach an external map process. That
process must write a temporary NPZ in the same directory and rename it into
place so autonomy never reads a partial archive.

The small simulator NPZ contract is:

| Field | Shape and type | Meaning |
| --- | --- | --- |
| `cost` | H×W float | Terrain cost from 0 to 1. Optional when `obstacle` exists. |
| `obstacle` | H×W float | Occupancy probability from 0 to 1. Optional when `cost` exists. |
| `uncertainty` | H×W float | Variance or normalized uncertainty from 0 to 1. Optional. |
| `origin_xy` | 2 float values | World coordinates of the lower-left raster edge. |
| `resolution` | float | Square cell size in metres. |
| `sequence` | integer | Producer sequence for status and experiment records. |

Ground-truth simulator results additionally contain `occupancy` (H×W uint8,
0 or 100) and `semantic_label` (H×W uint8 GOOSE ID).

Row zero is the south edge. Columns increase toward positive X. Rows increase
toward positive Y. The loader also accepts the existing Meridian Drive UAV NPZ
contract when its frame is `world`, `map`, or `odom`, or its CRS is `local`,
`enu`, or `world`. Geographic rasters must be changed to local ENU coordinates
by the map generator before upload. The autonomy status is written to
`runtime/autonomy_status.json`.

The uncertainty path follows Meridian Drive's counterfactual boundary. It
evaluates occupancy and semantic evidence separately over the swept rollout
footprints. Occupancy cells are uncertain at variance 0.04 or ambiguous
probability 0.20–0.80. Semantic assistance uses Meridian's posterior traversal-
cost variance with a 0.04 cutoff; the entropy/support/age composite remains a
viewer diagnostic only. Deterministic Gazebo occupancy observations carry
three effective observations for assistance confidence without weakening the
repeated-sweep requirement for a physical `SOLID` classification.
`--uav-uncertainty-threshold` controls the fraction of touched cells that must
be uncertain and defaults to 0.20. Crossings must persist for 2 seconds. A
higher value reduces requests only when exposure falls between the two values;
fully unknown rollout footprints score 1.0 and therefore remain triggers.
Before entering that source-level persistence timer, the same unresolved swept
cells must remain relevant for `--mapping-uncertainty-maturity` seconds
(default 1.0); newly encountered frontier cells cannot inherit one another's
age.

`ground_only` records the same signal and never requests help. `greedy_uav`
requests without holding. `counterfactual_uav` also applies the occupancy
free-versus-occupied viability test, makes a smooth stop, confirms the rover is
settled, requests the source selected by the rollout evaluation, admits the
new map, waits for fusion settling, and resumes. As in Meridian Drive, it
permits two requests for one 5 m region and source, then holds for operator
action. Unlike the field policy's expanding retry, every simulator response
remains exactly 25 m by 25 m.

`explore_then_drive` is the naive/exhaustive baseline used in the same logical
family as Delmerico et al. (2017) and Zhang et al. (2022): the UAV first surveys
the whole route-relevant area, fuses the resulting traversability map, and only
then does the UGV start. The simulator supplies the route-wide map before the
first control tick. The experiment harness models a lawnmower flight over the
route bounding box plus a half-swath margin and adds that flight time to the
UGV simulator time; it does not run UAV dynamics in Gazebo.

`always_on_uav` is an optimistic simulator baseline. Exact physical occupancy
and semantic traversability covering the full route plus a 12.5 m margin are
fused before the first control tick and remain available throughout the run.
It makes no reactive requests and never holds the rover.

UAV resource accounting is analytical. The fixed horizontal speed is 5 m/s,
the current PX4 `MPC_XY_CRUISE` default for autonomous modes including
missions. With the default 25 m map width, one counterfactual assist is charged
as one map-width observation transect, or 5 seconds, so its flight time is
`request_count * 5 s` and that service time is added to total navigation time.
Greedy and always-on assistance are concurrent with driving, so their UAV
flight time equals UGV navigation time and is not added again. For
explore-then-drive, `total_navigation_time_s = uav_flight_time_s + sim_time_s`.
Takeoff, landing, and depot transit are excluded because the experiment does
not specify a UAV depot.

For example, run Route 11 at the 20% starting point and inspect its request
count and channel scores before changing the policy:

```bash
./scripts/run_experiment.sh --routes Route-11 --directions forward \
  --cycles 1 --assistance counterfactual_uav \
  --uav-uncertainty-threshold 0.20
```

`runtime/autonomy_status.json` and the campaign CSV both record the request
count and threshold. Every request is also retained in
`runtime/uav_request_history.json`, including its selected source, ROI, and
exposure at the trigger.

The native stack intentionally leaves out ROS, hardware drivers, GNSS
localization, and the field deadman. Gazebo's world-pose stream supplies
simulation ground-truth position and heading; wheel odometry supplies speed.
The planner keeps Meridian Drive's velocity-command bicycle model, receding
horizon sampling, route costs, obstacle costs, and UAV assistance modes. The
Gazebo rover uses physical front-wheel Ackermann steering with the same 0.29 m
wheelbase and limits the inside front wheel to 45 degrees. The corresponding
virtual bicycle angle is 32.2 degrees because the two front wheels follow
different radii, giving a 0.460 m tightest centre turn radius. The transport boundary converts that bicycle angle to the
yaw-rate command accepted by Gazebo's Ackermann plugin and enforces the same
lateral-acceleration limit as the planner rollout.
Gazebo's Ackermann plugin applies one speed limiter to both its linear and its
angular channel, so every `*_velocity` and `*_acceleration` bound in the model
is sized for yaw and the linear envelope is enforced on the published command in
`gazebo_node.py` instead. The MPPI model independently prevents reverse linear
commands, so the symmetric velocity window costs nothing on that axis. The
plugin also derives its tightest turn as `wheel_base/sin(steering_limit)` rather
than the bicycle model's `wheel_base/tan(steer_max)`, so `steering_limit` carries
an `asin(tan(...))` pre-compensation to reach the intended 0.460 m radius.

## Headless operation

For CI or batch tests, omit the GUI:

```bash
./scripts/run_sim.sh -s
```

The GPU lidar still needs a working render device. On a machine without GPU/EGL support, change `gpu_lidar` to `lidar` in the rover SDF or run with software rendering (slower).

Run the GPU-independent physics, sensor, autonomy, and motion check with:

```bash
./scripts/smoke_test.sh
```

Gazebo is throttled to real time by the world's `real_time_factor`. `--rtf`
overrides that per run without editing the SDF:

```bash
./scripts/run_sim.sh --rtf 3
```

The autonomy node paces itself on simulator time rather than wall time, so it
keeps its nominal rate in-sim as the world speeds up, and prints a warning if
the planner starts missing cycles. Measured on this machine: 2x is clean, 3x
costs 1-2 late cycles per 100, and 5x misses about 40 per 100. Lower
`--samples` or raise `max_step_size` to push it further.

## Route experiments

`scripts/run_experiment.sh` drives a campaign and records one CSV row per
trial. It runs each named route forward and reversed, cycling the seed. The
default is ground-only assistance; pass `--assistance counterfactual_uav` for
the request-driven ground-truth UAV condition, `--assistance
explore_then_drive` for the sequential exhaustive-survey baseline, or
`--assistance always_on_uav` for the route-wide optimistic baseline:

```bash
./scripts/run_experiment.sh --cycles 5 --rtf 3
```

Select GP-Navigation for the same campaign harness with `--planner
gp_navigation`. For example, a single forward Route 11 ground-only trial is:

```bash
./scripts/run_experiment.sh --routes Route-11 --directions forward \
  --cycles 1 --seed 7 --rtf 1 --planner gp_navigation \
  --assistance ground_only
```

Defaults to Route-11, Route-12, and Route-13 from `paths/from_truck/` for six
trials per cycle; reversed copies are generated into `runtime/experiment_routes/`.
When the rover makes no progress for `--stuck-s` seconds of simulator time the
harness drags it `--drag-m` further along the route and counts an intervention,
so one bad spot cannot silently end a run. The drag follows the route arc
rather than the straight bearing to the next waypoint, because that bearing
usually points through whatever is wedging the rover. Interventions are applied
from the harness, not from `gazebo_node`, so the planner under test stays the
same code that runs against Meridian Drive.

Recorded per trial: outcome, success, UGV simulator seconds, total navigation
seconds, analytical UAV flight seconds, UAV sorties, survey distance, the
fixed speed and per-assist duration used, wall seconds, distance driven, route
length, interventions, how many of those interventions the rover actually
drove clear of, assistance mode, uncertainty threshold, cell-maturity window,
and UAV request count. Dividing simulator by wall seconds gives the speed-up the run
really achieved, which is the number to trust when tuning `--rtf` on new
hardware.

Everything a run writes goes to `runtime/experiments/<run-id>/`: the campaign
CSV, one autonomy log per trial, the Gazebo log, and that run's ground map and
status files. `--run-id` names it and defaults to a timestamp. A `--run-id` may
contain one `/` to nest a trial under a campaign directory, which is how
`run_campaign.sh` groups its trials; the Gazebo partition flattens the `/` to
an underscore.

### Drag placement

A drag that lands the rover in the next bush has not helped it. These routes
thread dense vegetation — Route-11 has 255 collision sites within 0.8 m of its
own path — so the harness loads the baked vegetation collision mesh into a
0.25 m occupancy grid (cached at `runtime/obstacle_cells.npy`) and walks forward
along the route and then laterally beside it until it finds a spot with a
body-width of clearance and drivable grade; it never silently falls back to an
occupied target. The earlier forward-only clearance search freed 17/20 wedged
rovers on Route-11 and 15/17 on the held-out Route-12 (86% combined), versus
9/19 for a fixed 3 m hop. The lateral search covers the remaining failure mode:
a long occupied centerline belt with clear ground beside it. The Route-11 seed
7 validation resolved all 5 interventions and completed the route.

### Rounds with randomised obstacles

`scripts/run_campaign.sh` is the batch entry point. It sweeps rounds, and a
round is one seed used for both the obstacles and the planner across every
route, so a route's variation between rounds is variation in both:

```bash
./scripts/run_campaign.sh --rounds 5 --jobs 4 --rtf 3
```

That is 5 rounds x 3 routes x 2 directions = 30 trials, four simulators at a
time. Round `r` uses seed `--seed + r` (default 7). Every (round, route,
direction) is an independent trial with its own simulator, so rounds overlap
and a slow route never holds up the rest of its round.

One invocation writes one directory. Every trial lands in
`runtime/experiments/<campaign>/s<seed>_<route>_<direction>/`, alongside a
`failures.txt` listing any trial that did not finish (removed when none did),
so a campaign is a single thing to inspect, archive, or delete. `--campaign`
names the directory and defaults to a timestamp.

Each round's obstacles are baked by `tools/make_vegetation.py` into
`runtime/vegetation/seed-<n>/`, from the painted masks in
`maps/vegetation_paint.npz`. The painted regions, the plant lattice, and the
plant counts are fixed; the seed only drives per-plant jitter, scale, and yaw.
Between two seeds the 1,307 bushes and trees keep their count and move a median
0.30 m (p90 0.72 m), which is enough to open or close a marginal gap without
moving where the route is drivable. Baking a variant takes about 4 seconds and
167 MB, and an existing one is reused rather than rebuilt.

The variant directory goes on `GZ_SIM_RESOURCE_PATH` ahead of `models/`, so
`model://painted_vegetation` resolves to that round's obstacles without editing
the world, and campaigns with different obstacles can share a machine. The
harness grades its drag targets against the same variant's collision mesh, with
the occupancy grid cached per variant — pointing it at the wrong mesh would aim
the rover at cells another round left clear. Each trial's `veg_seed` column
records which variant it drove.

### Running many campaigns at once

Each campaign gets its own `GZ_PARTITION` (derived from the run id), so several
can share a machine without their topics, services, or `set_pose` calls
reaching each other. `run_campaign.sh` relies on this, and
`scripts/run_experiment.sh` can be driven the same way by hand:

```bash
for i in 1 2 3 4; do
  ./scripts/run_experiment.sh --run-id batch$i --seed $((i * 100)) --cycles 5 --rtf 3 &
done
wait
python tools/summarize_experiments.py
```

Pass `--veg-root runtime/vegetation/seed-<n>` to give a hand-driven campaign a
baked variant; without it the committed `models/painted_vegetation` is used.

Two concurrent campaigns each held 2.83x on the development machine, so a
larger box should scale until GPU sensor rendering saturates. Give each run a
different `--seed` or the cycles will repeat the same trials.

`tools/summarize_experiments.py` pools every `runtime/experiments/*/campaign.csv`
and `runtime/experiments/*/*/campaign.csv`, so it picks up both standalone runs
and campaign directories. It reports success rate, achieved speed-up, per-route
means, the path-length to route-length ratio, intervention counts, and a list of
failures. Pass explicit paths to summarise a subset, such as
`runtime/experiments/<campaign>/*/campaign.csv` for one campaign.
