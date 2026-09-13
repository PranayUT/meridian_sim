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
```

The default exchange directory is `runtime/`. Autonomy writes the current map
request atomically to `runtime/uav_request.json`. The request has a world-frame
`roi_xy` polygon, requested products, uncertainty exposure, and a
`hold_requested` value. Your map process must write its result atomically to
`runtime/uav_map.npz`. Write a temporary file in the same directory and rename
it into place so autonomy never reads a partial archive.

The small simulator NPZ contract is:

| Field | Shape and type | Meaning |
| --- | --- | --- |
| `cost` | H×W float | Terrain cost from 0 to 1. Optional when `obstacle` exists. |
| `obstacle` | H×W float | Occupancy probability from 0 to 1. Optional when `cost` exists. |
| `uncertainty` | H×W float | Variance or normalized uncertainty from 0 to 1. Optional. |
| `origin_xy` | 2 float values | World coordinates of the lower-left raster edge. |
| `resolution` | float | Square cell size in metres. |
| `sequence` | integer | Producer sequence for status and experiment records. |

Row zero is the south edge. Columns increase toward positive X. Rows increase
toward positive Y. The loader also accepts the existing Meridian Drive UAV NPZ
contract when its frame is `world`, `map`, or `odom`, or its CRS is `local`,
`enu`, or `world`. Geographic rasters must be changed to local ENU coordinates
by the map generator before upload. The autonomy status is written to
`runtime/autonomy_status.json`.

`ground_only` records uncertainty and never requests help. `greedy_uav`
requests maps while it continues to drive. `counterfactual_uav` publishes a
zero command after a selected rollout crosses uncertain space. It resumes when
a new valid NPZ replaces the prior result. It permits two requests for one
5 m region. The second request expands the region by 3 m on each side. It holds
for operator action if the second result does not clear the uncertainty.

The native stack intentionally leaves out ROS, hardware drivers, GNSS
localization, and the field deadman. Gazebo's world-pose stream supplies
simulation ground-truth position and heading; wheel odometry supplies speed.
The planner keeps Meridian Drive's velocity-command bicycle model, receding
horizon sampling, route costs, obstacle costs, and UAV assistance modes. The
Gazebo rover uses physical front-wheel Ackermann steering with the same 0.29 m
wheelbase and limits the inside front wheel to 60 degrees. The corresponding
virtual bicycle angle is 40.7 degrees because the two front wheels follow
different radii. The transport boundary converts that bicycle angle to the
yaw-rate command accepted by Gazebo's Ackermann plugin and enforces the same
lateral-acceleration limit as the planner rollout.
Gazebo's Ackermann plugin applies one speed limiter to both its linear and its
angular channel, so every `*_velocity` and `*_acceleration` bound in the model
is sized for yaw and the linear envelope is enforced on the published command in
`gazebo_node.py` instead. The MPPI model independently prevents reverse linear
commands, so the symmetric velocity window costs nothing on that axis. The
plugin also derives its tightest turn as `wheel_base/sin(steering_limit)` rather
than the bicycle model's `wheel_base/tan(steer_max)`, so `steering_limit` carries
an `asin(tan(...))` pre-compensation to reach the intended 0.337 m radius.

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
trial. It runs each named route forward and reversed, cycling the seed, with
ground-only assistance:

```bash
./scripts/run_experiment.sh --cycles 5 --rtf 3
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

Recorded per trial: outcome, success, simulator seconds, wall seconds, distance
driven, route length, interventions, and how many of those interventions the
rover actually drove clear of. Dividing simulator by wall seconds gives the
speed-up the run really achieved, which is the number to trust when tuning
`--rtf` on new hardware.

Everything a run writes goes to `runtime/experiments/<run-id>/`: the campaign
CSV, one autonomy log per trial, the Gazebo log, and that run's ground map and
status files. `--run-id` names it and defaults to a timestamp.

### Drag placement

A drag that lands the rover in the next bush has not helped it. These routes
thread dense vegetation — Route-11 has 255 collision sites within 0.8 m of its
own path — so the harness loads the baked vegetation collision mesh into a
0.25 m occupancy grid (cached at `runtime/obstacle_cells.npy`) and walks forward
along the route until it finds a spot with a body-width of clearance and
drivable grade. Measured against deliberately wedged rovers at real obstacle
sites: a fixed 3 m hop freed 9/19 (47%), clearance-aware placement freed 17/20
on Route-11 and 15/17 on the held-out Route-12 (**86% combined**). In a full
campaign the in-situ rate is lower, around 72%, because repeated drags in one
bad patch each count separately.

### Running many campaigns at once

Each campaign gets its own `GZ_PARTITION` (derived from the run id), so several
can share a machine without their topics, services, or `set_pose` calls
reaching each other:

```bash
for i in 1 2 3 4; do
  ./scripts/run_experiment.sh --run-id batch$i --seed $((i * 100)) --cycles 5 --rtf 3 &
done
wait
python tools/summarize_experiments.py
```

Two concurrent campaigns each held 2.83x on the development machine, so a
larger box should scale until GPU sensor rendering saturates. Give each run a
different `--seed` or the cycles will repeat the same trials.

`tools/summarize_experiments.py` pools every `runtime/experiments/*/campaign.csv`
and reports success rate, achieved speed-up, per-route means, the path-length
to route-length ratio, intervention counts, and a list of failures. Pass
explicit paths to summarise a subset.
