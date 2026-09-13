# Rugged UGV Simulation — Local Setup

## Prerequisites

- Linux (or WSL2) with Miniconda/Anaconda installed.
- A GPU with working OpenGL/EGL and DRI device permissions (you're in the
  `video` or `render` group) if you want the interactive GUI — check with
  `ls -l /dev/dri` and `groups`. Software rendering (llvmpipe) works for
  headless smoke tests but is too slow for dragging objects around live.
- This project is not currently a git repo, so copy the whole `teaming_sim/`
  directory to the target machine (rsync/scp/tar) rather than `git clone`.

## Install

```bash
./scripts/setup.sh
conda activate rugged-ugv
source scripts/env.sh
```

`setup.sh` builds the `rugged-ugv` Conda env from the packages also listed in
`environment.yml` (gz-sim8, gz-launch7, numpy, pillow). The real terrain
(`models/hill_terrain/meshes/terrain.tif` + `collision.obj`) is already
committed, so no GDAL, `.vrt`, or network access is needed just to run the
sim — only if you want to re-crop the DEM (see below).

## Run

```bash
./scripts/run_sim.sh          # GUI (default) — use this for interactive editing
./scripts/run_sim.sh -s       # headless, for CI/batch
```

In a second activated terminal:

```bash
source scripts/env.sh
./scripts/run_autonomy.sh
```

This starts the Gazebo-native Meridian Drive MPPI stack. See the UAV assistance
section in `README.md` for the top-down camera and the map exchange contract.

## Move objects interactively

1. Launch with the GUI (`./scripts/run_sim.sh`, no `-s`).
2. Use the **Transform Control** toolbar (translate/rotate/scale gizmo) to
   drag `tree_*`, `rock_*`, or `hill_rover` into place on the real terrain.
3. Open the **Component Inspector** panel on the selected entity to read
   back its exact resulting pose.
4. Hand-edit the matching `<pose>` in `worlds/hill_country.sdf` — the GUI
   does not save pose edits back to the SDF file automatically.

## Notes on this world

- Terrain is real 1 m/pixel lidar-derived elevation (USGS 3DEP, Central
  Texas Hill Country) covering the area in `map_bounds.txt`. The visual
  heightmap loads the GeoTIFF DEM directly (`meshes/terrain.tif`); the
  collision mesh is a decimated version of the same data
  (`meshes/collision.obj`, built by `tools/build_hill_terrain_collision.py`).
- `worlds/hill_country.sdf` and `worlds/headless_smoke.sdf` both set
  `<spherical_coordinates>` to the real site (30.326139, -98.148264,
  230.112 m elevation) so NavSat output is geographically accurate.
- Tree/rock/rover poses in `worlds/hill_country.sdf` still reflect the old
  synthetic 200×200 m demo terrain and have not been repositioned for the
  real terrain yet — that repositioning is what the interactive GUI workflow
  above is for.
- To re-crop the DEM (different bounds, larger area, etc.) you need
  `maps/TX_Central_B1_2017.vrt`, GDAL (`gdal_translate`/`gdaltransform` —
  already present in the `rugged-ugv` env via `gz-common`'s GDAL dependency),
  and network access, since the `.vrt` streams tiles from USGS's S3 bucket
  rather than storing them locally.

## Moving to a bigger machine for batch runs

Since this is not a git repo, copy the directory. Three things to get right:

- **`models/` must travel in full (175 MB).** `painted_vegetation/meshes/` came
  from the interactive painting tool, and `hill_terrain/meshes/terrain.tif` and
  `collision.obj` are a crop of a real DEM. Neither can be rebuilt from what is
  in the repo.
- **Do not let `terrain.tif` go missing.** `setup.sh` and `run_sim.sh` both run
  `tools/generate_terrain.py` when it is absent, and that script writes a
  *synthetic demo* heightmap and overwrites the real `collision.obj` with it.
  Nothing fails and nothing warns — you would simply be driving a different
  world, and results would not be comparable with anything measured here.
  Checksum it on both ends before trusting a batch.
- **`maps/` (1.5 GB) is optional.** It is the DEM and vegetation-paint source,
  needed only to re-crop the terrain or re-paint vegetation. An experiment box
  does not need it. `runtime/` is disposable output; exclude it too.

```bash
rsync -a --exclude runtime/ --exclude maps/ --exclude '__pycache__' \
  teaming_sim/ user@big-machine:~/teaming_sim/          # ~180 MB

# same terrain on both ends, or the results are not comparable
md5sum models/hill_terrain/meshes/terrain.tif models/hill_terrain/meshes/collision.obj
ssh user@big-machine 'cd teaming_sim && md5sum models/hill_terrain/meshes/terrain.tif models/hill_terrain/meshes/collision.obj'

ssh user@big-machine 'cd teaming_sim && ./scripts/setup.sh'
```

Then confirm the new box before committing to a long campaign:

```bash
./scripts/smoke_test.sh                                   # physics + control, no GPU needed
./scripts/run_experiment.sh --cycles 1 --routes Route-13  # ~3 min, exercises everything
```

### Finding the speed ceiling on new hardware

Two separate limits, and they move independently:

1. **Gazebo** — GPU sensor rendering. Compare the `-z` target against the
   achieved speed-up that the campaign summary prints.
2. **The planner** — MPPI costs about 7.3 ms per cycle at `--samples 128` on a
   50 ms simulator budget, so it alone could sustain roughly 6.9x; mapping and
   message decoding eat the rest. The autonomy node prints
   `Controller missed N of the last 100 cycles` when it can no longer keep up.

On the development machine 2x was clean, 3x cost 1-2 late cycles per 100, and
5x missed about 40 per 100. Raise `--rtf` until those warnings appear, then
back off one step. A faster CPU moves limit 2; a faster GPU moves limit 1.

Beyond that, scale **wide** rather than fast: several campaigns in parallel,
each with its own `--run-id` and `--seed`, beats one campaign at an `--rtf`
high enough to starve the controller. See "Running many campaigns at once" in
`README.md`. Degrading the controller to gain wall-clock speed changes the
thing you are measuring; running more independent seeds does not.

## SSH / remote access

Running the Gazebo GUI over plain `ssh -X` (X11 forwarding) is often
unreliable for Ogre2 — indirect GLX frequently can't provide the OpenGL
context it needs, and even when it does work it's too slow for interactive
dragging. A remote-desktop protocol that renders on the remote GPU and
streams pixels (VNC — TigerVNC/x11vnc — or NoMachine) works much better for
this than X11 forwarding. If the remote machine has no GPU at all, headless
smoke-testing still works (`gz sim --headless-rendering`, falls back to
software rendering) but interactive placement will be too slow to be
practical — do that part on a machine with a real GPU and display.
