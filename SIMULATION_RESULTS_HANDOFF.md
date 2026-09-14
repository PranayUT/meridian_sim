# Simulation results handoff for the real-world experiment analysis

Snapshot date: **2026-09-14**

This document summarizes the simulation design and the currently available results in:

- `runtime/experiments/ground_all_final`
- `runtime/experiments/uav_all_final`
- `runtime/experiments/always_on_all_final`

It is intended as a handoff for preparing paper figures and relating the simulation to the real-world experiments. The results are still partial: the intended design has 60 trials per policy, but only 59 ground-only and 58 reactive-UAV trials have final CSV rows. Treat every number below as a descriptive snapshot until the three missing trials are rerun and the final statistical analysis is frozen.

## One-slide summary

The experiment compares three information policies while a Meridian Drive MPPI controller follows three routes in both directions over ten randomized vegetation/planner seeds.

| Policy / folder | Completed rows | Endpoint successes | Harness interventions | Mean interventions/trial | Mean UAV requests/trial | Mean simulator time among successes |
|---|---:|---:|---:|---:|---:|---:|
| Ground only (`ground_all_final`) | 59/60 | 56/59 (94.9%) | 148 | 2.51 | 0 | 330.0 s |
| Reactive UAV (`uav_all_final`) | 58/60 | 54/58 (93.1%) | 117 | 2.02 | 6.09 | 285.3 s |
| Always-on UAV (`always_on_all_final`) | 60/60 | 60/60 (100%) | 101 | 1.68 | 0 reactive requests | 261.6 s |

The defensible descriptive reading is:

- Endpoint success is already high for ground-only and reactive assistance. Reactive UAV did **not** improve aggregate endpoint success in this partial snapshot (93.1% versus 94.9% using all available rows).
- Reactive UAV used fewer harness interventions on average than ground-only (2.02 versus 2.51), and successful reactive runs were faster on average (285 versus 330 simulator seconds). These are promising trends, not yet inferential conclusions.
- Always-on aerial knowledge is the optimistic reference: it completed all 60 trials, used the fewest interventions, and had the lowest successful-run time. It is not a deployable policy and its zero request count means that information was preloaded, not that no UAV information was used.
- Route 11 is the difficult route and drives most of the separation. It is also where reactive assistance incurs most of its requests.

## What was actually compared

### Experimental unit and matching

The intended factorial design is:

```text
10 seeds (7 through 16)
× 3 routes (Route-11, Route-12, Route-13)
× 2 directions (forward and reverse)
= 60 trials per policy
```

Each seed controls both the procedurally baked vegetation and the MPPI sampler. All trials use fixed Gazebo physics seed `4207`. A seed/route/direction combination should therefore be compared to the same combination under the other policies. Trials sharing one seed are related because they share a vegetation variant; do not analyze all route-direction rows as fully independent replicates.

Route lengths recorded in the CSVs are:

| Route | Length |
|---|---:|
| Route-11 | 563.15 m |
| Route-12 | 223.77 m |
| Route-13 | 99.49 m |

The final campaigns ran the controller in **lockstep**: 50 physics steps of 0.001 s per 0.05 s control tick (20 Hz control). The simulator launcher settings differ in some logs, but lockstep controls the progression of simulator time. Use `sim_time_s`, not `wall_time_s`, for policy-performance comparisons.

### Policy definitions

**Ground only** uses the local lidar occupancy map and front semantic/depth map. It records the same uncertainty/probe signals as the assisted run but never requests or loads a UAV map. Mapped reverse recovery is disabled.

**Reactive UAV**, despite the implementation mode name `counterfactual_uav`, is chiefly the final calibrated **forward-probe assistance system**, not a counterfactual-only condition.

Among the 353 requests in the 58 completed CSV trials, 319 (90.4%) came from the forward probe, 32 (9.1%) from the mobility fallback, one from the older decision-relevant counterfactual, and one from the action-relevant semantic path. The forward probe requests a combined occupancy and semantic product. The other paths can request a source-specific product.

**Always-on UAV** preloads exact route-wide physical occupancy and semantic traversability, with a 12.5 m margin around the route, before the first control tick. It makes no reactive requests or assistance holds. It is an optimistic upper-reference for information availability. Like reactive UAV, it permits mapped recovery; ground-only does not. Consequently, ground-only versus either UAV arm is a comparison of the complete assistance package—information plus enabled mapped recovery—not a pure sensing-only ablation.

### Simulator and mapping details worth stating in Methods

- Gazebo Harmonic supplies ground-truth world position and heading; wheel odometry supplies wheel speed.
- The rover uses physical front-wheel Ackermann steering and a 0.29 m wheelbase. The MPPI planner uses the corresponding velocity-command bicycle model.
- MPPI target speed is 1.7 m/s and maximum speed is 2.2 m/s.
- Terrain grades above 42% are treated as impassable.
- The ground lidar and semantic maps are 40 m square at 0.25 m/cell.
- Lidar input is a 15 Hz, 720 × 16 scan with a 45 m maximum range; mapping consumes it at approximately 10 Hz.
- Semantic observations come from co-located segmentation and depth cameras and are back-projected from 0.3 to 8 m.
- The simulated UAV response is an exact ground-truth local raster, not a noisy image-processing pipeline. Its primary GOOSE labels are soil 31, grass 50, brush 17, tree 28, and rock 40.

The implementation details and rationale are in [README.md](README.md) and [UAV_UNCERTAINTY_EXPERIMENTS.md](UAV_UNCERTAINTY_EXPERIMENTS.md).

## Current results

### Results by route, pooling directions

Times and driven distances in this table are calculated only over successful runs. Intervention and request means use every available final row, including failed endpoints.

| Route | Policy | n | Success | Mean interventions | Mean UAV requests | Successful-run mean time | Successful-run mean driven distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Route-11 | Ground only | 19 | 18/19 | 5.26 | 0 | 697.6 s | 753.5 m |
| Route-11 | Reactive UAV | 19 | 18/19 | 3.84 | 12.89 | 564.3 s | 629.7 m |
| Route-11 | Always-on UAV | 20 | 20/20 | 3.30 | 0 reactive | 515.7 s | 570.6 m |
| Route-12 | Ground only | 20 | 19/20 | 1.10 | 0 | 190.9 s | 231.9 m |
| Route-12 | Reactive UAV | 19 | 18/19 | 1.00 | 3.26 | 204.6 s | 245.1 m |
| Route-12 | Always-on UAV | 20 | 20/20 | 1.50 | 0 reactive | 194.0 s | 214.4 m |
| Route-13 | Ground only | 20 | 19/20 | 1.30 | 0 | 120.9 s | 120.5 m |
| Route-13 | Reactive UAV | 20 | 18/20 | 1.25 | 2.30 | 86.9 s | 112.4 m |
| Route-13 | Always-on UAV | 20 | 20/20 | 0.25 | 0 reactive | 75.3 s | 98.1 m |

The route-level result is not uniformly “more information means fewer interventions.” For example, always-on averages more interventions than ground-only on Route 12 (1.50 versus 1.10), while its strongest intervention reduction is on Route 13. Preserve route and direction in plots and models rather than reporting only a pooled grand mean.

### Distribution summaries

| Policy | Intervention median [IQR] | Request median [IQR] | Successful time median [IQR] | Successful distance median [IQR] |
|---|---:|---:|---:|---:|
| Ground only | 1 [1, 4] | 0 | 184.8 [135.8, 604.0] s | 223.7 [144.0, 618.1] m |
| Reactive UAV | 1 [0, 2.75] | 3 [2.25, 10] | 179.8 [99.4, 522.3] s | 223.9 [115.7, 602.9] m |
| Always-on UAV | 1 [0.75, 3] | 0 reactive | 180.1 [89.1, 489.3] s | 214.3 [98.9, 566.3] m |

The means differ more visibly than the medians because route lengths differ substantially and a few failed or repeatedly stuck cases are extreme. This is another reason to facet or adjust by route.

### Matched descriptive comparisons

These comparisons use only seed/route/direction keys present in both relevant folders. Time and distance use only pairs in which both policies reached the endpoint; intervention comparisons use every common final row. Percent changes are descriptive and have not been assigned confidence intervals or hypothesis tests.

| Comparison | Common rows | Endpoint result in common rows | Mean interventions | Both-success mean time | Both-success mean driven distance |
|---|---:|---|---|---|---|
| Ground → Reactive | 57 | Ground 54/57; reactive 53/57; 50 both; 4 ground-only; 3 reactive-only | 2.51 → 2.00 (**−20.3%**) | 334.3 → 273.4 s (**−18.2%**, n=50) | 364.4 → 317.0 m (**−13.0%**, n=50) |
| Ground → Always-on | 59 | Ground 56/59; always-on 59/59 | 2.51 → 1.66 (**−33.8%**) | 330.0 → 257.7 s (**−21.9%**, n=56) | 361.7 → 289.5 m (**−20.0%**, n=56) |
| Reactive → Always-on | 58 | Reactive 54/58; always-on 58/58 | 2.02 → 1.62 (**−19.7%**) | 285.3 → 259.9 s (**−8.9%**, n=54) | 329.1 → 294.3 m (**−10.6%**, n=54) |

For the main ground-versus-reactive comparison, the matched endpoint outcomes are nearly balanced and discordant in both directions. A safe statement is that reactive assistance reduced interventions and conditional completion time descriptively, but did not improve endpoint reliability in this sample.

## Metric definitions and interpretation

The authoritative endpoint row is each trial's `campaign.csv`, produced by [tools/run_experiment.py](tools/run_experiment.py).

| CSV field | Meaning | Important interpretation |
|---|---|---|
| `success` / `outcome` | Success requires reaching within 1.0 m of the final point after at least 90% monotonic route progress; other outcomes include timeout and stuck | Use as the primary endpoint/reliability measure |
| `sim_time_s` | Simulator elapsed time from trial start to termination | Comparable under lockstep; failures are censored/terminated outcomes, not ordinary completion times |
| `wall_time_s` | Host elapsed time | Measures compute throughput and contention, not navigation performance |
| `path_length_m` | Integrated physical travel; steps of at least 1 m are omitted so harness teleports do not count | Can exceed route length because of detours, loops, and recovery motion |
| `route_length_m` | Nominal route arc length | Fixed for a route and direction |
| `route_progress_m` | Maximum monotonic progress along the route | Harness drags update it to the drag target, so it is not purely autonomous progress |
| `interventions` | Harness drags after 30 simulator seconds without at least 0.5 m displacement | A safety/recovery burden, not simply a collision count |
| `interventions_resolved` | Drags after which the rover subsequently moved at least 0.5 m from the placed position under its own power | An unresolved final drag can accompany timeout/stuck termination |
| `uav_requests` | Final request count copied from autonomy status | Always-on is zero because its aerial map is preloaded |
| `seed` / `veg_seed` | Planner and vegetation variant identifiers | Use for matching and clustered uncertainty estimates |

The harness drags the rover at least 3 m along the route to a nearby collision-free and grade-safe location. Every intervention therefore both penalizes a run (at least 30 s of prior stalled time) and helps it continue. Time, distance, progress, and intervention count are not independent outcomes.

## Recommended paper figures

1. **Endpoint success by policy.** Show numerator/denominator and a binomial confidence interval. Mark the three missing trials separately rather than folding them into the failure rate.
2. **Paired intervention-count plot.** One line per matched seed/route/direction from ground-only to reactive UAV, faceted by route and direction. Overlay medians or estimated marginal means. This reveals whether the pooled reduction is broad or driven by a few difficult trials.
3. **Completion-time distribution with failures visible.** Use time-to-event curves or explicitly mark timeouts/stuck cases. A box plot over successes alone is useful only as a secondary conditional-efficiency plot.
4. **Route efficiency.** Plot `path_length_m / route_length_m` for successful matched pairs, faceted by route. State that path length excludes harness teleport jumps but includes autonomous detours and reverse recovery.
5. **Assistance burden.** For reactive UAV, show requests per run by route and trigger type. The current mean is 6.09 requests/trial and median is 3; Route 11 averages 12.89. If the real system has flight/communication cost, add map latency, bytes, flight time, or operator time rather than using request count alone.
6. **Spatial event map.** Overlay the route, request ROIs, request rover positions, and stuck positions from the two histories for representative matched runs. Use the same axes and world frame for ground and assisted panels.
7. **Representative event timeline.** From `assistance_trace.jsonl`, plot probe occupancy exposure, trigger threshold, world/wheel speed or slip, request/load state, recovery state, and harness interventions. Select the run using a predeclared rule (for example, median paired intervention improvement), not because it looks most favorable.

Avoid bar charts of means without the individual trials. Counts are zero-heavy and skewed, route lengths differ by more than fivefold, and rare severe failures materially affect the mean.

## Recommended statistical treatment

- Declare endpoint success and harness-intervention count as separate outcomes. Success is reliability; intervention count is recovery burden conditional on the harness allowing continued motion.
- Preserve matching on seed/route/direction. For intervention counts, use a paired/cluster bootstrap at the **seed** level or a negative-binomial mixed model with policy, route, and direction as fixed effects and seed as a random effect. Check zero inflation and overdispersion.
- For endpoint success, report exact counts and Wilson intervals. A paired comparison can use McNemar's test, but the current ground/reactive discordant count (4 versus 3) is too small to support a strong superiority claim.
- Treat non-successes as censored or competing terminal outcomes in completion-time analysis. Reporting time only for successful runs creates survivor bias.
- Report both absolute and relative effects: intervention difference per trial, rate ratio, completion-time difference, and requests required per intervention avoided.
- Include a sensitivity analysis that (a) uses only the fully matched rows, (b) treats missing runs separately, and (c) excludes obvious startup/pose failures only under a stated, policy-blind exclusion rule.
- If combining these results with real-world data, include domain as an interaction or present domains side by side. Do not pool simulator and field trials as exchangeable repetitions.

## What should align with the real-world presentation

For each real-world run, capture the closest equivalents of:

- policy and software/configuration version;
- course/route and direction;
- environment or vegetation condition;
- endpoint completion and a prespecified completion criterion;
- elapsed mission time;
- physical distance traveled and route progress;
- human or harness interventions with an operational definition;
- UAV requests, accepted maps, failures, response latency, and coverage area;
- timestamps and positions for requests, stalls, recoveries, and interventions;
- localization source and uncertainty;
- map resolution, semantic classes, and whether UAV products are inferred or ground truth.

The strongest real/simulation comparison will be directional: whether assistance changes completion reliability, intervention burden, completion time, and travel efficiency in the same direction and on the same types of difficult segment. Absolute simulator request counts and perfect-map performance should not be presented as predictions of field performance.

## Claims to avoid or qualify

- Do not claim that reactive UAV improves success: it does not in the current aggregate or matched snapshot.
- Do not call `uav_all_final` a pure counterfactual policy; 90.4% of completed-run requests were forward-probe requests.
- Do not call always-on “zero UAV use”; it uses complete preloaded aerial information and simply makes zero reactive requests.
- Do not attribute ground-versus-UAV differences solely to map information; the UAV arms also enable bounded mapped reverse recovery.
- Do not interpret `wall_time_s` as mission speed.
- Do not interpret harness interventions as independent collisions or route progress as fully autonomous progress.
- Do not treat the simulated UAV map as a realistic perception output. It is an exact ground-truth occupancy/semantic surrogate.
- Do not report statistical significance until missing trials, exclusions, analysis model, and uncertainty intervals are finalized.

## Short presentation-ready conclusion

> Across the currently completed simulation trials, reactive UAV assistance preserved roughly the same endpoint success as ground-only autonomy while descriptively reducing intervention burden and successful-run time, especially on the longest and most difficult route. Complete route-wide aerial knowledge performed best and completed every trial, establishing an optimistic reference rather than a deployable result. Because the reactive policy adds both local ground-truth maps and bounded mapped recovery, and because three trials are incomplete, final claims should use matched, route-aware analyses after rerunning the missing combinations.

