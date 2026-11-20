# Cluster follow-up queues

Submitted November 20 after the first convergence queues finished. All new work
uses separate frozen releases and run directories. Primary 5090/C8 queues and
their source, model configuration, and environments are unchanged.

| Platform | Job | Work |
|---|---|---|
| Legacy | 232668 | Two GPUs: exact-augmentation convergence, 25 and 50 demos, seeds 0/1/2 |
| Legacy | 232669 | Two GPUs: exact-augmentation convergence, 100 and 200 demos, seeds 0/1/2 |
| SICC V100 | 28265 | C4 convergence, 50 demos, remaining seeds 1/2 |
| SICC V100 | 28266 | C4 convergence, 200 demos, remaining seeds 1/2 |
| SICC H800 | 28269 | Image RQ3 fixed goal: DP, C4, C4+gate, three seeds (nine runs) |
| SICC A100 | 28270 | Image RQ3 rotated goal: C4+gate, three seeds |

The existing C8 job 27812 and official reference 28232 retain their slots.
Together these occupy the two submission slots per A100/H800/V100 partition.
The legacy jobs request the user's four-GPU/two-running-job maximum; submission
does not imply allocation. Normal SICC QoS allows seven submissions overall,
but the three relevant GPU partitions admit only six collectively. The legacy
four-submission cap permits two additional pending jobs, but no duplicate or
unvalidated filler is submitted just to consume that cap.

Convergence uses fresh 20k/40k/60k validation curves. It is a follow-up designed
after primary results, not a pre-registered result or primary test replacement.
Augmentation completes the missing comparison arm; extra C4 seeds turn two
single-seed diagnostics into three-seed curves. The V100 allocations repeat CUDA
and full native replay checks before training, including after the independent
official environment's Gym compatibility fix.

Image RQ3's 18 comparison cells reuse six completed rotated DP/C4 primary runs
by global run ID and add 12 distinct trainings. This avoids duplicate baseline
work; it is a documented revision of the proposal's 18 additional image runs.
The RGB target is visible, so the hidden-goal state hypothesis does not apply
unchanged. The new image gate and visible-goal environment have unit tests.
Every image allocation also checks full replay, full-size batch64 backward and
native rollout before formal work. A failed check prevents training dispatch.

All eight new releases are registered with the verified archive worker and fleet
monitor. Completed runs must be transferred, hash-verified locally, and cleaned
remotely before the next seed. Code/configuration snapshots are published under
experiments/releases, evidence under the corresponding research task.
