# Corrected E4 evaluation (2026-11-24)

The frozen training runner disables robosuite hard resets. In the pinned
MimicGen Square_D1/D2 implementation, peg geometry is sampled in `_load_model`.
Consequently the old final scores sample nuts but retain one peg per worker.
The reported seed-0 D1/D2 zeros remain real outputs of that restricted protocol,
but are not validated estimates over the full target distribution. This finding
does not establish that corrected scores will increase. Main Push-T studies
and their frozen report use a separate evaluator and are unaffected.

This independent evaluator uses the saved epoch-500 EMA, with no retraining or
target-based checkpoint selection. All three domains are reevaluated with
100 seeds (200000--200099), 400 control steps, four environments, the same
absolute-action conversion, camera metadata, normalizer and binary success
metric. Each reset seeds NumPy before a complete model reset. States-only
caching is bypassed. Peg/nut poses, model/state hashes, and observation hashes
are recorded for every episode. D2 retains its official camera change.

Two independent processes with different constructor seeds must reproduce
identical scenes and initial images for three episode seeds, including a
repeat after intervening scenes. D1/D2 peg poses must vary; D2 rotations must
vary. All domains also require a real-model CUDA rollout smoke test. Formal
scores are produced only after these gates pass. Every four-episode shard
uses a fresh process, so full-reset rendering allocations cannot accumulate
across the 100-episode suite. The upstream multiprocessing runner uses spawn;
the reset override is installed inside each spawned environment factory.

Diffusion RNG is fixed per four-episode shard. Its schedule differs from the
old continuous evaluation stream, so before/after score differences cannot
be attributed exclusively to the scene reset. Across methods, scene pairing
uses the same deterministic reset schedule, independently of diffusion RNG.
SICC inference uses its existing PyTorch environment; version and GPU are
recorded per shard, as the local training environment differs.

`export.py` creates an inference-only EMA package and provenance hashes.
`submit.py` requests a separate SICC GPU allocation using the existing private
graphics runtime. `evaluate.py` runs validation and evaluation. `watch.py`
copies results locally every minute and verifies every completed result file
and the retained local EMA before removing only the remote EMA duplicate.
Active local and remote training releases, environments and queues are unchanged.

Runtime outputs belong under `tasks/rq1_transfer/results/e4_corrected_eval_20261124/`.
Current frozen deployment: `e4_corrected_eval_20261124_v2`. Previous outputs are retained.

Deployment evidence: GPU reset job 29923 passed on SICC RTX 2080 Ti. Initial
suite 29929 passed constructor-independent D1 scene/image checks, then failed
before formal episodes because ESCNN expanded-filter caches were absent in the
train-mode EMA checkpoint. Version 2 explicitly calls `train()` before strict
state loading, then rebuilds caches with `eval()`. A CPU check confirms every
loaded EMA tensor exactly equals the exported checkpoint. Replacement job
29935 uses the same weights, metadata and scene protocol, and must pass all
GPU smoke tests before formal evaluation. No training source was modified.
The archived v1 remote weight duplicate was removed only after verifying its
local backup and all failed-job result/log files. Version 2 is independently
monitored under `results/e4_corrected_eval_20261124/v2/`.
