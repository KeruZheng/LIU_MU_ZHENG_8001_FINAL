# November 26: E4 continuation and evaluator recovery

The original seed-1/2 jobs continue unchanged. Independent continuation releases
restore model, EMA, optimizer, epoch/update counters and the LR schedule. Local
and actual remote checkpoint checks passed; SICC validation job 30414 completed
with exit 0:0, including the full formerly failing D2 shard under the new evaluator.
The old checkpoints lack RNG state, so bitwise uninterrupted equivalence is not claimed.

At this reporting cutoff, continuation submission is deferred by the account's
MaxSubmitJobsPerAccount limit. A persistent observer retries every minute with
afterany dependencies and duplicate-submission guards. No continuation job ID
has yet been issued. This is not a claim that continuation training has started.

Local seed-0 DP and EquiDiff final models are being evaluated sequentially with
fresh complete-scene construction, all-100-seed independent audits, explicit
scene pairing and bounded child-process failure handling. Formal v3 scores are
not yet complete. The old 112 v2 episode records remain separate.

See [implementation and protocol](../../../operations/e4_recovery_20261126/README.md)
and [validation evidence](../../../operations/e4_recovery_20261126/validation_evidence.json).
These are dated records, not a live dashboard. Main Push-T evidence remains frozen.
