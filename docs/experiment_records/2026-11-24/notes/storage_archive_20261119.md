> Historical record exported 2026-11-24T08:58:05.434451+00:00. Source: `notes/storage_archive_20261119.md`. Statuses and instructions below describe their original dates, not the current queue. Private connection values are placeholders; local-only references are labelled. See the experiment-record index for current interpretation.

# Remote result archive and quota recovery

User authorization: copy remote results to local storage, then delete remote
duplicates that have a verified local backup. Preserve running training and its
source. No training processes are stopped by this operation.

## Findings

- Remote project: `${SICC_PROJECT}`, about 48 GiB including environments.
- Result/source archive inventory: 738 regular files, 39.18 GiB, including 86 checkpoints.
- Local filesystem has about 1.9 TiB available.
- `Disk quota exceeded` is recorded by C8 job 27693. Filesystem-wide free
  space is not the account/group quota. `quota` is unavailable and `xfs_quota`
  cannot query the NFS mount, so the actual quota limit is still unknown.
- RQ3 job 27692 separately failed on CUDA initialization error 804; deleting
  files does not establish that this CUDA error has been fixed.
- C8 `runs/equidiff_c8_n50_s0/last.pt` is 3,645,964,288 bytes and does not have
  a valid ZIP directory. It cannot be assumed resumable. It is backed up and
  retained remotely because the run is incomplete.

## Operation

Worker: `notes/archive_remote_20261119.py` (independent of all training releases).
PID and log: `development/proposal_v2/results/remote_archive/20261119_quota/{pid,archive.log}`.

Archive root:
`development/proposal_v2/results/remote_archive/20261119_quota/robotic/`

- Copies all project artifacts except root `envs`, `data`, and `upstream`.
  Symlinks are not dereferenced. Frozen release source and manifests are copied.
- All 652 non-checkpoint regular files were copied and SHA-256 verified at
  20:25 CST. They are retained remotely too.
- Copies checkpoints with four concurrent transfers; hashes each local file,
  fsyncs it, records local proof, then checks the remote file's size, timestamp,
  and SHA-256 before any deletion.
- Deletes only `.pt` files from completed calibration runs or formal runs with
  all three test checkpoint reports marked complete. Acquires existing run and
  supervisor locks before deletion. Incomplete runs retain all checkpoints.
- Keeps remote logs, configurations, test results, completion markers, datasets,
  environments, and source files. No active queue source is changed.
- At 20:25 CST, 26 checkpoints were verified, 1.49 GiB was freed, and a remote
  1 MiB write probe succeeded. The full archive transfer is still in progress.

Audit files in the archive parent directory:

- `inventory.json`: initial relative paths, sizes, timestamps, metadata hashes.
- `verification.jsonl`: local hash proof and remote verification/deletion receipts.
- `summary.json`: written when the worker finishes; `state=complete` means all
  planned checkpoint transfers and metadata checks succeeded. Review `errors`
  if `state=needs_attention`.

The first transfer attempt encountered NFS exclusive-lock restrictions on a
read-only file descriptor. Only the archive worker was restarted with writable
lock descriptors; those initial errors are preserved in the audit log. The
current attempt serializes remote hash/deletion operations and has verified
successful deletions. No training was interrupted.

## Restoring checkpoints

Each archived checkpoint retains its original path relative to the remote
project root. Copy it back to that exact relative path if later evaluation or
extended training requires it. A completed run's remote `status.json` may point
to an archived checkpoint: use the archive inventory and receipt to locate it.
Do not erase completion markers or re-run completed calibration automatically.

## Remaining follow-up

- Wait for the worker's final summary and confirm the final freed byte count.
- RQ3 and C8 training jobs have not been restarted by this archive operation.
- Check GPU CUDA availability before resubmitting RQ3.
- C8 needs a valid checkpoint or a separately recorded fresh restart; its
  incomplete `last.pt` must not silently be treated as a valid resume point.
- Completed future image checkpoints need regular verified offloading to avoid
  filling the quota again.

## 20:33 CST quota and ownership inspection

- `/data/${CLUSTER_ACCOUNT}/${CLUSTER_USER}` used about 104-105 GiB during active cleanup:
  `task_6_paper` about 62 GiB and `robotic` about 42-43 GiB.
- `/data/${CLUSTER_ACCOUNT}` contains only `${CLUSTER_USER}` and also measured about 104 GiB.
  This does not account for files charged to the group outside that directory.
- `/home/${CLUSTER_USER}` uses only 136 KiB. The documented allowances are 1 TB shared
  by the research group under `/data`, and 200 GB per account under `/home`.
- Whole-filesystem free space: `/data` about 28 TiB; `/home` about 25 TiB.
  These values are not the user's available quota. Local `/mnt/hdd1` has about
  1.9 TiB free.
- Both NFS servers expose no rquota service in `rpcinfo -p`; `quota` is not
  installed, and `xfs_quota` cannot query the remote NFS filesystem. Exact live
  user/group limits and remaining allowance are still unconfirmed.
- Significant ownership discrepancy: release output files belong to primary
  numeric GID `1614000513`, while the official storage guide requires files
  under `/data` to use `group_${CLUSTER_ACCOUNT}` (GID 1069). At inspection, 576 release
  files totaling 21,753,273,847 bytes had the primary GID. The C8 release root
  has the correct group but lacks setgid, and its `runs` directories and
  checkpoint have the primary GID. This may affect quota attribution; it is
  not proof of which quota triggered the failure. This inspection did not
  change ownership or permissions.
