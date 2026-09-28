# ASPIRE2A to rented-host bundle transfer

`transfer_bundle.pbs` sends the already exported 74,570,166-byte archive directly from a real ASPIRE2A CPU allocation to the rented host. It has an `afterok:25579500.pbs101` dependency, verifies the expected source size, uses a per-job temporary Ed25519 key, and requires strict SSH host-key verification. It does not read `credentials.env`, download or rebuild the bundle, or use a login/local-node transfer path.

The job writes its public key and fingerprint under:

```text
/scratch/users/ntu/yguo017/lewm-rolling-ball-async/runs/<PBS_JOBID>/transfer_key_ed25519.pub
/scratch/users/ntu/yguo017/lewm-rolling-ball-async/runs/<PBS_JOBID>/transfer_key_fingerprint.txt
```

After the job reaches `R`, use the existing pinned helper to retrieve only those small files. Authorize the public key on the rented instance using the provider's normal access route. Then place these three small files in the same job output directory, writing `authorized.ready` last:

`autodl_endpoint.env` contains exactly four plain, unquoted entries (no passwords or shell commands):

```text
AUTODL_HOST=<provider hostname or IPv4>
AUTODL_USER=<SSH username>
AUTODL_PORT=22
REMOTE_DIR=/absolute/remote/directory
```

`autodl_known_hosts` contains the independently verified pinned host-key entry for that host and port. Do not use `ssh-keyscan` during the job, disable checking, or accept an unknown key. `authorized.ready` contains the exact values printed by the job:

```text
job_id=<PBS_JOBID>
key_fingerprint=SHA256:<fingerprint>
```

The allocation waits up to 15 minutes for all three files and matching marker values. It uploads to `REMOTE_DIR/rolling-ball-lewm-epoch100.tar.gz.partial.<PBS_JOBID>`, checks the remote byte count, and performs a no-clobber rename to `rolling-ball-lewm-epoch100.tar.gz`. The transfer summary and runner/wrapper exit files stay in the PBS run directory. The cleanup trap removes only that job's private key; the public key and fingerprint remain as small control evidence. A failed or timed-out transfer is reported and is never resubmitted automatically.
