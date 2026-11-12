#!/bin/bash
# EEE CPU job: check the relayed tar against its ASPIRE2A hash, extract it under /projects/yguo017/scratch (the
# container sees it at /scratch/users/ntu/yguo017) and check every extracted file against FILES.sha256.
#SBATCH --job-name=laq840un
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/projects/yguo017/laq840/logs/%x-%j.out
set -euo pipefail
IN=/projects/yguo017/incoming/laq840
S=/projects/yguo017/scratch
cd "$IN"
sha256sum -c SHA256SUMS
tar -xf laq840.tar -C "$S"
cd "$S"
sha256sum --quiet -c "$IN/FILES.sha256"
echo "files $(wc -l < "$IN/FILES.sha256")"
rm -f "$IN/laq840.tar"
{ getfattr -n ceph.dir.rbytes /projects/yguo017/ 2>/dev/null || true; } | tail -1
echo UNPACK_PASS
