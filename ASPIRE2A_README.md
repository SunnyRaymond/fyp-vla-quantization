# NSCC ASPIRE2A access and PBS job quick start

This folder contains a tested Windows launcher for connecting to **NSCC ASPIRE2A** through the **NTU Jump Host**, plus the essential commands for running work with **PBS Pro**.

> This setup targets **ASPIRE2A** (`aspire2antu.nscc.sg`, NVIDIA A100), not **ASPIRE2A+** (`aspire2pntu.nscc.sg`, NVIDIA H100). Access to the two systems is provisioned separately.

## One-click connection

1. Connect the computer to the **NTU network** (or an NTU remote-access service that can reach the Jump Host).
2. Make sure [`nscc-access/nscc-credentials.env`](nscc-access/nscc-credentials.env) contains the current NTU and NSCC credentials.
3. Double-click [`Connect-ASPIRE2A.cmd`](Connect-ASPIRE2A.cmd).
4. Wait for the ASPIRE2A prompt. Type `exit` when finished.

The launcher:

- authenticates to the NTU Jump Host;
- opens a tunnel to ASPIRE2A;
- verifies the saved host keys for both machines; and
- opens an interactive ASPIRE2A shell without displaying the passwords.

The credential file is intentionally excluded by [`nscc-access/.gitignore`](nscc-access/.gitignore). It still contains plaintext secrets: do not commit, email, sync, or share it. If a password has appeared in chat or another exposed place, rotate it.

## Connect with VS Code Remote - SSH

The Windows SSH config contains an `ASPIRE2A` host that routes through `NTU-JumpHost`. The one-click launcher reads both passwords automatically from `nscc-access/nscc-credentials.env`; they are not copied into the SSH config.

The easiest option is to double-click [`Open-ASPIRE2A-in-VSCode.cmd`](Open-ASPIRE2A-in-VSCode.cmd). It opens the remote home directory `/home/users/ntu/yguo017` in a new VS Code window.

Alternatively, inside an already-open VS Code window:

1. Open the Command Palette with `Ctrl+Shift+P`.
2. Select `Remote - SSH: Connect to Host...`.
3. Select `ASPIRE2A`.
4. Enter the NTU and NSCC passwords if prompted, then open `/home/users/ntu/yguo017`.

Use the one-click launcher when you want automatic password entry. Its AskPass settings apply only to the VS Code process it launches, so unrelated SSH hosts are unaffected.

> Opening VS Code connects only to the ASPIRE2A login node and does not allocate a GPU. Run compute workloads through PBS; do not run them directly in the VS Code terminal on the login node.

## Manual connection fallback

From Windows PowerShell, connect to the NTU Jump Host:

```powershell
ssh -o KexAlgorithms=ecdh-sha2-nistp256 -c aes256-ctr -m hmac-sha2-256 YOUR_NTU_ID@172.21.26.100
```

Then, from the Jump Host, connect to ASPIRE2A:

```bash
ssh YOUR_NSCC_ID@aspire2antu.nscc.sg
```

Use the NTU password for the first step and the NSCC password for the second. The explicit SSH algorithms avoid the `Corrupted MAC on input` failure observed with the default Windows OpenSSH negotiation. Do not store files on the Jump Host.

## First checks after login

The login node is for editing, transferring files, compiling small programs, and submitting jobs. Do not run heavy computation there.

```bash
# Projects and remaining allocation
myprojects

# Usage history
myusage

# Personal storage usage and quota
myquota

# Available queues and your jobs
qstat -Q
qstat -u "$USER"

# Available software
module avail
```

The account check performed on 3 September 2026 showed:

- personal project: `personal-yguo017`;
- allocation: 100,000 SU, with 0 used at that snapshot;
- conversion: 1 SU per CPU core-hour or 64 SU per GPU card-hour;
- theoretical maximum if used for only one resource type: 100,000 CPU core-hours or 1,562.5 GPU card-hours;
- personal storage: 50 GB in `/home` and 100 TB in `/scratch`.

Always run `myprojects`, `myusage`, and `myquota` again for the current values. `/scratch` is temporary working storage, not an archival location.

## Submit a small CPU job

Create `hello_cpu.pbs` on ASPIRE2A:

```bash
#!/bin/bash
#PBS -N hello_cpu
#PBS -P personal-yguo017
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=00:05:00
#PBS -j oe

cd "$PBS_O_WORKDIR"
echo "Job ID: $PBS_JOBID"
echo "Compute node: $(hostname)"
date
```

Submit and monitor it:

```bash
qsub hello_cpu.pbs
qstat -u "$USER"
qstat -f JOB_ID
```

When it finishes, PBS writes a file such as `hello_cpu.o16123456` in the submission directory:

```bash
cat hello_cpu.o*
```

To cancel one of your own queued or running jobs:

```bash
qdel JOB_ID
```

## Submit a small GPU job

Create `hello_gpu.pbs`:

```bash
#!/bin/bash
#PBS -N hello_gpu
#PBS -P personal-yguo017
#PBS -q normal
#PBS -l select=1:ncpus=16:mem=110gb:ngpus=1
#PBS -l walltime=00:10:00
#PBS -j oe

cd "$PBS_O_WORKDIR"
echo "Compute node: $(hostname)"
nvidia-smi
```

Then submit it with `qsub hello_gpu.pbs`. ASPIRE2A enforces a ratio of 16 CPU cores and 110 GB memory for each requested GPU. Request a realistic walltime: PBS temporarily reserves allocation based on the request, then reconciles it after the job ends.

## Interactive compute sessions

Use an interactive PBS job when exploring software or debugging. The prompt changes only after PBS allocates a compute node.

```bash
# CPU session
qsub -I -P personal-yguo017 -q normal \
  -l select=1:ncpus=4:mem=16gb \
  -l walltime=01:00:00

# GPU session
qsub -I -P personal-yguo017 -q normal \
  -l select=1:ncpus=16:mem=110gb:ngpus=1 \
  -l walltime=01:00:00
```

Type `exit` to end the compute session and release its resources. Type `exit` once more to leave ASPIRE2A.

## Software environments

ASPIRE2A provides software through Environment Modules:

```bash
module avail
module load miniforge3
module list
python --version
```

Remove loaded modules when changing environments:

```bash
module purge
```

Prefer a virtual environment or Conda environment in your own storage rather than installing packages into the system Python.

## Observed compute-node configuration

A completed one-GPU test job was allocated:

- AMD EPYC 7713 compute node;
- 16 allocated CPU slots;
- 110 GB allocated memory;
- 1 NVIDIA A100-SXM4-40GB GPU.

The node exposed 128 hardware threads and about 503 GiB physical RAM, but a job may use only the resources granted in its PBS allocation. Seeing the full node hardware does not grant permission to use all of it.

## Troubleshooting

- **The launcher cannot reach the Jump Host:** confirm the computer is on the NTU network and that the NTU account has Jump Host access.
- **NTU authentication failed:** update `NTU_PASSWORD` in the credential file.
- **ASPIRE2A authentication failed:** update `NSCC_PASSWORD`; do not substitute the ASPIRE2A+ host.
- **Host-key mismatch:** stop. Do not bypass the check. Confirm a legitimate NSCC key rotation before updating the saved key.
- **Job remains queued:** inspect `qstat -f JOB_ID`; resource availability, queue policy, project allocation, and requested walltime can all affect scheduling.
- **Job exits immediately:** inspect the generated `.oJOB_NUMBER` output and `qstat -xf JOB_ID`.

## Local references

- [`nscc-access/Using NTU JumpHost to NSCC ASPIRE-2A.pdf`](nscc-access/Using%20NTU%20JumpHost%20to%20NSCC%20ASPIRE-2A.pdf)
- [`nscc-access/ASPIRE2A_LIVE_PROBE.md`](nscc-access/ASPIRE2A_LIVE_PROBE.md)
- [NSCC ASPIRE2A FAQ](https://help.nscc.sg/aspire2a/faqs/)
- [NSCC ASPIRE2A User Guide](https://help.nscc.sg/aspire2a/user-guide/)
- [NSCC ASPIRE2A General QuickStart Guide](https://help.nscc.sg/wp-content/uploads/2024/05/ASPIRE2A-General-Quickstart-Guide.pdf)
