# NSCC ASPIRE2A Live Probe

Date: 2026-09-03 (Singapore time)

## Access result

- NTU Jump Host authentication succeeded through `172.21.26.100`.
- NSCC authentication succeeded on `aspire2antu.nscc.sg` (ASPIRE2A).
- NSCC authentication was rejected on `aspire2pntu.nscc.sg` (ASPIRE2A+).
- The historical NTU activation email and guide refer to ASPIRE2A, so the available allocation appears to be on ASPIRE2A rather than ASPIRE2A+.

## Live allocation

Project: `personal-yguo017`

| Item | Live value |
|---|---:|
| Expiry | 2032-12-31 |
| Granted | 100,000 SU |
| Reported used | 0 SU |
| Reported balance | 100,000 SU |
| CPU rate | 1 SU per CPU-core-hour |
| GPU rate | 64 SU per GPU-card-hour |

Equivalent theoretical maxima if the entire balance is spent on only one resource type:

- 100,000 CPU-core-hours.
- 1,562.5 GPU-card-hours.

The accounting snapshot shown by `myprojects` was timestamped earlier than the probe. It had not yet incorporated the probe job.

## Scheduled compute-node probe

PBS job: `16110366.pbs101`

| Item | Observed value |
|---|---|
| Final state | Finished |
| Exit status | 0 |
| Actual walltime | 1 second |
| Queue | `gdev` |
| Compute node | `x1000c3s1b0n0` |
| Requested allocation | 1 GPU, 16 CPU slots, 5 minutes |
| Scheduler-assigned memory | 110 GB |
| GPU visible to job | NVIDIA A100-SXM4-40GB |
| GPU memory | 40,960 MiB |
| NVIDIA driver | 570.124.06 |
| Operating system | Red Hat Enterprise Linux 8.10 |
| Kernel | 4.18.0-553.125.1.el8_10.x86_64 |

The scheduler temporarily reserved a maximum of about 5.33 SU for the five-minute GPU request. Based on the one-second runtime and the 64 SU/GPU-hour rate, the proportional usage is about 0.018 SU, subject to NSCC accounting granularity and reporting delay.

## Physical node versus job allocation

The compute node exposed the following physical hardware information:

- AMD EPYC 7713 64-Core Processor.
- 64 physical cores and 128 hardware threads.
- Four NUMA nodes.
- About 503 GiB total system RAM visible.

This does not authorize the job to consume the whole node. The PBS allocation was limited to 16 CPU slots, 110 GB RAM, and one A100 GPU.

## Storage

| Scope | Capacity or quota |
|---|---:|
| Personal home quota | 50 GB |
| Personal scratch quota | 100 TB |
| Shared `/home` filesystem | About 1 PB total |
| Shared `/scratch` filesystem | About 9.5 PB total |

Shared filesystem size is not the user's personal allowance; the personal quota is the binding limit.

## Software environment sample

The live environment includes Cray Programming Environment modules, GCC 8/10/11, NVIDIA programming environments, CUDA accelerator targets, Miniforge, Apptainer, MPI implementations, profiling tools, MATLAB runtimes, and scientific applications. Use `module avail` on the login node to inspect the complete current list.

