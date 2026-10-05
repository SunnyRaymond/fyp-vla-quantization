# Fast-WAM LIBERO one-episode smoke

This folder contains a reusable PBS entry for one real LIBERO closed-loop episode. The entry uses the official FastWAM repository at `7faa71108368fbb3b6885649f112af607427a2d4`, the published `yuanty/fastwam` Optional IDM checkpoint, and its matching dataset stats. `action_infer_mode=first_frame` selects the Fast-WAM direct-action mode: each observation goes through `infer_action`, and the predicted action chunk is stepped in LIBERO. This is not an IDM-mode run or a full-suite score.

The prepared ASPIRE2A environment uses the existing Python 3.11.7 venv and LIBERO Apptainer image. Rendering uses CPU OSMesa; inference uses the single PBS-allocated CUDA GPU. The episode runs with the prepared model components offline.

Submit from the ASPIRE2A login node:

```bash
qsub /scratch/users/ntu/yguo017/fastwam-smoke/osmesa_smoke.pbs
```

The script exits unless it has a PBS job ID, runs outside a login node, sees one allocated GPU, and confirms that PyTorch sees the same GPU UUID. It samples that GPU's utilization and memory every 15 seconds into `job.log`. Artifacts go to `/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/<job_id>/`: `job.log`, `eval.log`, `results.json`, `rollout.mp4`, `exit_code.txt`, and either `TASK_SUCCESS` or `TASK_FAILURE`. `PIPELINE_COMPLETE` means evaluation and output checks completed; use the outcome marker and JSON to determine task success.

To choose another task, edit `EVALUATION.task_suite_name` and `EVALUATION.task_id` in `osmesa_smoke.pbs`. Keep one trial for a smoke. Official code and release instructions: [FastWAM](https://github.com/yuantianyuan01/FastWAM); published checkpoint and stats: [yuanty/fastwam](https://huggingface.co/yuanty/fastwam).
