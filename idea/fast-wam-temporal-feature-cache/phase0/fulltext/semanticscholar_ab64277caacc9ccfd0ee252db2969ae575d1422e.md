# Trajectory-Consistent Calibration for Cache-Accelerated Diffusion Models

paper_id: semanticscholar:ab64277caacc9ccfd0ee252db2969ae575d1422e
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Diffusion Transformers (DiTs) [Peebles and Xie, 2023] have substantially advanced visual generation
by bringing scalable transformer backbones to diffusion models. However, their iterative denoising
procedure requires repeated forward passes of a large backbone, making inference computationally
expensive. Since generating a single sample involves multiple denoising steps [Ho et al., 2020, Song
et al., 2021a], reducing repeated computation has become critical for efficient diffusion sampling.
Cache-based acceleration methods address this challenge by caching and reusing intermediate
representations across nearby denoising steps in DiT. By exploiting temporal redundancy along the
diffusion trajectory, they reduce repeated computation and provide a practical training-free route
to faster inference [Ma et al., 2024b, Wimbauer et al., 2024]. However, cache reuse is inherently
approximate rather than lossless. Because a reused representation is computed from a previous
or neighboring denoising state [Ma et al., 2024b], it can deviate from the representation that full
computation would produce at the current state. As sampling proceeds, this approximation error can
further propagate through later denoising steps, causing the cache-accelerated trajectory to drift from
the full-computation trajectory and ultimately degrade generation quality.
A key observation of this work is that the deviation introduced by cache reuse is not purely local.
Instead, effective calibration for cache reuse should account for two coupled effects: the direct
∗Equal contribution.
†Corresponding author.
Preprint.
arXiv:2605.24870v1  [cs.CV]  24 May 2026

Figure 1: Qualitative comparison on PixArt-α under cache-accelerated sampling, with prompts shown
verbatim. TCC recovers prompt-relevant content weakened by cache reuse, such as the parking sign
and green fire hydrants, and mitigates detail degradation when semantics are preserved, e.g., facial
details. The third row shows that TCC preserves visual quality under complex compositions. All
comparisons use matched prompts, seeds, and sampling settings.
mismatch between a reused representation and its full-computation counterpart at the current denoising
step, and the trajectory shift induced when earlier reuse and calibration alter the subsequent cache-side
trajectory. Existing cache acceleration methods mainly improve reuse itself. One line of work focuses
on reuse decisions, designing or learning cache schedules that determine which timesteps, layers,
tokens, or states can be safely reused [Selvaraju et al., 2024, Ma et al., 2024a, Zou et al., 2024].
Another line introduces optimization, calibration, or compensation mechanisms to better approximate
the computation skipped by caching [Chen et al., 2025, Qiu et al., 2025a]. However, these approaches
do not explicitly model the coupled effect of direct mismatch and the trajectory shift introduced by
reuse correction. Prior correction methods typically estimate a fixed error prior from the original
cache and full-computation trajectories. In contrast, TCC treats calibration as part of the trajectory
and estimates each subsequent prior under the corrected history induced by previous calibrations.
To address this challenge, we propose Trajectory-Consistent Calibration (TCC), a training-free

## Method

underlying cache schedule or model parameters. TCC takes the full-computation trajectory as a
