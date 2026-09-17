# Prior-art map: REPA / iREPA-style alignment in VLA and WAM

> **Search snapshot:** 2026-08-31  
> **Question:** Has the idea “transfer `REPA` / `iREPA` representation alignment to `VLA`” already been implemented?  
> **Short answer:** `REPA`-style transfer is already crowded; an exact `iREPA` spatial recipe in a matched robot-policy study was not verified.

## 1. Decision rule

I classify a paper as a **direct implementation** only if it has most of the following:

1. a trainable `VLA`, action `DiT`, video `DiT`, or `WAM` hidden representation;
2. a frozen or slowly updated external representation target;
3. an explicit feature-space alignment loss, usually cosine-based;
4. joint optimization with action, flow-matching, diffusion, or world-model loss;
5. a training-time-only teacher/projector or a clearly stated inference path;
6. robot-control evaluation.

An **exact iREPA transfer** additionally requires both:

- a spatially local projector comparable to the `3×3 Conv`; and
- spatial normalization of patch-token teacher targets as defined by `iREPA`.

## 2. Evidence-ranked map

| Date / status | Work | Class | What is aligned | Relation to the idea | Verified limitation |
|---|---|---|---|---|---|
| 2025, CoRL final | [FLARE](https://proceedings.mlr.press/v305/zheng25a.html) | **Direct** | intermediate future tokens in a flow-matching action `DiT` → future-observation embeddings | Explicitly states similarity to `REPA`; joint `L_fm + λL_align`; future branch removed from control output | future embedding target and `MLP`; not exact `iREPA` |
| 2025 preprint; ICLR 2026 work | [Spatial Forcing](https://arxiv.org/abs/2510.12276) | **Direct** | intermediate `VLA` visual tokens → frozen `VGGT` spatial features | Classic frozen-teacher cosine alignment inside a `VLA`; reports up to `3.8×` convergence acceleration | local file is arXiv v2 because official download was inaccessible; no verified `iREPA` pair |
| 2026 arXiv v1 | [FRAPPE](https://arxiv.org/abs/2602.17259) | **Direct / extension** | future-prefix representations → multiple visual foundation models | Multiple future representation alignment, two-stage fine-tuning | future/world-model route; exact spatial normalization not verified |
| 2026 arXiv v1 | [FutureVLA](https://arxiv.org/abs/2603.10712) | **Partial / extension** | joint visuomotor predictive embeddings → downstream `VLA` latent space | Representation-level foresight and downstream latent alignment | pretraining and model design change together; not a clean iREPA ablation |
| 2026 arXiv v1 | [VEGA](https://arxiv.org/abs/2605.10485) | **Direct** | `OpenVLA-OFT` visual-encoder patch tokens → frozen `DINOv2-FiT3D` | Very clean REPA-like current-frame spatial transfer; teacher/projector removed at inference | explicitly uses `LayerNorm + two-layer MLP`, not iREPA `Conv + spatial norm` |
| 2026 arXiv v1 | [AGRA](https://arxiv.org/abs/2606.12217) | **Direct WAM** | intermediate video `DiT` features → frozen `DINOv2` | Title and method explicitly repurpose representation alignment at the world/action interface | `WAM` rather than a standard visual-backbone VLA; exact iREPA recipe absent |
| 2026 arXiv v1 | [SAM3D-Guided](https://arxiv.org/abs/2607.25912) | **Direct / object-centric** | intermediate `π0` visual features → target-object dense `SAM3D` features | Object-centric 3D spatial teacher for a `VLA` | changes target selection and 3D teacher; no exact iREPA component test |
| 2026 arXiv v2 | [Mind-VLA](https://arxiv.org/abs/2608.04633) | **Direct / instruction-aware** | instruction-selected object representation → `VAE` / `VGGT` features | Instruction-aware 3D representation alignment | compact custom backbone and object-view pipeline complicate clean attribution |
| 2026 arXiv v1 | [Robust-WAM](https://arxiv.org/abs/2608.05903) | **Direct WAM** | future query outputs in action stream → future-frame `DINOv3 CLS` | Training-time semantic foresight alignment with matched temporal positions | uses global `CLS`, not local patch structure; query tokens remain at inference |
| 2025 arXiv v1; AAAI 2026 record | [ReconVLA](https://arxiv.org/abs/2508.10333) | **Adjacent** | VLA visual outputs condition gaze-region reconstruction | Auxiliary training signal improves grounding | target is reconstructed pixels/region, not frozen-teacher feature alignment |

## 3. The closest papers

### 3.1 FLARE — closest early action-DiT transfer

**Why it matters**

`FLARE` is the strongest reason not to claim that nobody has moved `REPA` into robot action generation. Its method section explicitly compares the framework to `REPA`.

**How it works**

- Adds learnable future tokens to a flow-matching action `DiT` sequence.
- Slices their hidden states at an intermediate layer.
- Applies an `MLP` and aligns them by cosine similarity to future-observation embeddings.
- Optimizes `L = L_fm + λL_align`.

**What differs from REPA/iREPA**

- The teacher target is a future observation, not the clean version of the current noised input.
- The future-token stream is separated from noised action tokens but interacts via self-attention.
- The target encoder is made action-aware and updated by `EMA` in the reported recipe, rather than being a permanently frozen generic visual encoder.
- No exact `iREPA` spatial projector/normalization test is reported.

### 3.2 Spatial Forcing — closest classic VLA port

**Why it matters**

This is the closest direct answer if the idea means “force a `VLA` hidden representation to look like a strong external spatial encoder.”

**How it works**

- Processes multi-view robot observations with a frozen `VGGT` teacher.
- Aligns intermediate per-pixel/patch `VLA` visual tokens after `BatchNorm` and a two-layer `MLP`.
- Uses cosine similarity and combines alignment with the normal action objective.
- Evaluates simulation and real-world robot control, plus convergence and data efficiency.

**Evidence boundary**

The reported `3.8×` value is a paper claim about convergence under its setup. It is not a universal wall-clock speedup and should not be compared directly with `REPA`'s `17.5×` training-step statement.

### 3.3 VEGA — clean spatial-teacher baseline

**Why it matters**

`VEGA` makes the alignment location unusually explicit: it aligns the output of the `DINOv2` branch of `OpenVLA-OFT` before linguistic entanglement.

**How it works**

- Student: second-to-last `DINOv2` visual-backbone patch tokens.
- Teacher: final-layer `DINOv2-FiT3D` patch tokens.
- Projector: `LayerNorm + two-layer MLP + GELU`.
- Loss: mean cosine distance plus action loss.
- Teacher and projector are removed at inference.

This is a strong baseline for any claimed `Action-iREPA`: replace only the projector/target normalization first, without changing the rest of the `VLA`.

### 3.4 AGRA — literal WAM repurposing

**Why it matters**

`AGRA` is the most literal paper-title match to the proposed idea: *Making Foresight Actionable: Repurposing Representation Alignment in World Action Models*.

**How it works**

- Diagnoses a representation mismatch between plausible generated futures and the action decoder.
- Aligns intermediate video `DiT` features to spatially coherent frozen `DINOv2` representations.
- Optimizes the alignment jointly with the `WAM` action path.

This largely closes novelty for “REPA in WAM,” but leaves open a controlled test of `iREPA`'s spatial mechanisms in a direct `VLA` action expert.

### 3.5 Robust-WAM — useful counterexample to iREPA's local-spatial story

`Robust-WAM` aligns future action-stream queries with per-frame `DINOv3 CLS` targets. Its ablation reports that the global `CLS` target outperforms several patch/depth alternatives under its setup. That is important counter-evidence: robot control may prefer compact future semantics in some architectures even if image generation benefits from local patch structure.

The correct research question is therefore empirical:

> Under which `VLA` representation, layer, time, and task does iREPA-style local structure outperform a global future-semantic target?

## 4. Novelty boundary

### Claims that are no longer defensible

- “No one has used representation alignment in `VLA`.”
- “No one has aligned robot-policy hidden states to frozen visual foundation models.”
- “No one has used future representation prediction/alignment with a flow-matching action policy.”
- “A training-only alignment teacher with no inference teacher overhead is new.”
- “Applying REPA to a WAM is new.”

### Search-bounded opening

The following narrower claim remains plausible as of the snapshot:

> A controlled `VLA` study that ports the exact `iREPA` spatial recipe—local `Conv` projection plus spatially normalized patch targets—to action-relevant hidden states, and isolates its effect from teacher choice, data, architecture, and control protocol.

This should be phrased as a hypothesis until an updated prior-art search and experiment support it.

## 5. A reviewer-defensible reformulation

### Working title

`Action-iREPA: When Does Spatial Representation Alignment Improve Flow-Based VLA Policies?`

### Core research question

Does preserving teacher patch geometry with an iREPA-style projector and target normalization improve optimization speed, closed-loop success, and robustness beyond existing MLP/cosine spatial alignment?

### Minimal method contribution

Start from one public flow-based `VLA` baseline and keep its backbone, data, action horizon, controller, and evaluation fixed.

1. Select one student grid that directly affects action prediction.
2. Compare current-frame and future-frame teacher targets.
3. Compare global targets (`CLS`) with patch targets.
4. Replace the baseline `MLP` projector with a local projector that preserves the 2D grid.
5. Apply iREPA spatial normalization to teacher patch tokens.
6. Optionally extend the grid operator to multi-view or 3D correspondences only after the exact port is isolated.

### Stronger extension directions

- **Multi-view iREPA:** separate intra-camera `3×3 Conv` from cross-camera alignment.
- **Object-conditioned iREPA:** normalize and align only instruction-selected object tokens.
- **Temporal iREPA:** preserve local correspondences across future frames, not only within one frame.
- **Noise/time-aware alignment:** schedule `λ(t)` over the action flow-matching timestep.
- **3D-aware projector:** compare 2D `Conv`, cross-view attention, and neighborhood graph operations under matched parameter/compute budgets.

## 6. Minimum matched experiment

| Variant | Projector | Target processing | Teacher target | Purpose |
|---|---|---|---|---|
| A. Base | none | none | none | action-policy baseline |
| B. REPA-style | two-layer `MLP` | none | patch features | reproduces existing alignment family |
| C. Conv only | `3×3 Conv` | none | patch features | isolates local projector |
| D. Spatial norm only | two-layer `MLP` | iREPA normalization | patch features | isolates target normalization |
| E. Exact iREPA-style | `3×3 Conv` | iREPA normalization | patch features | tests the proposed transfer |
| F. Global future | linear/MLP | none | future-frame `CLS` | counter-baseline inspired by `Robust-WAM` |
| G. 3D patch | matched projector | matched normalization | `VGGT` or `DINOv2-FiT3D` | tests teacher geometry |

Keep identical:

- initial checkpoint and trainable modules;
- robot dataset and frame sampling;
- action representation, horizon, and noise schedule;
- optimizer, batch size, update count, augmentations, and seed count;
- simulator version, task definitions, episode count, termination rules, cameras, and controller frequency;
- evaluation checkpoints.

Report at least:

- success/progress with confidence intervals across seeds and tasks;
- learning curves, area under the curve, steps-to-threshold, and wall-clock-to-threshold;
- action error as a diagnostic, not a replacement for rollouts;
- teacher feature extraction time, training throughput, peak training memory, and total GPU-hours;
- inference `P50/P95/P99` latency, achievable policy/control frequency, peak inference memory, and power/energy;
- in-distribution and appearance/spatial/object-level OOD behavior.

## 7. Falsification criteria

The idea should be considered unsupported if, under matched settings:

- exact iREPA-style alignment does not beat the REPA-style `MLP` baseline across seeds;
- gains disappear when teacher compute is included in wall-clock-to-threshold;
- offline alignment metrics improve but closed-loop success does not;
- gains arise only from a stronger teacher, extra future frames, or additional data;
- spatial normalization harms instruction semantics or multi-camera consistency;
- robustness gains do not survive task/object/camera shifts.

## 8. Search method and limitations

The search combined exact terms (`REPA`, `iREPA`, `representation alignment`) with `VLA`, robot policy, flow matching, future latent, spatial teacher, and `WAM` terms across arXiv, OpenAlex, Crossref, DBLP, and Semantic Scholar. Primary proceedings/arXiv records and local full text were then used for inclusion decisions.

- `OpenReview` could not be queried through the installed environment because `openreview-py` was unavailable.
- `Semantic Scholar` returned repeated rate limits for several broad queries.
- The installed search CLI had no `--json` flag; [search-results-unfiltered.json](search-results-unfiltered.json) was recovered through the same skill's programmatic API.
- The exact negative search for `iREPA` was also checked against every local candidate PDF; only the iREPA paper itself contained the term.

See [SEARCH_REPORT.md](SEARCH_REPORT.md) for the reproducible query log and filtered set.

## AI-assisted preparation disclosure

This map was produced with AI-assisted searching and synthesis. Paper identities and the direct/partial/adjacent classifications were checked against primary full texts. Independent replication was not performed, and numerical results from unmatched papers were not treated as direct comparisons.

