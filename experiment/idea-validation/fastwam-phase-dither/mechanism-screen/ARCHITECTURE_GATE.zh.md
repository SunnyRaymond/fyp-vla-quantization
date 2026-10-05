# FastWAM first_frame 架构门槛（源码只读）

## 版本与范围

源码目录 `D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM` 的 Git HEAD 为 `7faa71108368fbb3b6885649f112af607427a2d4`；检查时 `git status --short` 为空。这里只核源码；没有读取 checkpoint/config，不能确认特定权重、层数或加载来源，也不据此声称 checkpoint 已实际运行。

候选定义：`../proposal/final_candidate.json`。Optional IDM 的 `infer_action(action_infer_mode="first_frame")` 委托给 `FastWAM.infer_action`；`idm` mode 才走 IDM 方法（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam_optional_idm.py:54-57,59-118`）。`FastWAM.infer_action` 要求 `video_attention_mask_mode="first_frame_causal"`（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:996-1000`）。所以 first_frame 是 Optional IDM 类上的一个推理路由；不能从源代码推出磁盘 checkpoint 的具体内容。

## first_frame 实际调用路径

入口编码输入图像为 first-frame latents，调用 `video_expert.prepare(..., action=None)`，再在 scheduler 循环前只调用一次 `prefill_video_cache_tensor`；之后每个 action scheduler step 调 `_denoise_action_with_video_cache` 并更新 action latents（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:1034-1099,1120-1127,1133-1159`）。每个 action step 执行 `action_expert.prepare → mot.forward_action_with_video_cache_tensor → action_expert.post`（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:703-736`）。MoT 对每层重算 action Q/K/V，将它们与缓存 video K/V 拼接后做 attention，再跑 block post（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\mot.py:528-578`）。

MoT 每个 block 对当前 token 做 norm/modulation 和 self-attention Q/K/V；post 执行 attention O、cross-attention、FFN 与 gated residual（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\mot.py:151-210,131-149`）。模块定义为 self-attention `q/k/v/o`、cross-attention `q/k/v/o`、双 Linear FFN、LayerNorm 与 modulation 参数（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\wan_video_dit.py:170-245`）。ActionDiT 每步还执行 `action_encoder`、两层 `time_embedding`、`time_projection`、两层 `text_embedding`，最后执行 action `head`（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\action_dit.py:73-97,282-331`）。若模型启用且传入 proprio，还会执行 `proprio_encoder` 并把 token 拼入 context（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:58-62,224-245,1060-1065`）。Video expert 与 ActionDiT 是分开的专家对象，MoT 同时注册二者（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:136-159`）。

## 缓存与无效计算边界

Video prefill 每层计算后只保存并返回 K/V 缓存；Action 路径跨 denoise steps 复用它们（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\mot.py:474-526`; `D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:1120-1127,1144-1152`）。前 `L-1` 层的 video Q/O、cross-attention、FFN 会改变后层输入及其缓存 K/V，因此影响 action endpoint。末层 video K/V 仍进入 action attention；末层 video Q 与 post 中的 O、cross-attention、FFN 虽然执行，却只更新最终 `x`，而函数只返回 K/V 列表，所以这些输出不影响 action（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\mot.py:487-526`）。Video `head/post` 在该入口没有调用；video `action_embedding` 因 `action=None` 不执行（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:1083-1090`; `D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\wan_video_dit.py:686-694,722-731`）。

## 固定输入性质的适用边界

固定输入的 subtractive-dither 条件性质，只对输入独立于共享 draw `u_b`、且未触及有限码本边界的局部调用成立。首个量化点的上游图像 token、外部 text context 或 sinusoidal timestep 可能满足；不能把它推广到整条路径。第一个量化 Linear 之后，FFN 第二层、后续 block、attention O/cross-attention/FFN/action head 的输入都可能是先前同一 `u_b` 量化结果的函数。Action latents 也由每步 denoiser 输出经 scheduler 更新（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\fastwam.py:1139-1155`）。

缓存 K/V 虽在 action steps 之间不变，但若 video prefill 用同一个 `u_b` 量化，它仍是 `u_b` 的函数；step 间复用不等于独立于 dither。`t_mod` 和 context 会被传入多个 block（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\action_dit.py:288-307,312-327`; `D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\mot.py:553-578`）。共享 conditioning 本身不是 novelty；待测的是活跃位点 phase 对完整 W4A8 action endpoint 配对 MSE 的影响。源码不支持声称全路径边际不变、无偏或已有 phase 收益。

## 完整 W4A8 的合法覆盖建议

保留候选的全路径范围：W4 权重与 A8 activation 覆盖所有影响 first_frame action endpoint 的 Linear 调用，不缩成单层 pilot：

- Action expert：`action_encoder`、`time_embedding`、`time_projection`、`text_embedding`、所有 `blocks[i].self_attn.{q,k,v,o}`、`blocks[i].cross_attn.{q,k,v,o}`、`blocks[i].ffn.{0,2}`、`head`；按实际 scheduler step 记录调用。
- Video expert：`time_embedding`、`time_projection`、`text_embedding`；前 `L-1` 层的 self/cross-attention 与 FFN Linear；末层至少 `self_attn.k/v`，因为它们进入缓存并影响 action。Video prefill 每次 `infer_action` 只运行一次，不要按每个 action step 重复记账。
- 若启用且传入 proprio，包含 `proprio_encoder`。

末层 video Q/O、cross-attention、FFN 是执行但无 endpoint 影响的输出。若为了统一部署 recipe 仍对其 W4A8，phase 必须在各 arm 固定为共同基线，不纳入 action-MSE phase 搜索或有效位点数；删去这些计算则属于另一项干预。Video `head` 和 video `action_embedding` 不在这条路径。`patch_embedding` 是 Conv3d；VAE/T5 输入准备不属于本候选 Linear 集合；LayerNorm、attention、residual、scheduler、modulation 参数本身也不是 Linear。

以 canonical module + stream + 实际调用位置（video prefill 或 action denoise step）为 phase key，避免同一专家同时通过 `video_expert` 与 `mot.mixtures.video` 等引用被重复计数。最终结论仍须来自全路径 W4A8、其余配置配对固定的 endpoint MSE；本门槛只界定合法调用点，不预报实验结果。
