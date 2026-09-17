# Supplement：五-member stratified stochastic-rounding coupling

**日期：2026-09-13**
**审查状态：`conditional_go`（仅 source / asset gate；未运行；novelty 未认证）**

本 supplement 保留 `PRIOR_GATE.zh.md` 的边界，专门审查一个更直接作用于官方 planner `avg` 的 recipe。它不是对原文件的改写，也不把 `min` 当主结果。

## 1. Recipe 与可识别问题

保留 TD-MPC2 的五个已训练 Q members、同一 W4 weight quantizer、同一 scale 和同一 tensor granularity。对每个 weight coordinate (c)，先抽取 (U_c\sim\mathrm{Uniform}(0,1))，再抽取一个独立的随机 permutation \(\pi_c\) of \(\{0,1,2,3,4\}\)，令

\[
 U_{j,c}=\bigl(U_c+\pi_c(j)/5\bigr)\bmod 1,
 \qquad j=0,\ldots,4.
\]

`U_{j,c}` 只决定 member (j) 在该 coordinate 的 stochastic rounding up/down；每个 member 的边际仍是 uniform，五个 members 在该 coordinate 覆盖五个等距 strata。`\pi_c` 必须按固定 coupling seed 可复现；不能用一个看过输出后选择的 permutation，也不能把同一 `U` 误作五个 member 的完全相同 rounding decision。

三个 arms 固定为：

- `W4-independent-SR`：五个 member 使用独立 uniform stream；
- `W4-stratified-SR`：使用上述 per-coordinate random permutation coupling；
- `W4-RTN`：相同 scale / clipping / W4 grid 的 deterministic round-to-nearest control；另保留 `FP32` reference。

不增加 members、不增加 Q forward 次数、不训练、不改 loss、planner 或 bit allocation。每个 arm 的五个 Q members 都保留；随机 two-Q aggregation 仍由官方 source 决定。

可识别性来自联合分布，而不是单 member 的幸运 rounding。对 member (j) 的 Q error 写成 (e_j)，独立和 stratified arms 有相同的 per-member marginal law，但不同的 cross-member joint law。若对应 weight fractional phase、输入敏感方向和 Q member 结构相近，strata 覆盖可能让 two-member average 的误差抵消；若这些相位不相近，covariance 可能为正、为负或接近零。于是 recipe 没有无条件的数学优越性，这正是可反驳点。

需要特别区分两件事：五-member 的全量平均即使因 stratification 获得 variance reduction，也不自动推出官方使用的随机 two-member average 会改善；后者必须用同一 `torch.randperm(... )[:2]` pair schedule 在真实 planner terminal scoring 上测量。不能用全五-member mean 代替官方 two-Q `avg`。

## 2. 真实官方路径与 DMControl reset 输入

TD-MPC2 source 固定为 `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`：

- [官方 `world_model.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py) 建立 `cfg.num_q` 个 Q members；`WorldModel.Q` 允许 `min`、`avg`、`all`，前两者先执行 `torch.randperm(self.cfg.num_q)[:2]`。
- [官方 `tdmpc2.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py) 的 `_estimate_value` 在 imagined latent rollout 后使用 live-Q `return_type='avg'`；`_td_target` 才使用 target-Q `return_type='min'`。本 recipe 的 primary outcome 是前者。
- `return_type='all'` 是 source 的逐-member two-hot value logits；screen 若用它记录五个 member，必须先按官方 `two_hot_inv` 解码成 scalar，再做 pair average 和 MSE，不能平均 raw logits。live `_Qs` 与 `_detach_Qs_params` 共享参数存储，target-Q 参数是独立 clone；weight transaction 只能改 live `_Qs`，并核对 alias 关系与 target digest，避免把 target 分支或 alias 误算成额外 member。
- [官方 `dmcontrol.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/envs/dmcontrol.py) 的 `make_env` 将 `cartpole-balance` 拆为 `cartpole` / `balance`，以 `task_kwargs={'random': cfg.seed}` 调用 `suite.load`，再包 `action_scale`、TD-MPC2 `DMControlWrapper` 和 `Timeout(max_episode_steps=500)`。

`cartpole-balance` reset-only 输入可以由官方 source 直接、低成本地产生，不需要下载 20/34 GB multi-task dataset，也不需要完整 episode rollout。具体结构已由 [官方 `cartpole.py`](https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/suite/cartpole.py) 和 [官方 `dmcontrol.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/envs/dmcontrol.py) 核对：

1. `Balance.get_observation` 返回 `OrderedDict(position, velocity)`。
2. `position` 是 `bounded_position()`：`cart_position` 一个 scalar，接着单 pole 的 `xmat[2:, ['zz','xz']].ravel()` 两个值；`velocity` 是 MuJoCo `qvel` 两个值。因此 TD-MPC2 `_obs_to_array` 的 flatten 顺序是 `[cart_position, pole_zz, pole_xz, cart_velocity, pole_angular_velocity]`，shape `(5,)`。禁止用 `qpos/qvel` 或其他自行设计的五维排列替代它。
3. Balance 默认 `swing_up=False`；每个 episode reset 从 seed 控制的 `RandomState` 采样 cart position、pole angle 和小初始 velocity。[官方 `cartpole.py`](https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/suite/cartpole.py)
4. 每个 fixed seed 必须建立一个 fresh env、只调用一次 `reset()` 并立即记录 observation；在同一 env 连续 reset 会推进其 `RandomState`，不等价于独立 seed blocks。
5. `DMControlWrapper.step` 每个 TD-MPC2 action 调用底层 `env.step` 两次，因此真实 action repeat 是 2；本 screen 不调用 `step`，不会把 reset-only 输入伪装成 full trajectory。底层 `cartpole.balance` 默认 time limit 是 10 秒，TD-MPC2 外层仍以 `max_episode_steps=500` 记录模型 episode convention。

官方依赖也可核验于 [TD-MPC2 `environment.yaml`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/docker/environment.yaml)：Python 3.11、PyTorch 2.7.1、`dm-control==1.0.16`、`mujoco==3.1.2`、`gymnasium==0.29.1`、Hydra、TensorDict/TorchRL 等。未来 reset-only preparation 可以是受 guard 保护的 CPU allocation；本轮不安装、不执行、不连接 cluster。

## 3. Asset 状态

| 项目 | 当前判断 | 证据与限制 |
|---|---|---|
| 小 checkpoint | **可核验但未 staging** | 官方 [models page](https://www.tdmpc2.com/models) 说明 single-task 是 5M；官方 HF [DMControl tree](https://huggingface.co/nicklashansen/tdmpc2/tree/main/dmcontrol) 列出 `cartpole-balance-1.pt` 31.3 MB。固定 revision 为 [`73a50e2719ed8258c72c7d1fefd23b781d66e35e`](https://huggingface.co/nicklashansen/tdmpc2/commit/73a50e2719ed8258c72c7d1fefd23b781d66e)，LFS pointer size `31344610` bytes，OID `4919e562d7f22f41a11118d1db1a0ebcb0e2b4681fc7fc594d772f0d5940869b`。本轮没有下载或 hash。 |
| 复现 config | **可核验** | [官方 `config.yaml`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/config.yaml) 默认 `obs=state`、`mpc=true`、`num_q=5`、`compile=true`；future screen 必须把 `compile=false` 作为显式 resolved override 记录。`model_size=5` 的实际 architecture 见 [common `__init__.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/__init__.py)。 |
| reset-only input | **可 bounded-prep** | 8 个不同 seed 的真实 `(5,)` observations 可由官方 DMControl reset 产生；不需要离线大 dataset，也不执行完整 rollout。该输入尚未在本机或 cluster 采集，需后续先做 CPU allocation preparation 并保存 raw observations、seed、source/config/dependency identity。 |
| 当前实验状态 | **未运行** | 未 load checkpoint、未运行 env、未做数值计算；future heavy I/O、checkpoint staging 和 preparation 必须在真实 SLURM compute allocation，并在脚本最前检查 job、actual hostname、owner/partition/node。 |

因此该 recipe 的 input concept 不再是先验 `resource_blocked`；它是 **asset-conditional**：小 checkpoint 可核验，8 个 reset observations 可由官方路径准备，但实际 asset 尚未 staging。若 CPU preparation 发现缺少精确依赖、source revision 或 reset observation identity，则立即回到 `resource_blocked`，不换成 synthetic state 或下载大 dataset。

## 4. Falsifiable hypothesis（avg 为唯一 primary）

固定 reset seed `5201..5208`，action seed `6201`，coupling seeds `[4101, 4102, 4103]`。每个 reset observation 对应 64 条固定 candidate action sequences，shape `[64, 3, 1]`；这些 actions 只在一次 FP world-model preparation 中生成并保存，所有 arms 复用同一 actions。FP world model 将每条 sequence roll 到 terminal latent，形成同一批 Q inputs；不执行 policy闭环或 env step。

对每个 arm，保存五个 member scalar outputs，并使用同一 pair schedule、同一 reduction dtype 和同一 candidate order 调用官方 `return_type='avg'`。若为固定 pair 需要 wrapper，wrapper 只能重放已记录的官方 `torch.randperm(... )[:2]` 结果；不能把全量五-member mean 改名为官方 avg。FP terminal `pi` action 和 latent Q input 也要逐 arm exact 对齐；其随机性由每个 coupling seed 的相同 RNG reset 控制。

定义每个 state block 的 `avg_aggregate_MSE` 为 64 个 candidate 上官方 two-Q avg score 相对 FP32 avg score 的平方误差，再对三 coupling seeds 与 candidate 等权平均。Recipe 的预注册 primary prediction 是：

- `W4-stratified-SR` 相对 `W4-independent-SR` 的 `avg_aggregate_MSE` 至少降低 25%；
- 8 个 state blocks 中至少 6 个有严格相同方向的 state-level improvement；
- `W4-stratified-SR` 的该 MSE 不高于 `W4-RTN`，并在至少 6/8 blocks 不高于 RTN；
- 为证明结果来自 joint law 而不是某个幸运 member，五个 member 各自相对 FP32 的 MSE 都不得在至少 6/8 blocks 达到 25% reduction。若单 member 明显改善而 aggregate 通过，只能报告“member-specific rounding benefit”，判定本 recipe 的 joint-law hypothesis 未通过；
- 若 stratified arm 只改变 member spread，却没有 two-Q avg aggregate MSE 改善，则 primary mechanism no-go；不转而把 spread 当 uncertainty claim。

这里的 25%、6/8 和 3 seeds 必须在看到结果前固定。`min` 只在 source audit 中保留为官方 target path 事实，本 supplement 不用它做 primary metric，也不声称对 TD target 或训练有效。

这不是纯代数测试：虽然 marginal law 相同，实际 metric 经过五个不同训练得到的 Q functions、latent terminal inputs、随机 two-member subset 和 candidate ranking。若五个 Q 的 fractional phases 不相似、输入敏感方向不共享，stratified coupling 可能没有收益；该 null 直接否定此 recipe 在该 checkpoint 上的 load-bearing mechanism。

## 5. Closest prior 与撞车风险

最直接的邻近 primary prior 是 [Ex Uno Pluria: Insights on Ensembling in Low Precision Number Systems（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/f10ceee5c6979988f334058561cac89f-Paper-Conference.pdf)。它用 Bernoulli stochastic rounding 从一个 pretrained model 构造低精度 ensemble，利用 rounding diversity 处理 ensemble memory cost；它没有保留一个已训练的 RL Q ensemble，也没有比较对应 coordinate 的 stratified shared law、official random-two `avg` 或 planner candidate scoring。

更接近“shared randomness + quantization”抽象的 primary prior 是 [Better than Optimal: Improving Adaptive Stochastic Quantization Using Shared Randomness（ACM 2025）](https://doi.org/10.1145/3771564)。它研究允许 quantizer/dequantizer 共享 randomness 的 adaptive unbiased quantization，并非 TD-MPC2、neural Q ensemble 或 random-subset aggregation；因此它使 generic shared-randomness novelty 风险更高，但没有显示本 recipe 的具体 RL path 已被实现。

[Generation of Ensemble Perturbations Using Low-Precision Floating-Point Numbers](https://www.jstage.jst.go.jp/article/jmsj/103/4/103_2025-022/_html/-char/en) 将 rounding error 用作天气 model ensemble perturbation，但没有 shared-strata coupling 或 Q planner。当前 campaign 的 `antithetic-rounding` practical no-go 比较的是增加两个 W4 members 与一个 W8 member 的成本/精度；这里不增加 member、不做 2W4 vs 1W8，只改变五个既有 critics 的 joint rounding law，但 generic stochastic/antithetic coupling overlap 仍使 novelty 只能保持 conditional、未认证。

## 6. Strong controls / fail-closed rules

screen 只有在下列工程条件全部通过后才允许运行：

1. **Source identity**：`tdmpc2` source commit、`dm_control`/MuJoCo dependency versions、HF revision、LFS OID/size、resolved `task=cartpole-balance`、`obs=state`、`model_size=5`、`num_q=5` 和 `compile=false` 全部落盘；源文件 hash 不得为空。
2. **Observation contract**：8 个 fresh env 每个只 reset 一次；manifest 保存 seed、flatten key order、shape `(5,)`、observation bytes/hash 和 source revision。任何把 qpos/qvel、旧 dataset 或连续 reset 当作同一 input 的情况立即停止。
3. **Action/Q-input contract**：保存 `[8,64,3,1]` candidate actions、action seed、FP terminal latent/action 输入；四个 arms 的 Q input exact 一致，candidate order、pair schedule、reduction dtype 一致。若从 `return_type='all'` 取输出，保存解码后的 scalar member values 以及 `two_hot_inv` identity。不得运行 policy闭环或 environment step。
4. **Quantizer contract**：四个 arms 共享 W4 grid、scale、clipping 和 tensor granularity；`independent-SR` 与 `stratified-SR` 每 member 的 marginal rounding law 相同；每 coordinate 的 permutation seed/definition 可复算；RTN 只作 deterministic control。任何 arm 因不同 scale 或 clipping 获益均判工程失败。
5. **Weight transaction**：只量化 live `_Qs`；encoder、dynamics、reward、policy、target Q 和 planner code 保持 FP32，除非另写并审查一个 target-Q 子实验。由于 `_detach_Qs_params` 是 live `_Qs` 的 alias，而 target-Q 是 cloned 参数，必须按实际 `id(parameter)` / `named_parameters()` 记录量化对象、bypass 集合与 target digest。每个 arm 前后保存完整 weight digest，no-op FP32 与 restore exact 失败则 fail closed。
6. **Primary gate**：只按第 4 节预注册的 avg aggregate MSE、6/8 state、25% reduction、RTN control 和 member-wise joint-law guard 判定。不得以 min/spread、额外 seed、额外 task、另一个 horizon 或 full rollout 挽救 null。

若 reset-only CPU preparation 成功，下一步只可在真实 SLURM V100 allocation 进行一次 `<=15 min`、内部 `<=12 min` 的 fixed-input screen；本 supplement 不授权提交或执行。结果只能支持该 TD-MPC2 checkpoint 与该 reset/action manifest 上的 conditional mechanism evidence，不能写成通用 stochastic quantizer、epistemic uncertainty 或 closed-loop task-success 方法。

## 7. 本轮记录

- 只读取本地 prior docs 与官方 primary source；未运行 model/env、未下载、未连接或提交 cluster。
- 已确认 reset-only `(5,)` input 的官方 flatten/action-repeat/episode wrapper 语义；当前 checkpoint/input 尚未 staging。
- 当前 novelty 仍是 conditional、未认证；若官方 shared-randomness / low-precision ensemble prior 被进一步证明覆盖该 exact recipe，应转 `novelty_no_go`。
