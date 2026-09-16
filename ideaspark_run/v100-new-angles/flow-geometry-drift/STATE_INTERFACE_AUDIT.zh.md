# SmolVLA state / camera / dataset interface audit

日期：2026-09-13  
范围：只读核对 LeRobot v0.4.4 官方 source、CCDS CPU metadata probe 与当前 smolvla_cpu_prepare.py / flow_screen.py；未连接集群、未加载模型、未做本地数值计算。

## 结论

**state interface 可以按 8 → official MEAN_STD → pad32 执行；禁止 8 → 6 截断。** checkpoint input_features["observation.state"].shape=[6] 是保存的 nominal feature metadata，不是运行时 state vector 的切片指令。CPU64756 的只读 header 证据显示 observation.state.{mean,std} 均为 [8]，dataset observation.state 也是 [8]，而 checkpoint max_state_dim=32。因此当前 _checkpoint_mapping 中保留 8 维并要求 stats [8] 的方向正确。若实际运行的旧脚本仍把 checkpoint shape 与 dataset shape 做 exact equality，必须停用该旧逻辑。

官方 normalize_processor.py 的 _normalize_observation（约 L216–234）只按 feature key/type 应用 transform；MEAN_STD（约 L288–301）直接使用 saved mean/std，没有按 PolicyFeature.shape 截断。官方 modeling_smolvla.py 的 prepare_state（约 L427–431）先取最后 observation frame，再调用 pad_vector；pad_vector（约 L141–152）只建立目标宽度并拷贝已有坐标。state_proj 的输入是 max_state_dim=32（约 L519–521）。这条调用链支持 8 维 stats/state 后补零到 32；没有发现官方 8→6 adapter。输入 6 维反而会与 [8] stats 不相容。

## camera 合同

CPU raw NPZ 应保留 dataset 原名 observation.images.image 与 observation.images.image2，每帧 uint8 CHW。flow_screen.py 当前 L670–675 已在调用 saved policy preprocessor 前转为 float32 [0,1]；不能直接把 uint8 [0,255] 交给官方 prepare_images，因为该函数（约 L365–402）执行的是 img*2-1。

artifacts/64756/metadata_probe.json 中 saved policy_preprocessor 的 rename map 是：

- observation.images.image → observation.images.camera1
- observation.images.image2 → observation.images.camera2

官方 rename_processor.py（约 L23–45）是 observation key 的 old→new 映射，未列出的 key 保留。故 GPU 端必须先运行完整 saved preprocessor，再调用 policy；不要在 CPU raw 文件中提前改名。checkpoint 还声明 camera3，但本 dataset 没有它且 empty_cameras=0。官方 prepare_images 会处理 present keys，并且在 empty_cameras=0 时不会伪造 camera3 图像；这是本 screen 应保留的两 camera 行为。GPU contract gate 应记录 post-preprocessor keys 为 camera1/camera2、无 camera3 零图，并确认至少一个 image 输入存在。

## task / episode 合同

CPU64756 证据显示 tasks.parquet 的列是 task_index 与 Pandas index 导出的 __index_level_0__（实际为 instruction text）；episode parquet 只有 episode_index、length 与数据/视频 pointers，没有 tasks。因此旧的直接读取 episode tasks 会重现 CPU64754 的失败。

当前脚本已加入两步修复：_task_rows L235–255 仅在 Pandas metadata 明确声明该 index 时把 __index_level_0__ 当 text；_join_episode_tasks L493–514 从 data parquet 的 episode_index/task_index 建立 association，并要求每个 episode 唯一一致；之后 _select_episodes 才执行冻结的前四 task、每 task 前三 episode、floor(length/4)。这与 probe schema 一致。若未来 metadata schema 改变，应 fail closed，不能把任意 index 或 episode 顺序当 instruction。

## 最小执行门

1. 在真实 SLURM allocation 内复核 manifest/identity 的 checkpoint 与 dataset revision、hash 与 raw state shape [8]。
2. 对一个 batch 记录：raw state [1,8]；normalizer 后仍 [1,8]；prepare_state 后 [1,32]，且不出现截断/手写 stats。
3. 复核 preprocessor 后 camera keys、image dtype/range、empty_cameras，并保持 camera1/camera2 的真实顺序。
4. 旧的 CPU64754 报错仅代表当时 __index_level_0__ 尚未被识别；不要据此改成任意 task 选择。当前代码尚未在本审查中重新运行。

未知项仅限于运行时安装的 LeRobot 版本是否确为 v0.4.4，以及 GPU runner 是否真的使用本地 saved preprocessor；两者必须作为 execution gate 记录。没有证据支持寻找或发明 8→6 转换。

