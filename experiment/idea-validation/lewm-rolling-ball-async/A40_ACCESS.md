# ASPIRE2A A40 可视化访问核实

**核实日期：2026-09-27。** 结论：A40 提供了值得追查的硬件路径；官方资料尚不足以证明当前账号可使用 `pbs104`，也没有证明 Isaac Sim 能在该节点/Portal profile 上运行。没有登录 Portal、连主机、提交作业或启动模拟器。

| 项目 | 官方依据 | 可下的结论 |
|---|---|---|
| ASPIRE2A 资源 | [NSCC 2024 Introductory Workshop](https://help.nscc.sg/wp-content/uploads/2024/05/NSCC-Introductory-Workshop-Theory-ASPIRE2A.pdf)，v1.3（2024-03-06），第18页：8个 visualization nodes，每节点1张 A40；[Advanced Job Management Guide](https://help.nscc.sg/wp-content/uploads/ASPIRE2A-ADVANCED-JOB-MANAGEMENT-TRAINING_GUIDE.pdf)，v1.4（2025-04-08），第158页：`pbs104` 是8个 A40 节点的 Visualization cluster，可经 Visualization Portal 访问。 | `pbs104` 是官方文档记载的**集群/server**名称，不是已确认的 queue 名。两份材料都不是当前可用状态或个人权限证明。`pbs101` 的 queue 查询不能代表 `pbs104`。 |
| A40 与 Isaac Sim | [NVIDIA A40 datasheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf)：48 GB、84个第二代 RT Cores；[当前 Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)：最低 GPU 栏列 GeForce RTX 4080，并明确“不带 RT Cores 的 GPU（A100、H100）不支持”。 | A40 避开了“无 RT Core”这一明确排除条件，显存也超过表列最低值；但 A40 **没有出现在最低 GPU 型号列表**。节点驱动、Isaac 版本、容器/扩展、GPU/vGPU profile 与传感器场景是否兼容均未验证，所以不能声称 Isaac RGB evaluator 已可运行。 |
| 合法访问入口 | NSCC 当前 [ASPIRE2A user-guide 页面](https://help.nscc.sg/aspire2a/user-guide/)链接到 [Job and Visualization Portals User Guide](https://help.nscc.sg/wp-content/uploads/aspire2a-job-portal-guide_Feb2026.pdf)（PDF封面标 v1.5、2025-12-22，内页又标 v1.4）。可视化登录页列 `https://ntuvisual.nscc.sg`（NTU network）及 `https://visual.nscc.sg`（NSCC VPN），使用 ASPIRE2A credentials；第25页说明从 Visualization Portal 的 Desktops 选择应用会提交 interactive job。 | 对 NTU 用户，下一条文档支持的访问检查是从 NTU 网络自行打开 `ntuvisual.nscc.sg`，登录后**只查看** Desktops 目录及可见的资源/作业详情。Portal 是 interactive desktop 路径，不等于批准任意 batch workload。指南没有给出 `pbs104` 的 queue 名。 |

## 下一步检查

1. 用户从 NTU 网络打开 [`ntuvisual.nscc.sg`](https://ntuvisual.nscc.sg)，登录后只看是否有可用的 A40/Visualization Desktop 项，以及界面是否显示 server、queue、GPU、walltime、project 等字段；本说明不要求也没有执行 session submission。
2. 若没有该入口、没有可见 A40 项，或详情没有标出实际资源，请走 NSCC [Service Desk Portal](https://keris.service-now.com/csm) 确认以下事项后再申请 session：账号是否有 `pbs104` 权限；该集群当前的 queue/Portal profile 与获批的短交互申请方式；GPU、walltime、SU/project 限制；是否允许用户自定义命令运行 Isaac Sim/Isaac Lab；支持的 OS/container、driver/CUDA/GPU passthrough profile，以及资产下载/网络要求。
3. 只有在获批的 A40 allocation 内，再做最小运行时确认（识别到的 GPU/driver、Isaac Sim Compatibility Checker 与目标 RGB 场景启动）。这些检查尚未运行；未确认前，Isaac-based native evaluation 仍是硬件/运行时未决项。不要从 `pbs101` 猜 queue 或把其通用 GPU 配额套到 `pbs104`。

**证据边界：** 本核实仅使用公开的 NSCC/NVIDIA 文档。`pbs104` queue、实时节点状态、个人访问权限、Portal 当前应用目录及任意 Isaac workload 的政策均无公开证据支持，需通过上述 Portal 可见信息或 NSCC 官方支持确认。
