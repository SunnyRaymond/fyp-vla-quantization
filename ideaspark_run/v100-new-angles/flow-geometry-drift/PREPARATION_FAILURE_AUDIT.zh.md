# CPU preparation启动故障记录

2026-09-13，root复核。64743、64745、64748、64749、64750、64751均失败，具体scheduler状态保留于campaign scheduler目录与jobs.json；部分未取得完整日志，不推定各次完全同因。

64751原始日志明确为 `cp: cannot stat '/tc1home/UG/yguo017/v100_newangles_ccds/control/smolvla_prepare.py': No such file or directory`。远端与本地shell当时确实引用这个不存在的basename，实际文件为smolvla_cpu_prepare.py。cp位于allocation guard之后，所以“未进入guard”这一早期说法不成立；未进入Python准备或安装。

无证据支持stale sbatch或并发写者。root停止原agent写入，修正cp和执行入口，上传小shell并读回确认，随后仅提交64752（TC1N05，CPU-only）。没有删除缓存、锁、进程或重跑已完成科学实验。

这是工程失败及同协议修复，不产生科学no-go。原始失败历史不覆盖。

64752在真实CPU allocation运行33s，进入安装后失败：旧bootstrap pip对typing-extensions wheel名称规范化报不一致，退回sdist，而sole PyTorch index缺少flit_core。root保留原log，改为先从PyPI安装pip25.3，PyPI主index与固定CUDA wheel extra-index，并将pip cache放在本任务assetdir减少重复下载。提交64754，TC1N06，CPU-only；未开始模型推理、未改科学协议。

64754在TC1N06运行11m30s，完成环境安装、checkpoint/base metadata/dataset metadata下载，因tasks.parquet文本字段预期不符而停止。CPU探针64755/64756记录实际schema：task文本是Pandas index `__index_level_0__`；episode metadata缺tasks，task_index存在于tabular data；video feature metadata是HWC，decoder输出CHW。

相机名称差异由checkpoint保存的rename processor明确解决：image→camera1、image2→camera2，保持raw keys让官方processor执行rename。state nominal config6与真实dataset8不一致，但保存的normalizer mean/std是8维；v0.4.4 normalizer直接按stats广播，prepare_state保留全部输入后pad32，没有8→6截断。root按此真实接口修复，记录差异，不改变checkpoint、stats或数据数值。

恢复只允许显式旧job64754，先验证lock owner与sacct FAILED，再将旧lock存档并建立新单写者lock；不盲目删除锁。task映射读取tabular data的episode/task列，维持“前4 task、各前3 episode、quarter frame”的冻结规则，新增tabular IO计入同一总字节上限。

## 64757：decoder dynamic library，非scientific failure

CPU allocation TC1N03，FAILED 1:0，5m52s。task mapping与下载阶段已通过；首个视频decode进入TorchCodec后失败，完整trace保存在artifacts/64757/run.log，实际错误为libpython3.10.so.1.0无法加载。没有GPU inference或flow结果。root在读取原始trace及scheduler终态后，将同一LeRobot v0.4.4 decode_video_frames的backend显式设为pyav，保留相同source video、timestamp、tolerance和CHW uint8契约；不再调用该版本不支持的return_uint8参数。官方source：https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/video_utils.py。新作业仅可显式retire 64757的FAILED lock，不覆盖之前记录。
