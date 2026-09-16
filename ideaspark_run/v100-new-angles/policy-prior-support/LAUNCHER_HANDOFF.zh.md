# Launcher 当前合同

root审查后的policy_support_gpu.sh已用于GPU64833，UGGPU-TC1单V100、4CPU、24GB、10min，真实allocation_guard优先。外层timeout540s，内部480s；无model/data操作在head或本地。

复制policy_support_screen.py、tdq_screen.py、allocation guards、policy_support_protocol.zh.md、policy_support_input_freeze.json和launcher自身到当前numericjob目录。CLI为--manifest --source-root --checkpoint --config --protocol --input-freeze --output --max-seconds480；helper直接使用job目录内的固定文件，无额外--tdq-helper参数。

结束记录exit_status.json。独立CPU作业另行验证；原始结果contract见VERIFIER_HANDOFF.zh.md。
