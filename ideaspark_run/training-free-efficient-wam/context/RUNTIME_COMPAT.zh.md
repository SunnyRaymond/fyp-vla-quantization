# 本轮 publication 接口兼容

2026-10-02，原 `quality_publish` 的验证已完成，但渲染调用报错：`render_one() got an unexpected keyword argument 'compile_pdfs'`。

已安装的 `quality_cli.publish` 传入 `compile_pdfs=True`，而同一 skill 的 `render_pdf.render_one(expansion, out_dir)` 没有此参数，且其原实现默认就执行 PDF compilation。

本轮用 run-local `publish_runtime_compat.py` 包装这一调用：仅接受 `compile_pdfs=True` 并调用原 renderer。继续使用原 validation、Markdown/TeX/PDF renderer、compile 和 publication receipt；不跳过任何 gate，不修改 scientific content，也不修改 installed skill。没有安装 compiler 或依赖。

这解决的是函数签名版本不一致，不是科研验证。最终 PDF 的实际成功/失败状态以 `phase4/work/render_status.json` 为准。

原 publication validators 和 receipt 成功后，旧 `needs_work.json` 仍只记录该已解决的 execution-contract 错误，原 navigator 会继续停在该旧错误。确认 marker 仅含此错误、current publication receipt 有效且 publication findings 无 fail 后，将其保存为 `context/resolved_publish_api_error.json`；原 `next` 随后实际返回 DONE。未删除或忽略科研 findings。

本轮 Markdown/TeX 保存成功，PDF 因环境没有可用 compiler 而未生成；没有安装依赖。
