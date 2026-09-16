# GPU 前实现审查修复

独立审查的原始问题保留在 GPU_IMPLEMENTATION_AUDIT.zh.md。提交64815的 runner LF SHA256 为 `361761c49c2babd4d7cfdfdb620f645da4f132f072d9a938b66246da44d92e07`，49,824 bytes。

修复了合法 frozen snapshot 在第二步误报、FP 被误识别为 draw、不同 condition 的 prefix 被相互比较，以及 protocol staging 文件名问题。最终实现还绑定外部批准的 manifest/base/extension/helper/protocol hashes，检查 extension parent chain、state8/action7，保存原 FP weights、codes/scales、实际 call labels/time 和 common-path input hash。实际参数别名通过 named_parameters(remove_duplicate=False) 枚举；每步 readback 与 restore 仍由 GPU 已批准源代码执行，不将 CPU 对 receipt 的读取称为重新观察历史 GPU 写操作。

初始 FP 手写 Euler/official no-op 与 RTN reuse/reload 对照各自为两次10-step trajectory，因此实际有 **40 个工程 control calls**；协议中“另20”是工作量估算少计，完整科学 workload 仍为1320次 suffix calls。此次修复未改科学 arms、样本、threshold、call recipe 或10分钟allocation/540s内部上限。

CPU source preparation 的完整源文件 hash 清单保留在 compute manifest；GPU再次核对最终 sample NPZ 与 manifest/base/extension 身份，而不是重新解码所有原视频。研究结论须等完整 raw 与独立 CPU replay，不能根据已通过的初始工程对照提前宣布可行。
