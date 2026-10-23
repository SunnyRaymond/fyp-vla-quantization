#!/usr/bin/env python3
"""Build a small, local HTML gallery from retrieved Prism visualization files."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path, PureWindowsPath


MAX_PREVIEW_BYTES = 220 * 1024
REMOTE_RESULT_ROOT = "/scratch/users/ntu/yguo017/fastwam-prism-visualization-20261012/results/full"
STREAMS = ("video_expert", "action_expert")
FOCUS_KINDS = ("paired", "actual_pre", "actual_post", "weight")
TRANSFORMS = (
    ("paired_raw", "raw", "paired_raw_to_qdq"),
    ("paired_learned_spinquant", "learned SpinQuant", "paired_learned_spinquant_to_qdq"),
    ("paired_smooth05_hadamard", "Smooth α=0.5 + H128", "paired_smooth05_hadamard_to_qdq"),
)
ARMS = (
    ("bf16", "BF16"),
    ("spinquant_native_w4a4", "SpinQuant-adapted · native INT4"),
    ("smooth_scalar_native_w4a4", "SmoothScalar · native INT4"),
    ("row_scale_vq_a4", "RowVQ · decoded BF16"),
    ("micro_scale_vq_a4", "MicroVQ · decoded BF16"),
)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 顶层必须是 JSON object")
    return value


def safe_preview(root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest 中有空的 JPEG preview 路径")
    normalized = value.replace("\\", "/")
    rel = Path(normalized)
    if rel.is_absolute() or PureWindowsPath(normalized).is_absolute() or ".." in rel.parts:
        raise ValueError(f"preview 必须是 plots 目录内的相对路径: {value!r}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"preview 超出 plots 目录: {value!r}") from exc
    if not candidate.is_file() or candidate.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError(f"本机 JPEG preview 不存在: {value}")
    size = candidate.stat().st_size
    if size <= 0 or size > MAX_PREVIEW_BYTES:
        raise ValueError(f"JPEG preview 大小须为 1–220 KiB: {value} ({size} bytes)")
    return rel.as_posix(), candidate


def figure_pages(root: Path, figure: dict) -> list[tuple[dict, str | None, Path | None]]:
    pages = figure.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"figure {figure.get('id')!r} 缺少 pages")
    result = []
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError(f"figure {figure.get('id')!r} 的 page 必须是 JSON object")
        if "preview" in page:
            rel, path = safe_preview(root, page["preview"])
            result.append((page, rel, path))
        else:
            result.append((page, None, None))
    return result


def figure_previews(root: Path, figure: dict) -> list[tuple[str, Path]]:
    return [(rel, path) for _, rel, path in figure_pages(root, figure) if rel is not None and path is not None]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary 中的数值字段不是 JSON number: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("summary 中包含非有限数值")
    return format(value, ".6g") if isinstance(value, float) else str(value)


def image_card(root: Path, rel: str, path: Path, title: str, alt: str) -> str:
    src = esc(rel)
    return (
        '<figure class="figure-card">'
        f'<a href="{src}" target="_blank" rel="noopener"><img src="{src}" '
        f'alt="{esc(alt)}" loading="lazy"></a>'
        f"<figcaption>{esc(title)} <span>{path.stat().st_size / 1024:.1f} KiB</span></figcaption>"
        "</figure>"
    )


def paired_table(summary: dict, stream: str) -> str:
    main = summary.get("main_activation_records", {})
    records = main.get("paired_local", [])
    matches = [row for row in records if row.get("stream") == stream
               and row.get("block_index") == 10
               and isinstance(row.get("module"), str) and row["module"].endswith("ffn.2")]
    if len(matches) != 1:
        raise ValueError(f"summary 中 {stream} / block 10 / ffn.2 paired_local 记录应恰有一条")
    row = matches[0]
    arrays, nmse = row.get("arrays", {}), row.get("nmse", {})
    body = []
    for key, label, nmse_key in TRANSFORMS:
        values = arrays.get(key)
        if not isinstance(values, dict) or nmse_key not in nmse:
            raise ValueError(f"summary 中 paired_local 缺少 {key} 或 {nmse_key}")
        axes = values.get("axis_rms_cv", {})
        body.append(
            "<tr>"
            f"<th>{esc(label)}</th>"
            f"<td>{number(values.get('max_abs_to_rms'))}</td>"
            f"<td>{number(axes.get('column_axis'))}</td>"
            f"<td>{number(axes.get('row_axis'))}</td>"
            f"<td>{number(nmse[nmse_key])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>paired-local transform</th>'
        "<th>max_abs / RMS</th><th>column_axis (channel) CV</th><th>row_axis (token) CV</th>"
        "<th>NMSE to QDQ</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def weight_table(summary: dict, stream: str) -> str:
    records = summary.get("all_weight_records", [])
    matches = [row for row in records if row.get("stream") == stream
               and row.get("block_index") == 10
               and isinstance(row.get("module"), str) and row["module"].endswith("ffn.2")]
    if len(matches) != len(ARMS):
        raise ValueError(f"summary 中 {stream} / block 10 / ffn.2 weight 记录应恰有 {len(ARMS)} 条")
    by_arm = {row.get("arm"): row for row in matches}
    body = []
    for arm, label in ARMS:
        row = by_arm.get(arm)
        if row is None or "weight_to_reference" not in row.get("nmse", {}):
            raise ValueError(f"summary 中 weight 记录缺少 arm 或 weight_to_reference: {arm}")
        body.append(f"<tr><th>{esc(label)}</th><td>{number(row['nmse']['weight_to_reference'])}</td></tr>")
    return (
        '<div class="table-wrap"><table><thead><tr><th>actual arm</th>'
        "<th>weight NMSE to same-basis reference</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>"
    )


def build_html(root: Path, manifest: dict, focus: dict, summary: dict) -> str:
    if manifest.get("status") != "complete" or manifest.get("bundle_count") != 12:
        raise ValueError("manifest 必须是 complete 且包含 12 个总览 bundle")
    if focus.get("status") != "complete":
        raise ValueError("focus_manifest 必须是 complete")
    if summary.get("status") != "complete":
        raise ValueError("delivery_summary 必须是 complete")
    figures = manifest.get("figures")
    focus_figures = focus.get("figures")
    if not isinstance(figures, list) or len(figures) != 12:
        raise ValueError("manifest.figures 必须包含 12 组总览")
    if not isinstance(focus_figures, list) or len(focus_figures) != 8:
        raise ValueError("focus_manifest.figures 必须包含 8 张主 focus 图")
    main = summary.get("main_activation_records", {})
    if main.get("case_id") != 1004 or main.get("step") != 4:
        raise ValueError("delivery_summary 主 focus 必须是 case 1004 / step 4")

    focus_map = {}
    for figure in focus_figures:
        if not isinstance(figure, dict):
            raise ValueError("focus figure 必须是 JSON object")
        key = (figure.get("stream"), figure.get("focus_kind"))
        if key in focus_map:
            raise ValueError(f"重复的 focus figure: {key!r}")
        if figure.get("block") != 10 or figure.get("module") != "ffn.2":
            raise ValueError(f"focus figure 不属于 ffn.2 block 10: {figure.get('id')!r}")
        focus_map[key] = (figure, figure_previews(root, figure))
    expected = {(stream, kind) for stream in STREAMS for kind in FOCUS_KINDS}
    if set(focus_map) != expected:
        raise ValueError("focus_manifest 未覆盖两个 expert 的 paired / actual pre / actual post / weight")
    focus_paths = [rel for _, previews in focus_map.values() for rel, _ in previews]
    focus_jpeg_count = len(focus_paths)
    if focus_jpeg_count != 8:
        raise ValueError(f"focus JPEG 数量应为 8，实际为 {focus_jpeg_count}")

    overview_cards = []
    overview_jpeg_count = 0
    overview_paths = []
    for figure in figures:
        if not isinstance(figure, dict):
            raise ValueError("overview figure 必须是 JSON object")
        title = figure.get("title", figure.get("id", "总览图"))
        cards = []
        for page, rel, path in figure_pages(root, figure):
            page_label = f"page {page.get('page', '?')} · {page.get('description', '')}"
            if rel is None or path is None:
                cards.append(
                    '<p class="remote-only">'
                    f"{esc(page_label)}：完整图在远端 PNG/PDF，当前本机只提供 magnitude 预览。"
                    "</p>"
                )
            else:
                label = f"{title} · {page_label}"
                cards.append(image_card(root, rel, path, label, label))
                overview_paths.append(rel)
                overview_jpeg_count += 1
        overview_cards.append(
            f'<article class="bundle"><h3>{esc(title)}</h3><div class="overview-grid">'
            + "".join(cards) + "</div></article>"
        )
    if overview_jpeg_count != 16:
        raise ValueError(f"12 组总览应含 16 张 JPEG，实际为 {overview_jpeg_count}")
    all_preview_paths = focus_paths + overview_paths
    if len(all_preview_paths) != 24 or len(set(all_preview_paths)) != 24:
        raise ValueError("focus 与 overview 必须合计 24 个互不重复的本机 JPEG 路径")

    experts = []
    focus_titles = {
        "paired": "Paired-local：三种 transform",
        "actual_pre": "Actual：A4 前 activation",
        "actual_post": "Actual：A4 后 activation",
        "weight": "Actual weight：五个 arm",
    }
    for stream in STREAMS:
        cards = []
        for kind in FOCUS_KINDS:
            figure, previews = focus_map[(stream, kind)]
            if len(previews) != 1:
                raise ValueError(f"focus figure 应恰有一张 JPEG: {figure.get('id')!r}")
            rel, path = previews[0]
            title = f"{focus_titles[kind]} · case 1004 / step 4 / {stream} / ffn.2 block 10"
            cards.append(image_card(root, rel, path, title, title))
        name = "Video expert" if stream == "video_expert" else "Action expert"
        experts.append(
            f'<section class="expert"><h2>{name}</h2><div class="focus-grid">'
            + "".join(cards)
            + '</div><h3>paired-local 指标</h3>' + paired_table(summary, stream)
            + '<h3>Actual weight 指标</h3>' + weight_table(summary, stream)
            + "</section>"
        )

    local_links = []
    for filename, label in (
        ("protocol.json", "protocol.json"),
        ("manifest.json", "manifest.json"),
        ("focus_manifest.json", "focus_manifest.json"),
        ("delivery_summary.json", "delivery_summary.json"),
    ):
        path = root / filename
        if path.is_file():
            local_links.append(f'<a href="{esc(filename)}">{esc(label)}</a>')

    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fast-WAM Prism 本机图像入口</title>
<style>
:root{color-scheme:light;--ink:#202633;--muted:#657085;--line:#dbe1ea;--paper:#f5f7fb;--card:#fff;--accent:#335d9e}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,"Segoe UI",sans-serif}
main{max-width:1380px;margin:auto;padding:28px 22px 56px}h1{font-size:clamp(1.65rem,3vw,2.3rem);line-height:1.2;margin:0 0 8px}h2{font-size:1.35rem;margin:26px 0 12px}h3{font-size:1rem;margin:18px 0 8px}
p{margin:8px 0}.muted{color:var(--muted)}.panel,.expert,.bundle{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}
.focus-grid{display:grid;grid-template-columns:1fr;gap:12px}.overview-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.figure-card{margin:0;border:1px solid var(--line);border-radius:9px;padding:8px;background:#fff;min-width:0}.figure-card a{display:block}.figure-card img{display:block;width:100%;height:auto;border-radius:5px}.figure-card figcaption{font-size:.86rem;color:var(--muted);padding:7px 2px 1px;display:flex;justify-content:space-between;gap:8px}.remote-only{grid-column:1/-1;border-left:3px solid #b8c7df;padding:8px 10px;margin:0;color:var(--muted);background:#f7f9fc}
.arms{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.arm{padding:5px 9px;background:#eaf0fa;border:1px solid #ced9ec;border-radius:99px;font-size:.88rem}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;border-bottom:1px solid var(--line);padding:7px 9px;white-space:nowrap}thead th{background:#f2f5fa}details{margin-top:24px}summary{cursor:pointer;font-weight:650;font-size:1.1rem;padding:12px 0}code{overflow-wrap:anywhere;background:#edf0f5;padding:2px 5px;border-radius:4px}.local-links{display:flex;flex-wrap:wrap;gap:14px}a{color:var(--accent)}
@media(max-width:760px){main{padding:20px 12px 40px}.focus-grid,.overview-grid{grid-template-columns:1fr}.figure-card figcaption{display:block}}
</style></head><body><main>
<h1>Fast-WAM Prism 本机图像入口</h1>
<p class="muted">主视图固定为 case 1004、step 4、ffn.2 block 10。点击任一 JPEG 可单独打开本机预览。</p>
<section class="panel"><h2>Actual arms</h2><div class="arms"><span class="arm">BF16</span><span class="arm">SpinQuant-adapted · native INT4</span><span class="arm">SmoothScalar · native INT4</span><span class="arm">RowVQ · decoded BF16</span><span class="arm">MicroVQ · decoded BF16</span></div>
<p>两个 VQ arm 的 activation recipe 都是 Smooth + H128 + A4；MicroVQ 主要使用 weight B32 尺度。VQ 图展示 decoded BF16，不代表 native INT4。</p>
</section>
""" + "".join(experts) + """
<section class="panel"><h2>读图范围</h2>
<p>Activation 矩阵轴为 token × channel；weight 矩阵轴为 output × input。配对图在同一层内共享量程，颜色也共享。Actual activation 来自各 arm 自己的模型执行，包含上游量化影响。</p>
<p>表格仅展示 delivery summary 已计算的主窗口数字，不做重算或跨记录平均。CV 使用该窗口；512-channel 窗口的 channel CV 不能代表完整向量的 token 能量分布。正交旋转只在完整向量上守恒总能量。图像更均匀不等于任务 accuracy 更好；本交付没有任务 accuracy 或速度结论。</p>
<p>完整原始数据、metrics、NPZ、full PNG 与 PDF 保存在远端：<code>REMOTE_ROOT</code>；完整图目录：<code>REMOTE_ROOT/plots/</code>。此本机入口只链接已取回的 JSON、protocol 与 JPEG。</p>
<p class="local-links">LOCAL_LINKS</p></section>
<details><summary>展开 12 组 overview（16 张 JPEG）</summary>OVERVIEW</details>
</main></body></html>
""".replace("REMOTE_ROOT", esc(REMOTE_RESULT_ROOT)).replace(
        "LOCAL_LINKS", " · ".join(local_links) if local_links else "未取回本机 JSON/protocol 文件链接"
    ).replace(
        "OVERVIEW", "".join(overview_cards)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", type=Path, required=True, help="本机 plots 目录")
    args = parser.parse_args()
    root = args.plots.resolve()
    if not root.is_dir():
        parser.error(f"plots 目录不存在: {root}")
    try:
        manifest = read_json(root / "manifest.json")
        focus = read_json(root / "focus_manifest.json")
        summary = read_json(root / "delivery_summary.json")
        page = build_html(root, manifest, focus, summary)
        output = root / "gallery.html"
        output.write_text(page, encoding="utf-8", newline="\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {output} ({output.stat().st_size} bytes; 24 unique JPEG paths: 8 focus + 16 overview).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
