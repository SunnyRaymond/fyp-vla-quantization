#!/usr/bin/env python3
"""Build a local HTML index from the computed summary and small JPEG previews."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path, PureWindowsPath


MAX_PREVIEW_BYTES = 220 * 1024
EXPECTED_FAMILIES = {
    "distribution": 16,
    "block_error": 16,
    "input_output_rms": 16,
    "output_error": 16,
    "focus_output_error": 12,
    "aggregate": 4,
}
FAMILY_LABELS = {
    "distribution": "Weight 分布、3D 幅值与全矩阵 RMS",
    "block_error": "B32 block 误差与 scales",
    "input_output_rms": "完整 channel 的输入／输出 RMS",
    "output_error": "固定输入的输出误差",
    "focus_output_error": "重点层的六个输入上下文",
    "aggregate": "所有选定层的配对比较",
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def safe_preview(root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise RuntimeError("manifest page is missing its JPEG preview path")
    normalized = value.replace("\\", "/")
    rel = Path(normalized)
    if rel.is_absolute() or PureWindowsPath(normalized).is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"preview path must stay inside plots/: {value!r}")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"preview path escapes plots/: {value!r}") from exc
    if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg"):
        raise RuntimeError(f"local JPEG preview is missing: {value}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_PREVIEW_BYTES:
        raise RuntimeError(f"JPEG must be 1–220 KiB: {value} ({size} bytes)")
    return rel.as_posix(), path


def validate_bundle(plots: Path, data: dict, manifest: dict) -> tuple[dict, list]:
    if data.get("schema") != "weight-analysis-v1" or data.get("status") != "complete":
        raise RuntimeError("computed summary must be complete and use weight-analysis-v1")
    if manifest.get("status") != "complete":
        raise RuntimeError("plots/manifest.json must have status=complete")
    if manifest.get("module_count") != 16 or manifest.get("expected_module_count") != 16:
        raise RuntimeError("manifest must confirm all 16 selected modules")
    if manifest.get("focus_count") != 2 or manifest.get("aggregate_count") != 4:
        raise RuntimeError("manifest must confirm both focus modules and four aggregate figures")
    if manifest.get("preview_max_bytes") != MAX_PREVIEW_BYTES:
        raise RuntimeError("manifest preview size limit differs from the local gallery requirement")
    if manifest.get("page_count") != 112 or manifest.get("expected_page_count") != 112:
        raise RuntimeError("manifest must confirm all 112 expected plot pages")
    families = manifest.get("families")
    if not isinstance(families, dict):
        raise RuntimeError("manifest.families is missing")
    for family, expected in EXPECTED_FAMILIES.items():
        row = families.get(family, {})
        if row.get("expected") != expected or row.get("complete") != expected:
            raise RuntimeError(f"family {family} must cover {expected} complete figures")
    expected_family_pages = {
        "distribution": {"expected": 48, "complete": 48},
        "block_error": {"expected": 16, "complete": 16},
        "input_output_rms": {"expected": 16, "complete": 16},
        "output_error": {"expected": 16, "complete": 16},
        "focus_output_error": {"expected": 12, "complete": 12},
        "aggregate": {"expected": 4, "complete": 4},
    }
    if manifest.get("family_pages") != expected_family_pages:
        raise RuntimeError("manifest must confirm all required pages in each plot family")
    figures = manifest.get("figures")
    if not isinstance(figures, list) or len(figures) != manifest.get("expected_figure_count"):
        raise RuntimeError("manifest figure list is incomplete")
    ids = [str(row.get("id")) for row in figures]
    if len(set(ids)) != len(ids):
        raise RuntimeError("manifest contains duplicate figure IDs")

    modules = data.get("modules")
    if not isinstance(modules, list) or len(modules) != 16:
        raise RuntimeError("computed summary must list all 16 modules")
    module_ids = {str(row.get("id")) for row in modules}
    by_module = {}
    focus = set()
    all_previews = []
    for figure in figures:
        family = figure.get("family")
        if family not in EXPECTED_FAMILIES:
            raise RuntimeError(f"unexpected figure family: {family!r}")
        module_id = figure.get("module_id")
        if family != "aggregate" and module_id not in module_ids:
            raise RuntimeError(f"figure references an unknown module: {module_id!r}")
        pages = figure.get("pages")
        if not isinstance(pages, list) or not pages:
            raise RuntimeError(f"figure {figure.get('id')} has no pages")
        if family == "distribution" and len(pages) != 3:
            raise RuntimeError("each module distribution receipt must include histogram, |W| surface, and weight RMS pages")
        parsed_pages = []
        for page in pages:
            rel, path = safe_preview(plots, page.get("preview"))
            parsed_pages.append((page, rel, path))
            all_previews.append(rel)
        figure["_pages"] = parsed_pages
        if family == "aggregate":
            continue
        by_module.setdefault(str(module_id), {}).setdefault(family, []).append(figure)
        if family == "focus_output_error":
            focus.add(str(module_id))
    if set(by_module) != module_ids:
        raise RuntimeError("per-module figure set does not cover all 16 modules")
    for module_id, group in by_module.items():
        for family in ("distribution", "block_error", "input_output_rms", "output_error"):
            if len(group.get(family, [])) != 1:
                raise RuntimeError(f"{module_id} must have one {family} figure receipt")
    if len(focus) != 2 or any(len(by_module[module_id].get("focus_output_error", [])) != 6
                               for module_id in focus):
        raise RuntimeError("focus figures must cover six contexts for both FFN.2 / block 10 modules")
    if len(all_previews) != len(set(all_previews)):
        raise RuntimeError("preview paths must be unique")
    return by_module, [row for row in figures if row.get("family") == "aggregate"]


def image_card(page: dict, rel: str, path: Path, title: str) -> str:
    return (
        '<figure class="card">'
        f'<a href="{esc(rel)}" target="_blank" rel="noopener"><img src="{esc(rel)}" '
        f'alt="{esc(title)}" loading="lazy"></a>'
        f'<figcaption>{esc(title)} <span>{path.stat().st_size / 1024:.1f} KiB</span></figcaption>'
        '</figure>'
    )


def pages_html(figures: list[dict]) -> str:
    cards = []
    for figure in figures:
        for page, rel, path in figure["_pages"]:
            title = f"{figure.get('title', figure.get('id'))} · {page.get('description', '')}"
            cards.append(image_card(page, rel, path, title))
    return "".join(cards)


def module_order(data: dict) -> dict:
    rows = data.get("modules", [])
    return {str(row.get("id")): int(row.get("module_index", index))
            for index, row in enumerate(rows)}


def scalar_summary_html(data: dict) -> str:
    # The numeric summary is produced by the compute step; this formatter only displays it.
    summary = (data.get("paired_micro_vs_row_by_stream_suffix")
               or data.get("micro_vs_row_summary") or data.get("paired_summary")
               or data.get("comparison_summary") or {})
    if not isinstance(summary, dict) or not summary:
        return '<p class="muted">JSON 中未附 paired-win / median / range 摘要。</p>'
    cells = []
    def percent(value):
        return f"{value:+.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else esc(value)
    for group, values in summary.items():
        if not isinstance(values, dict):
            continue
        for metric in ("weight", "output"):
            stats = values.get(metric)
            if not isinstance(stats, dict):
                continue
            cells.append(
                f"<tr><th>{esc(group)}</th><td>{esc(metric)} NMSE</td>"
                f"<td>{esc(stats.get('paired_win_count'))}/{esc(stats.get('total'))}</td>"
                f"<td>{percent(stats.get('median_relative_improvement_pct'))}</td>"
                f"<td>{percent(stats.get('min_relative_improvement_pct'))} 至 "
                f"{percent(stats.get('max_relative_improvement_pct'))}</td></tr>"
            )
        trades = values.get("tradeoffs")
        if isinstance(trades, dict):
            cells.append(
                f"<tr><th>{esc(group)}</th><td>指标方向不一致</td>"
                f"<td colspan=2>weight 改善、output 变差："
                f"{esc(trades.get('micro_weight_helps_output_worsens_count'))}</td>"
                f"<td>output 改善、weight 变差："
                f"{esc(trades.get('micro_output_helps_weight_worsens_count'))}</td></tr>"
            )
    if not cells:
        return '<p class="muted">JSON 中未附可显示的 paired-win / median / range 摘要。</p>'
    return ('<div class="table-wrap"><table><thead><tr><th>Expert / 层位置</th><th>指标</th>'
            '<th>Micro 更准的配对数</th><th>改善中位数 (%)</th><th>最小值至最大值 (%)</th>'
            '</tr></thead><tbody>'
            + "".join(cells) + "</tbody></table></div>")


def build_html(plots: Path, data: dict, manifest: dict,
               by_module: dict, aggregates: list[dict]) -> str:
    order = module_order(data)
    module_by_id = {str(row["id"]): row for row in data["modules"]}
    focus_ids = {module_id for module_id, group in by_module.items()
                 if group.get("focus_output_error")}
    focus_sections = []
    for module_id in sorted(focus_ids, key=lambda key: order[key]):
        figures = sorted(by_module[module_id]["focus_output_error"], key=lambda row: str(row["context_id"]))
        focus_sections.append(
            f'<section class="panel"><h3>{esc(module_id)} · block 10 FFN.2 focus</h3>'
            '<div class="focus-grid">' + pages_html(figures) + '</div></section>'
        )

    module_sections = []
    for module_id in sorted(by_module, key=lambda key: order[key]):
        group = by_module[module_id]
        module = module_by_id[module_id]
        figures = [group[family][0] for family in
                   ("distribution", "block_error", "input_output_rms", "output_error")]
        module_sections.append(
            f'<details class="module"><summary>{esc(module_id)}'
            f' <span class="muted">· {esc(module.get("stream"))} · block {esc(module.get("block_index"))}</span></summary>'
            '<div class="module-grid">' + pages_html(figures) + '</div></details>'
        )

    links = []
    for rel, label in (("../weight_analysis_data.json", "computed summary JSON"),
                       ("../weight_protocol.json", "frozen protocol"),
                       ("manifest.json", "plot manifest")):
        target = (plots / rel).resolve()
        if target.is_file():
            links.append(f'<a href="{esc(rel)}">{esc(label)}</a>')
    metadata = data.get("metadata", {})
    window = metadata.get("windows", {}).get("output", {}) if isinstance(metadata, dict) else {}
    token_start, token_count = window.get("token_start", 0), window.get("token_count", 64)
    channel_start, channel_count = window.get("channel_start", 0), window.get("channel_count", 128)
    aggregates = sorted(aggregates, key=lambda row: (row.get("id") != "aggregate_micro_vs_row", str(row.get("id"))))
    aggregate_html = pages_html(aggregates)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fast-WAM Weight Analysis</title>
<style>
:root{{--ink:#202633;--muted:#647187;--line:#dce2ea;--paper:#f5f7fb;--card:#fff;--accent:#365f9e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:auto;padding:26px 20px 56px}}h1{{font-size:clamp(1.7rem,3vw,2.35rem);line-height:1.2;margin:0 0 8px}}h2{{font-size:1.35rem;margin:26px 0 10px}}h3{{font-size:1.05rem;margin:12px 0}}
p{{margin:8px 0}}.muted{{color:var(--muted)}}.panel,.module{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:17px;margin:14px 0}}
.aggregate-grid,.focus-grid,.module-grid{{display:grid;grid-template-columns:1fr;gap:12px}}
.card{{margin:0;border:1px solid var(--line);border-radius:9px;padding:8px;background:#fff;min-width:0}}.card img{{display:block;width:100%;height:auto;border-radius:5px}}
.card figcaption{{font-size:.83rem;color:var(--muted);padding:7px 2px 1px;display:flex;justify-content:space-between;gap:8px}}
.module summary{{cursor:pointer;font-weight:650;font-size:1rem}}.module-grid{{margin-top:12px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:7px 9px;white-space:nowrap}}thead th{{background:#f2f5fa}}
.links{{display:flex;gap:14px;flex-wrap:wrap}}a{{color:var(--accent)}}.note{{border-left:3px solid #89a0c4;padding:8px 12px;background:#f7f9fc}}
@media(max-width:760px){{main{{padding:18px 12px 40px}}.aggregate-grid,.focus-grid,.module-grid{{grid-template-columns:1fr}}.card figcaption{{display:block}}}}
</style></head><body><main>
<h1>Fast-WAM Weight 分布与误差分析</h1>
<p class="muted">16 个 Linear · cases 1004/1014 · steps 0/4/9 · B32 诊断分组 · 主展示输入为 case 1004、step 4。</p>
<section class="panel"><h2>怎样看这些图</h2>
<p>先看 3D |W| 和 channel RMS：数值是否集中在少数位置？再看 B32 误差热图：量化误差落在哪里？最后看固定输入的输出误差：这些 weight 误差有没有影响这份输入实际使用的方向？</p>
<p>每个方法的量化误差均相对自己的同基底 FP reference 计算。SpinQuant 使用 learned rotation，Row-scale VQ 和 MicroVQ 使用同一 frozen Smooth+H128 basis。3D 图使用共同的 128×512 weight 窗口和高度范围；全矩阵 RMS 曲线保留所有 input/output channels，并标出 CV。B32 是逐 output channel 的连续 32 个 input channels；SpinQuant 的实际量化 group 仍为 G128。</p>
<p>输出误差图最多展示 {token_count}×{channel_count}，起点为 token {token_start}、output channel {channel_start}；Video 实际显示 64×128，Action 为 32×128。汇总 NMSE 使用保存的全部 token rows 和完整 output width，并覆盖每层六个 case/step 上下文。</p>
<p class="note">这是固定输入、bias-free FP32 矩阵乘法，输入不做 A4。SpinQuant INT4 与 VQ banks 都离线解码，因此这些图不能直接证明 native kernel 速度、action RMSE 或 closed-loop success 的改善。</p>
<p class="links">{' · '.join(links)}</p></section>
<section class="panel"><h2>Micro 相对 Row 的收益</h2><div class="aggregate-grid">{aggregate_html}</div>
<p>下面直接展示计算节点生成的配对统计。每个 expert/位置有四个不同层的 weight 比较，output 为四层×两输入×三 steps，共 24 个上下文。它们存在依赖，不能当作 24 个独立场景。改善为 (Row NMSE−Micro NMSE)/Row NMSE，正数表示 Micro 更准。</p>
<h3>配对数值摘要</h3>{scalar_summary_html(data)}</section>
<section class="panel"><h2>重点看：block 10 的 FFN.2</h2><p class="muted">Video / Action 各一层，输出误差图覆盖每层六个上下文。</p>{''.join(focus_sections)}</section>
<section class="panel"><h2>展开查看全部 16 层</h2><p class="muted">每层有直方图、3D |W|、完整 weight RMS、B32 误差与 scales、完整 activation RMS，以及主输入的输出误差图。</p>{''.join(module_sections)}</section>
</main></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", type=Path, required=True, help="local plots directory with manifest and previews")
    parser.add_argument("--data", type=Path, help="local weight_analysis_data.json; defaults beside plots/")
    args = parser.parse_args()
    plots = args.plots.resolve()
    if not plots.is_dir():
        parser.error(f"plots directory does not exist: {plots}")
    data_path = args.data.resolve() if args.data else plots.parent / "weight_analysis_data.json"
    try:
        data = read_json(data_path)
        manifest = read_json(plots / "manifest.json")
        by_module, aggregates = validate_bundle(plots, data, manifest)
        page = build_html(plots, data, manifest, by_module, aggregates)
        output = plots / "gallery.html"
        output.write_text(page, encoding="utf-8", newline="\n")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {output} ({output.stat().st_size} bytes; {manifest['figure_count']} figure receipts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
