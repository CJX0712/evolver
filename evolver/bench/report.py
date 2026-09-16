"""Single-file HTML report for a benchmark run.

Self-contained by design: inline CSS, hand-rolled SVG charts, no CDN, no build
step, no JavaScript required to read it. Open it, email it, commit it -- it
renders identically everywhere and keeps working in five years.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence

from evolver.bench.runner import BenchmarkResult


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _nice(values: Sequence[float]) -> tuple:
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.15
    return max(0.0, lo - pad), hi + pad


def line_chart(
    values: Sequence[float],
    *,
    width: int = 560,
    height: int = 190,
    color: str = "#5b9dff",
    label: str = "",
    fmt: str = "{:.0f}",
    fill: bool = True,
) -> str:
    """A dependency-free SVG line chart."""
    if not values:
        return ""
    pad_l, pad_r, pad_t, pad_b = 46, 16, 18, 30
    lo, hi = _nice(values)
    n = len(values)
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def xy(i: int, v: float) -> tuple:
        x = pad_l + (plot_w * i / max(1, n - 1))
        y = pad_t + plot_h * (1 - (v - lo) / max(1e-9, hi - lo))
        return x, y

    pts = [xy(i, v) for i, v in enumerate(values)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pad_l},{pad_t + plot_h} " + poly + f" {pad_l + plot_w},{pad_t + plot_h}"

    grid = []
    for g in range(4):
        gy = pad_t + plot_h * g / 3
        gv = hi - (hi - lo) * g / 3
        grid.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" class="axis" text-anchor="end">'
            f"{fmt.format(gv)}</text>"
        )

    dots = []
    for i, (x, y) in enumerate(pts):
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="dot" style="fill:{color}">'
            f"<title>epoch {i}: {fmt.format(values[i])}</title></circle>"
        )
        dots.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 18:.1f}" class="axis" '
            f'text-anchor="middle">e{i}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart" role="img"
     aria-label="{_esc(label)}">
  <defs>
    <linearGradient id="g{abs(hash(color)) % 9999}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>
      <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  {''.join(grid)}
  {'<polygon points="' + area + '" fill="url(#g' + str(abs(hash(color)) % 9999) + ')" />' if fill else ''}
  <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.4"
            stroke-linejoin="round" stroke-linecap="round"/>
  {''.join(dots)}
  <text x="{pad_l}" y="{height - 4}" class="axis">{_esc(label)}</text>
</svg>"""


def _kpi(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    return (
        f'<div class="kpi {tone}">'
        f'<div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value">{_esc(value)}</div>'
        f'<div class="kpi-sub">{_esc(sub)}</div>'
        f"</div>"
    )


CSS = """
:root{
  --bg:#0d1117; --panel:#161b22; --panel-2:#1c2330; --line:#2a3340;
  --fg:#e6edf3; --muted:#8b98a8; --accent:#5b9dff; --good:#3fb950;
  --warn:#d29922; --bad:#f85149; --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",
  "PingFang SC","Microsoft YaHei",sans-serif;}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px 72px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:38px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 20px}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;
  font-weight:600;letter-spacing:.02em}
.badge.sim{background:rgba(210,153,34,.16);color:var(--warn);border:1px solid rgba(210,153,34,.35)}
.badge.live{background:rgba(63,185,80,.16);color:var(--good);border:1px solid rgba(63,185,80,.35)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:22px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.kpi-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.kpi-value{font:600 27px/1.25 var(--mono);margin:7px 0 3px}
.kpi-sub{font-size:12.5px;color:var(--muted)}
.kpi.good .kpi-value{color:var(--good)}
.kpi.accent .kpi-value{color:var(--accent)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.axis{fill:var(--muted);font:11px var(--mono)}
.dot{stroke:var(--bg);stroke-width:2}
table{width:100%;border-collapse:collapse;font:13.5px var(--mono);margin-top:6px}
th,td{padding:9px 11px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;
   letter-spacing:.04em;background:var(--panel-2)}
tbody tr:hover{background:rgba(91,157,255,.06)}
.up{color:var(--good)} .down{color:var(--bad)}
code{font:12.5px var(--mono);background:var(--panel-2);padding:2px 6px;border-radius:5px;
  color:#a5d6ff}
.note{background:var(--panel);border-left:3px solid var(--warn);border-radius:0 10px 10px 0;
  padding:14px 18px;margin:18px 0;color:#c9d3de;font-size:14px}
.note strong{color:var(--fg)}
.skills{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{background:var(--panel-2);border:1px solid var(--line);border-radius:7px;
  padding:5px 11px;font:12.5px var(--mono);color:#a5d6ff}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px}
"""


def render_html(result: BenchmarkResult, *, title: str = "Evolver · Evolution Report") -> str:
    """Render a complete standalone HTML report."""
    s = result.summary()
    epochs = result.epochs
    if not epochs:
        return "<html><body><p>No benchmark data.</p></body></html>"

    sr = [e.success_rate * 100 for e in epochs]
    st = [e.avg_steps for e in epochs]
    tk = [e.avg_tokens for e in epochs]
    hits = [e.skill_hit_rate * 100 for e in epochs]

    provider = (result.config.get("llm", {}) or {}).get("provider", "replay")
    simulated = provider == "replay"
    badge = (
        '<span class="badge sim">SIMULATED AGENT · mechanism validation</span>'
        if simulated
        else '<span class="badge live">LIVE MODEL · real evaluation</span>'
    )

    rows = []
    prev = None
    for e in epochs:
        d_sr = "" if prev is None else f'{e.success_rate - prev.success_rate:+.0%}'
        d_tk = "" if prev is None else f'{e.avg_tokens - prev.avg_tokens:+.0f}'
        cls_sr = "up" if prev is not None and e.success_rate > prev.success_rate else (
            "down" if prev is not None and e.success_rate < prev.success_rate else "")
        cls_tk = "up" if prev is not None and e.avg_tokens < prev.avg_tokens else (
            "down" if prev is not None and e.avg_tokens > prev.avg_tokens else "")
        rows.append(
            f"<tr><td>e{e.epoch}</td><td>{e.success_rate:.0%}</td>"
            f"<td class='{cls_sr}'>{d_sr}</td>"
            f"<td>{e.avg_steps:.2f}</td><td>{e.avg_tokens:.0f}</td>"
            f"<td class='{cls_tk}'>{d_tk}</td>"
            f"<td>{e.skill_count}</td><td>{e.skill_hit_rate:.0%}</td>"
            f"<td>{e.pitfalls_active}</td></tr>"
        )
        prev = e

    skills_html = "".join(f'<span class="chip">{_esc(n)}</span>' for n in result.skill_names)

    note = (
        "<strong>Read this before quoting these numbers.</strong> This run used "
        "the deterministic <code>replay</code> adapter -- a scripted agent, not a "
        "language model. It demonstrates that the <em>mechanism</em> works: skills "
        "and pitfalls are distilled, retrieved, and measurably improve outcome, "
        "steps, and tokens. It says nothing about how a specific frontier model "
        "behaves. For that, run with <code>--provider openai</code> or "
        "<code>--provider anthropic</code>."
        if simulated
        else "<strong>Live model run.</strong> Numbers come from real model calls; "
             "cost reflects the token accounting reported by the provider."
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>Evolver · 进化报告</h1>
  <p class="sub">{badge} &nbsp;·&nbsp; {len(epochs)} epochs × {len(result.task_ids)} tasks
     &nbsp;·&nbsp; {s.get('wall_time_s', 0)}s wall clock</p>

  <div class="kpis">
    {_kpi("成功率 Success rate", s.get("success_rate", "n/a"), "pitfall 驱动", "good")}
    {_kpi("平均步数 Avg steps", s.get("avg_steps", "n/a"), f"↓ {s.get('step_reduction','')}", "accent")}
    {_kpi("平均 Token", s.get("avg_tokens", "n/a"), f"↓ {s.get('token_reduction','')}", "accent")}
    {_kpi("习得技能 / 教训", f"{s.get('skills_learned', 0)} / {s.get('pitfalls_learned', 0)}",
          "skills / pitfalls", "neutral")}
  </div>

  <h2>进化曲线</h2>
  <div class="charts">
    <div class="panel">
      {line_chart(sr, color="#3fb950", label="success rate %", fmt="{:.0f}")}
    </div>
    <div class="panel">
      {line_chart(st, color="#5b9dff", label="avg steps per task", fmt="{:.1f}")}
    </div>
    <div class="panel">
      {line_chart(tk, color="#d29922", label="avg tokens per task", fmt="{:.0f}")}
    </div>
    <div class="panel">
      {line_chart(hits, color="#a371f7", label="skill hit rate %", fmt="{:.0f}")}
    </div>
  </div>

  <h2>逐轮明细</h2>
  <table>
    <thead><tr>
      <th>epoch</th><th>成功</th><th>Δ</th><th>步数</th><th>tokens</th><th>Δ</th>
      <th>技能数</th><th>命中率</th><th>教训数</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2>习得的技能库</h2>
  <div class="skills">{skills_html or '<span class="chip">（无）</span>'}</div>

  <h2>方法论与局限</h2>
  <div class="note">{note}</div>
  <div class="note">
    两种学习信号的贡献是可分离的：<strong>成功率</strong>主要由 Pitfall（失败教训）驱动，
    因为被警告过的 agent 会绕开已知陷阱；<strong>步数与 Token</strong>主要由 Skill（成功技能）驱动，
    因为蒸馏出的函数把多步推导压缩成一次调用。若只升成功率而步数不降，说明 pitfall 生效而
    skill 未生效 —— 这比笼统的"变好了"更有诊断价值。
  </div>

  <footer>
    Generated by Evolver · 单文件 HTML，无外部依赖 ·
    复现命令 <code>evolver bench --epochs {len(epochs)}</code>
  </footer>
</div>
</body>
</html>"""


def write_report(result: BenchmarkResult, path: str, **kw) -> str:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(result, **kw), encoding="utf-8")
    return str(p)
