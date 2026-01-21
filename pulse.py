from pathlib import Path
from datetime import datetime
import textwrap

import pandas as pd
import plotly.express as px
import plotly.io as pio

# ========= CONFIG =========
INPUT_CSV = Path("numbers.csv")
OUTPUT_HTML = Path("pulse_report_styled_white.html")

BRAND_NAME = "Pulse XCap/MXI2"

ACCENT = "#64748b"          # brand red "#64748b" #ff2d2d
ACCENT_DARK = "#c81f1f"
GOOD = "#1a7f37"            # green
NEUTRAL = "#8a8f98"         # gray
BAD = "#d1242f"             # red

# Layout constants
MAX_TITLE_CHARS = 52        # wrap chart titles at ~52 chars
CHART_HEIGHT = 320          # per chart (fits titles better)

# ========= HELPERS =========
def wrap_title(s: str, width: int = MAX_TITLE_CHARS) -> str:
    s = str(s).strip()
    return "<br>".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))

def score_color(score: float) -> str:
    """
    Map 1..5 to BAD..GOOD. Uses thresholds:
      >= 4.0 good
      >= 3.0 neutral
      else bad
    """
    if pd.isna(score):
        return NEUTRAL
    if score >= 4.0:
        return GOOD
    if score >= 3.0:
        return NEUTRAL
    return BAD

# ========= LOAD =========
df = pd.read_csv(INPUT_CSV)

timestamp_col = next((c for c in df.columns if c.lower().startswith("timestamp")), df.columns[0])
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

comment_col = next((c for c in df.columns if "optional" in c.lower() or "comment" in c.lower()), None)

question_cols = [c for c in df.columns if c not in {timestamp_col, comment_col}]
for c in question_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

n = len(df)
date_min = df[timestamp_col].min()
date_max = df[timestamp_col].max()
date_range_str = ""
if pd.notna(date_min) and pd.notna(date_max):
    date_range_str = f"{date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}"

# Summary stats
stacked = df[question_cols].stack(dropna=True) if question_cols else pd.Series(dtype=float)
overall_mean = float(stacked.mean()) if len(stacked) else float("nan")

summary = pd.DataFrame({
    "Question": question_cols,
    "Mean": [df[c].mean() for c in question_cols],
    "Median": [df[c].median() for c in question_cols],
    "Min": [df[c].min() for c in question_cols],
    "Max": [df[c].max() for c in question_cols],
}).sort_values("Mean", ascending=False)

best_q = summary.iloc[0]["Question"] if len(summary) else "-"
best_mean = float(summary.iloc[0]["Mean"]) if len(summary) else float("nan")
worst_q = summary.iloc[-1]["Question"] if len(summary) else "-"
worst_mean = float(summary.iloc[-1]["Mean"]) if len(summary) else float("nan")

# Overall distribution pie (across all questions + respondents)
dist_counts = (
    stacked.round()
    .astype(int)
    .value_counts()
    .reindex([1, 2, 3, 4, 5], fill_value=0)
)
dist_df = dist_counts.rename_axis("Score").reset_index(name="Count")
total_answers = int(dist_df["Count"].sum())
dist_df["Percent"] = (dist_df["Count"] / total_answers * 100) if total_answers else 0

# ========= PLOTLY THEME (WHITE) =========
template = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", color="#0f172a"),
        margin=dict(l=24, r=18, t=70, b=36),
        xaxis=dict(
            gridcolor="rgba(15,23,42,0.08)",
            zerolinecolor="rgba(15,23,42,0.10)",
            tickfont=dict(color="#334155"),
            title=dict(font=dict(color="#475569")),
        ),
        yaxis=dict(
            gridcolor="rgba(15,23,42,0.08)",
            zerolinecolor="rgba(15,23,42,0.10)",
            tickfont=dict(color="#334155"),
            title=dict(font=dict(color="#475569")),
        ),
        legend=dict(font=dict(color="#334155")),
    )
)

def dist_chart(col_name: str):
    counts = (
        df[col_name]
        .value_counts(dropna=False)
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        .rename_axis("Score")
        .reset_index(name="Responses")
    )

    fig = px.bar(
        counts,
        x="Score",
        y="Responses",
        title=wrap_title(col_name),
        text="Responses",
    )
    fig.update_layout(
        template=template,
        height=CHART_HEIGHT,
        xaxis=dict(dtick=1, title="Score (1–5)"),
        yaxis=dict(title="Responses"),
        title=dict(font=dict(size=14)),
    )
    fig.update_traces(
        marker_color=ACCENT,
        marker_line_color="rgba(15,23,42,0.15)",
        marker_line_width=1,
        textfont_color="#0f172a",
        hovertemplate="Score=%{x}<br>Responses=%{y}<extra></extra>",
    )
    return fig

def pie_chart():
    if total_answers == 0:
        fig = px.pie(
            names=["No data"],
            values=[1],
            title="Overall distribution (1–5)",
            hole=0.45,
        )
        fig.update_traces(
            marker=dict(colors=["#e2e8f0"], line=dict(color="#ffffff", width=2)),
            textinfo="label",
            textposition="inside",
        )
    else:
        fig = px.pie(
            dist_df,
            names="Score",
            values="Count",
            title="Overall distribution of all answers (1–5)",
            hole=0.45,
        )

        colors = [
            "#d1d5db",  # 1
            "#cbd5e1",  # 2
            "#94a3b8",  # 3
            "#64748b",  # 4
            "#475569",  # 5
        ]

        fig.update_traces(
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            textinfo="percent+label",
            hovertemplate="Score=%{label}<br>Count=%{value}<br>%{percent}<extra></extra>",
            sort=False,  # keeps 1..5 order
            textposition="inside",            # prevents outside labels -> removes leader lines
            insidetextorientation="radial",   # nicer inside text layout
            textfont_size=14,
        )

        # Hide labels automatically when they don't fit (fatias pequenas),
        # instead of pushing them outside and drawing leader lines.
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode="hide")

    fig.update_layout(
        template=template,
        height=340,
        margin=dict(l=12, r=12, t=70, b=12),
        title=dict(font=dict(size=14)),
        legend=dict(orientation="h", y=-0.05),
        showlegend=True,
    )
    return fig


# ========= COMMENTS =========
comments_html = ""
if comment_col:
    comments = df[comment_col].dropna().astype(str).str.strip()
    comments = comments[comments != ""]
    if len(comments):
        items = "\n".join([f"<div class='comment'>“{c}”</div>" for c in comments])
        comments_html = f"""
        <section class="panel span-12">
          <div class="panel-header">
            <div class="panel-title">Open-ended comments</div>
            <div class="panel-subtitle">{len(comments)} comment(s)</div>
          </div>
          <div class="panel-body comments-grid">
            {items}
          </div>
        </section>
        """

# ========= SUMMARY TABLE HTML =========
def fmt(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

rows = []
for _, r in summary.iterrows():
    rows.append(f"""
      <tr>
        <td class="q">{r['Question']}</td>
        <td>{fmt(r['Mean'])}</td>
        <td>{fmt(r['Median'])}</td>
        <td>{fmt(r['Min'])}</td>
        <td>{fmt(r['Max'])}</td>
      </tr>
    """)

summary_table_html = f"""
<div class="table-wrap">
  <table class="table">
    <thead>
      <tr>
        <th>Question</th>
        <th>Mean</th>
        <th>Median</th>
        <th>Min</th>
        <th>Max</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
"""

# ========= KPI COLORS =========
overall_color = score_color(overall_mean)
best_color = score_color(best_mean)   # likely green
worst_color = score_color(worst_mean) # likely red/neutral

# ========= CHART HTML =========
charts_html = "".join(
    [f"<div class='chart-card'>{pio.to_html(dist_chart(c), include_plotlyjs=False, full_html=False)}</div>" for c in question_cols]
)

pie_html = pio.to_html(pie_chart(), include_plotlyjs=False, full_html=False)

# ========= HTML =========
html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{BRAND_NAME} · Pulse Survey</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>

  <style>
    :root {{
      --accent: {ACCENT};
      --accent-dark: {ACCENT_DARK};
      --good: {GOOD};
      --neutral: {NEUTRAL};
      --bad: {BAD};

      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: rgba(15, 23, 42, 0.10);
      --text: #0f172a;
      --muted: #475569;
      --shadow: 0 18px 60px rgba(15, 23, 42, 0.10);
      --shadow-soft: 0 10px 28px rgba(15, 23, 42, 0.08);
      --radius: 18px;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: var(--text);
      background:
        radial-gradient(900px 450px at 20% 0%, rgba(255,45,45,0.10), transparent 60%),
        radial-gradient(900px 450px at 80% 0%, rgba(255,45,45,0.06), transparent 60%),
        linear-gradient(180deg, #ffffff 0%, var(--bg) 35%, var(--bg) 100%);
      min-height: 100vh;
    }}

    .nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255,255,255,0.88);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(15,23,42,0.10);
    }}

    .nav-inner {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      justify-content: flex-start;
      gap: 12px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}

    .brand-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #ff2d2d;
      box-shadow: 0 0 0 4px rgba(255,45,45,0.14);
    }}

    .container {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 26px 18px 46px;
    }}

    .hero {{
      border: 1px solid rgba(15,23,42,0.10);
      background: rgba(255,255,255,0.85);
      border-radius: calc(var(--radius) + 6px);
      padding: 22px 22px;
      box-shadow: var(--shadow-soft);
    }}

    .hero-title {{
      font-size: 28px;
      font-weight: 750;
      margin: 0 0 6px;
    }}

    .hero-sub {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}

    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border: 1px solid rgba(15,23,42,0.10);
      border-radius: 999px;
      background: rgba(255,255,255,0.80);
    }}

    .pill .badge {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: rgba(15,23,42,0.30);
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
      margin-top: 16px;
    }}

    .kpi {{
      grid-column: span 3;
      border: 1px solid rgba(15,23,42,0.10);
      background: rgba(255,255,255,0.92);
      border-radius: var(--radius);
      padding: 14px 14px;
      box-shadow: var(--shadow-soft);
    }}

    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
    }}

    .kpi .value {{
      font-size: 28px;
      font-weight: 750;
      letter-spacing: -0.4px;
      line-height: 1.1;
    }}

    .kpi .hint {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere; /* prevents "espremido" */
    }}

    .panel {{
      border: 1px solid rgba(15,23,42,0.10);
      background: rgba(255,255,255,0.92);
      border-radius: var(--radius);
      box-shadow: var(--shadow-soft);
      overflow: hidden;
    }}

    .panel-header {{
      padding: 14px 14px;
      border-bottom: 1px solid rgba(15,23,42,0.08);
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }}

    .panel-title {{
      font-weight: 700;
      font-size: 14px;
    }}

    .panel-subtitle {{
      color: var(--muted);
      font-size: 12px;
    }}

    .panel-body {{
      padding: 12px 12px 14px;
    }}

    .span-12 {{ grid-column: span 12; }}
    .span-8 {{ grid-column: span 8; }}
    .span-4 {{ grid-column: span 4; }}
    .span-6 {{ grid-column: span 6; }}

    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    .table th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
      padding: 10px 10px;
      border-bottom: 1px solid rgba(15,23,42,0.10);
      background: rgba(15,23,42,0.02);
    }}

    .table td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(15,23,42,0.06);
      color: rgba(15,23,42,0.92);
      vertical-align: top;
    }}

    .table td.q {{
      max-width: 560px;
      white-space: normal;
      word-break: break-word;
      line-height: 1.35;
    }}

    .table tr:hover td {{
      background: rgba(15,23,42,0.02);
    }}

    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .chart-card {{
      border: 1px solid rgba(15,23,42,0.10);
      background: rgba(255,255,255,0.98);
      border-radius: 16px;
      padding: 10px 10px 0;
      overflow: hidden;
    }}

    .comments-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}

    .comment {{
      border: 1px solid rgba(15,23,42,0.10);
      background: rgba(15,23,42,0.02);
      border-radius: 14px;
      padding: 12px 12px;
      color: rgba(15,23,42,0.90);
      line-height: 1.55;
      overflow-wrap: anywhere; /* fixes cramped long text */
    }}

    .footer {{
      margin-top: 18px;
      color: rgba(15,23,42,0.45);
      font-size: 12px;
      text-align: center;
    }}

    @media (max-width: 980px) {{
      .kpi {{ grid-column: span 6; }}
      .charts-grid {{ grid-template-columns: 1fr; }}
      .comments-grid {{ grid-template-columns: 1fr; }}
      .span-8, .span-4, .span-6 {{ grid-column: span 12; }}
    }}
  </style>
</head>

<body>
  <div class="nav">
    <div class="nav-inner">
      <div class="brand">
        <span class="brand-dot"></span>
        <span>{BRAND_NAME}</span>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="hero">
      <div class="hero-title">LP Pulse Survey</div>
      <p class="hero-sub">A consolidated snapshot of sentiment and feedback. Built for quick scanning and executive review.</p>
      <div class="meta-row">
        <span class="pill"><span class="badge"></span>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
        <span class="pill"><span class="badge"></span>Responses: {n}</span>
        <span class="pill"><span class="badge"></span>Date range: {date_range_str or "-"}</span>
      </div>
    </div>

    <div class="grid">
      <div class="kpi">
        <div class="label">Overall mean (1–5)</div>
        <div class="value" style="color:{overall_color};">{overall_mean:.2f}</div>
        <div class="hint">Rule of thumb: ≥ 4.0 good, 3.0–3.99 neutral, &lt; 3.0 needs attention.</div>
      </div>

      <div class="kpi">
        <div class="label">Best question (mean)</div>
        <div class="value" style="color:{best_color};">{best_mean:.2f}</div>
        <div class="hint">{best_q}</div>
      </div>

      <div class="kpi">
        <div class="label">Lowest question (mean)</div>
        <div class="value" style="color:{worst_color};">{worst_mean:.2f}</div>
        <div class="hint">{worst_q}</div>
      </div>

      <div class="kpi">
        <div class="label">Total answers</div>
        <div class="value">{total_answers/5}</div>
        <div class="hint">Responses × questions (excluding blanks).</div>
      </div>

      <section class="panel span-12">
        <div class="panel-header">
          <div class="panel-title">Summary by question</div>
          <div class="panel-subtitle">Mean / median / min / max</div>
        </div>
        <div class="panel-body">
          {summary_table_html}
        </div>
      </section>

      <section class="panel span-12">
        <div class="panel-header">
          <div class="panel-title">Overall distribution</div>
          <div class="panel-subtitle">All answers aggregated (1–5)</div>
        </div>
        <div class="panel-body">
          <div class="chart-card" style="padding: 10px 10px 0; border-radius: 14px;">
            {pie_html}
          </div>
        </div>
      </section>

      <section class="panel span-12">
        <div class="panel-header">
          <div class="panel-title">Distributions by question</div>
          <div class="panel-subtitle">Counts by score (1–5). Titles are wrapped for readability.</div>
        </div>
        <div class="panel-body">
          <div class="charts-grid">
            {charts_html}
          </div>
        </div>
      </section>

      {comments_html}
    </div>

    <div class="footer">Internal report · {BRAND_NAME}</div>
  </div>
</body>
</html>
"""

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"Saved: {OUTPUT_HTML.resolve()}")
