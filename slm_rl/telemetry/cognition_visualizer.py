"""Interactive HTML Cognition Inspector and Telemetry Dashboard for SLM RL."""

import argparse
import json
import os
import re
from typing import Any, Dict, List


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLM-RL: Cognition & Reasoning Telemetry</title>
    <style>
        :root {
            --bg: #090d16;
            --card-bg: #111827;
            --border: #1f2937;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --green: #10b981;
            --gold: #f59e0b;
            --purple: #8b5cf6;
            --red: #ef4444;
            --text: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 32px 24px; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
        .title-group h1 { font-size: 26px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
        .badge { background: var(--accent-glow); color: #60a5fa; border: 1px solid #3b82f6; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
        .subtitle { color: var(--text-muted); font-size: 14px; margin-top: 6px; }
        
        .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }
        .stat-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; position: relative; overflow: hidden; }
        .stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent); }
        .stat-card.green::before { background: var(--green); }
        .stat-card.gold::before { background: var(--gold); }
        .stat-card.purple::before { background: var(--purple); }
        .stat-label { font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
        .stat-value { font-size: 28px; font-weight: 700; margin-top: 8px; color: #fff; }

        .section-title { font-size: 18px; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        .chart-container { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 32px; }

        .timeline-table { width: 100%; border-collapse: collapse; background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
        .timeline-table th { background: #1a2234; padding: 12px 16px; text-align: left; font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
        .timeline-table td { padding: 14px 16px; border-top: 1px solid var(--border); font-size: 13px; }
        .timeline-table tr:hover { background: rgba(255, 255, 255, 0.02); }

        .think-box { background: #0b1120; border-left: 3px solid var(--purple); padding: 12px 14px; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; line-height: 1.5; color: #cbd5e1; margin-top: 6px; }
        .highlight-aha { background: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 2px 6px; border-radius: 4px; font-weight: 700; border: 1px solid rgba(245, 158, 11, 0.4); }
        .answer-badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; }
        .answer-badge.correct { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; }
        .answer-badge.wrong { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-group">
                <h1>🧠 SLM-RL Cognition & Reasoning Inspector <span class="badge">Live Telemetry</span></h1>
                <p class="subtitle">Real-time verification metrics, test-time compute, and thinking diagnostics for Small Language Models</p>
            </div>
        </div>

        <div class="grid-stats">
            <div class="stat-card green">
                <div class="stat-label">Final Mean Reward</div>
                <div class="stat-value">{final_reward:.3f}</div>
            </div>
            <div class="stat-card gold">
                <div class="stat-label">Test-Time Compute (Words)</div>
                <div class="stat-value">{avg_think_words:.1f}</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-label">Policy Entropy</div>
                <div class="stat-value">{policy_entropy:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">KL Drift vs Ref</div>
                <div class="stat-value">{kl_div:.4f}</div>
            </div>
        </div>

        <div class="section-title">📊 Step-by-Step Training Telemetry</div>
        <div class="chart-container">
            <table class="timeline-table">
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Reward</th>
                        <th>Policy Loss</th>
                        <th>KL Divergence</th>
                        <th>Entropy</th>
                        <th>Think Words</th>
                        <th>Active Gradient %</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(log_file: str, output_html: str) -> None:
    """Generate interactive HTML report from metrics.jsonl."""
    records = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass

    if not records:
        print(f"Warning: No valid records found in {log_file}")
        records = [{
            "step": 1,
            "mean_reward": 0.0,
            "policy_loss": 0.0,
            "kl_divergence": 0.0,
            "policy_entropy": 2.0,
            "avg_think_words": 0.0,
            "active_learning_fraction": 1.0,
        }]

    last_r = records[-1]
    final_reward = last_r.get("mean_reward", 0.0)
    avg_think_words = last_r.get("avg_think_words", 0.0)
    policy_entropy = last_r.get("policy_entropy", 0.0)
    kl_div = last_r.get("kl_divergence", 0.0)

    rows = []
    for r in reversed(records[-15:]):  # show last 15 steps
        step = r.get("step", 0)
        rew = r.get("reward_mean", r.get("mean_reward", 0.0))
        loss = r.get("loss", r.get("policy_loss", 0.0))
        kl = r.get("kl_divergence", 0.0)
        ent = r.get("policy_entropy", 0.0)
        words = r.get("avg_think_words", 0.0)
        act = r.get("active_learning_fraction", 1.0) * 100.0

        extra_diag = ""
        if "pivot_rate" in r:
            extra_diag = f" <span style='color:#ef4444;'>[pivots: {r['pivot_rate']:.1f}]</span>"
        elif "dpo_margin" in r:
            extra_diag = f" <span style='color:#3b82f6;'>[margin: {r['dpo_margin']:.2f}]</span>"

        row = (
            f"<tr>"
            f"<td><b>#{step}</b>{extra_diag}</td>"
            f"<td style='color:#10b981; font-weight:600;'>{rew:.3f}</td>"
            f"<td>{loss:.3f}</td>"
            f"<td style='color:#8b5cf6;'>{kl:.4f}</td>"
            f"<td>{ent:.2f}</td>"
            f"<td style='color:#06b6d4;'>{words:.1f}</td>"
            f"<td>{act:.0f}%</td>"
            f"</tr>"
        )
        rows.append(row)

    html_content = (
        HTML_TEMPLATE
        .replace("{final_reward}", f"{final_reward:.3f}")
        .replace("{avg_think_words}", f"{avg_think_words:.1f}")
        .replace("{policy_entropy}", f"{policy_entropy:.2f}")
        .replace("{kl_div}", f"{kl_div:.4f}")
        .replace("{table_rows}", "\n".join(rows))
    )


    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✓ Interactive Cognition Inspector HTML generated at: {output_html}")


def main():
    parser = argparse.ArgumentParser(description="Generate Interactive Cognition HTML Report")
    parser.add_argument("--log-file", type=str, default="runs/slm_rl/metrics.jsonl",
                        help="Path to metrics.jsonl file")
    parser.add_argument("--output", type=str, default="runs/slm_rl/cognition_report.html",
                        help="Path to save HTML report")
    args = parser.parse_args()
    generate_html_report(args.log_file, args.output)


if __name__ == "__main__":
    main()
