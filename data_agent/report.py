"""
Comprehensive Diagnostic Report Generator.
Produces structured Markdown and beautiful self-contained HTML diagnostic reports.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .schema import SchemaReport
from .cleaning import DataCleaner
from .eda import EDAReport
from .models.trainer import TournamentReport
from .explainability import ExplainabilityReport

class DiagnosticReportGenerator:
    """Generates rich diagnostic reports in Markdown and self-contained HTML."""

    @staticmethod
    def generate_markdown(
        schema: SchemaReport,
        cleaner: DataCleaner,
        eda: EDAReport,
        tournament: TournamentReport,
        shap_report: ExplainabilityReport,
    ) -> str:
        md = []
        md.append("# 🤖 Autonomous Data Science Diagnostic Report\n")
        md.append(f"**Task Type**: `{schema.task_type}` | **Target Column**: `{schema.target_column}` | **Total Records**: {schema.total_rows:,}\n")

        # 1. Dataset Health & Schema Summary
        md.append("## 1. Schema & Data Health Summary\n")
        md.append(f"- **Rows**: {schema.total_rows:,} | **Initial Columns**: {schema.total_columns}")
        md.append(f"- **Missing Cells**: {eda.total_missing_cells:,} | **Duplicate Rows**: {eda.duplicate_rows:,}")
        md.append(f"- **Memory Footprint**: {eda.memory_usage_mb:.2f} MB")
        md.append(f"- **Numeric Features**: {len(schema.numeric_features)} | **Categorical Features**: {len(schema.categorical_features)}")
        if schema.dropped_features:
            md.append(f"- **Anomalous Features Dropped (ID/Constant)**: `{', '.join(schema.dropped_features)}`\n")

        # 2. Automated Cleaning Operations
        md.append("## 2. Automated Cleaning & Imputation Pipeline\n")
        md.append(f"- **Imputed Columns**: {len(cleaner.impute_values)}")
        md.append(f"- **Outlier Bounds Applied (Winsorization)**: {len(cleaner.outlier_bounds)} columns")
        if cleaner.missing_indicators:
            md.append(f"- **Missingness Indicators Created**: `{', '.join(cleaner.missing_indicators)}`\n")

        # 3. Model Tournament Leaderboard
        md.append("## 3. Stratified Cross-Validation Tournament (4 Model Families)\n")
        md.append(f"**Champion Model**: 🏆 **{tournament.champion_model_name}**\n")
        md.append(tournament.leaderboard.to_markdown(index=False))
        md.append("\n")

        # 4. SHAP Explainability
        md.append("## 4. Granular SHAP Explainability & Attribution\n")
        md.append(f"### Top Global Features ({shap_report.model_name})\n")
        top_table = pd.DataFrame([
            {"Rank": i + 1, "Feature": f, "Mean |SHAP| Impact": round(shap_report.global_importance[f], 4)}
            for i, f in enumerate(shap_report.top_features[:8])
        ])
        md.append(top_table.to_markdown(index=False))
        md.append("\n")

        md.append("### Automated Model Insights\n")
        for insight in shap_report.insights:
            md.append(f"- 💡 {insight}")

        return "\n".join(md)

    @staticmethod
    def generate_html(
        schema: SchemaReport,
        cleaner: DataCleaner,
        eda: EDAReport,
        tournament: TournamentReport,
        shap_report: ExplainabilityReport,
        output_filepath: str,
    ):
        md_content = DiagnosticReportGenerator.generate_markdown(schema, cleaner, eda, tournament, shap_report)
        leaderboard_html = tournament.leaderboard.to_html(classes="styled-table", index=False)

        top_features_rows = "".join([
            f"<tr><td>{i+1}</td><td><strong>{f}</strong></td><td>{shap_report.global_importance[f]:.4f}</td></tr>"
            for i, f in enumerate(shap_report.top_features[:8])
        ])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Autonomous Data Science Diagnostic Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #2d3748; background: #f7fafc; padding: 40px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        h1 {{ color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        h2 {{ color: #2b6cb0; margin-top: 32px; border-bottom: 1px solid #edf2f7; padding-bottom: 8px; }}
        .badge {{ background: #ebf8ff; color: #2b6cb0; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.9em; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
        .stat-card {{ background: #f7fafc; border: 1px solid #e2e8f0; padding: 16px; border-radius: 8px; text-align: center; }}
        .stat-val {{ font-size: 24px; font-weight: 700; color: #2b6cb0; }}
        .stat-lbl {{ font-size: 13px; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }}
        .styled-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95em; }}
        .styled-table thead tr {{ background-color: #2b6cb0; color: #ffffff; text-align: left; }}
        .styled-table th, .styled-table td {{ padding: 12px 15px; border-bottom: 1px solid #e2e8f0; }}
        .styled-table tbody tr:nth-of-type(even) {{ background-color: #f7fafc; }}
        .insight-box {{ background: #feebc8; border-left: 4px solid #dd6b20; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Autonomous ML Pipeline Diagnostic Report</h1>
        <p>
            <span class="badge">Task: {schema.task_type}</span>
            <span class="badge">Target: {schema.target_column}</span>
            <span class="badge">Champion: {tournament.champion_model_name}</span>
        </p>

        <div class="stats-grid">
            <div class="stat-card"><div class="stat-val">{schema.total_rows:,}</div><div class="stat-lbl">Total Samples</div></div>
            <div class="stat-card"><div class="stat-val">{schema.total_columns}</div><div class="stat-lbl">Input Features</div></div>
            <div class="stat-card"><div class="stat-val">{eda.total_missing_cells:,}</div><div class="stat-lbl">Missing Cells Imputed</div></div>
            <div class="stat-card"><div class="stat-val">{len(shap_report.top_features)}</div><div class="stat-lbl">Selected Features</div></div>
        </div>

        <h2>🏆 Model Tournament Leaderboard (Stratified Cross-Validation)</h2>
        {leaderboard_html}

        <h2>🔍 Granular SHAP Feature Explainability</h2>
        <table class="styled-table">
            <thead><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP| Impact</th></tr></thead>
            <tbody>
                {top_features_rows}
            </tbody>
        </table>

        <div class="insight-box">
            <strong>Key Automated Insights:</strong>
            <ul>
                {"".join([f"<li>{i}</li>" for i in shap_report.insights])}
            </ul>
        </div>
    </div>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html)
        return output_filepath
