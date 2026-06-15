from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from weasyprint import HTML

from src.reporting.data_aggregator import DataAggregator

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def generate_conversion_chart(script_stats: List[Dict[str, Any]]) -> str:
    sns.set_theme(style="whitegrid", palette="Blues_d")

    names = [s.get("script_id", f"Script {i}") for i, s in enumerate(script_stats)]
    values = [s.get("conversion_rate", 0) * 100 for s in script_stats]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=sns.color_palette("Blues_d", len(names)))
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Script Conversion Rates")
    ax.bar_label(bars, fmt="%.1f%%", padding=2)
    sns.despine()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>VoiceGraph Weekly Report</title>
<style>
  body {{ font-family: 'DejaVu Sans', sans-serif; font-size: 12pt; color: #333; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #2e86c1; margin-top: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
  th {{ background-color: #2980b9; color: white; }}
  .kpi {{ display: inline-block; margin: 8px; padding: 12px;
         background: #eaf2f8; border-radius: 6px; min-width: 150px; }}
  .kpi-value {{ font-size: 24pt; font-weight: bold; color: #1a5276; }}
  .kpi-label {{ font-size: 10pt; color: #7f8c8d; }}
  .footer {{ margin-top: 40px; font-size: 9pt; color: #95a5a6; text-align: center; }}
</style>
</head>
<body>
<h1>VoiceGraph — Еженедельный отчёт</h1>
<p>Период: {period}<br>Сгенерирован: {generated_at}</p>

<div class="kpi"><div class="kpi-value">{total_calls}</div><div class="kpi-label">Всего звонков</div></div>
<div class="kpi"><div class="kpi-value">{active_campaigns}</div><div class="kpi-label">Активных кампаний</div></div>
<div class="kpi"><div class="kpi-value">{total_conversions}</div><div class="kpi-label">Конверсий</div></div>
<div class="kpi"><div class="kpi-value">{conversion_rate}%</div><div class="kpi-label">Конверсия</div></div>

<h2>Конверсия по скриптам</h2>
<img src="data:image/png;base64,{chart_base64}" alt="Conversion Chart" style="width:100%;max-width:700px;">

<h2>Ежедневная статистика</h2>
<table>
<tr><th>Дата</th><th>Звонков</th><th>Успешно</th><th>Договорённостей</th><th>Ср. длительность</th></tr>
{daily_rows}
</table>

<div class="footer">VoiceGraph Autonomous Predictive Outbound Agent &copy; 2026</div>
</body>
</html>
"""


class PDFReportGenerator:
    def __init__(self, aggregator: DataAggregator, output_dir: str = "reports"):
        self.aggregator = aggregator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_weekly_report_pdf(self, campaign_id: str | None = None) -> str:
        if campaign_id:
            kpi = await self.aggregator.campaign_summary(campaign_id)
            script_stats = await self.aggregator.script_conversion_stats(campaign_id)
        else:
            kpi = await self.aggregator.weekly_kpi_report()
            script_stats = []

        daily_rows = ""
        for day in kpi.get("daily_stats", []):
            daily_rows += (
                f"<tr><td>{day['date']}</td>"
                f"<td>{day['calls']}</td>"
                f"<td>{day['success']}</td>"
                f"<td>{day['agreements']}</td>"
                f"<td>{day['avg_duration_sec']}s</td></tr>"
            )
        chart_base64 = generate_conversion_chart(script_stats) if script_stats else ""

        html_content = REPORT_TEMPLATE.format(
            period="7 дней",
            generated_at=kpi.get("generated_at", datetime.now().isoformat()),
            total_calls=kpi.get("total_calls", 0),
            active_campaigns=kpi.get("active_campaigns", 0),
            total_conversions=kpi.get("total_conversions", 0),
            conversion_rate=round(kpi.get("overall_conversion_rate", 0) * 100, 2),
            chart_base64=chart_base64,
            daily_rows=daily_rows,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"weekly_report_{timestamp}.pdf"

        HTML(string=html_content).write_pdf(str(output_path))
        logger.info(f"PDF-отчёт сохранён: {output_path}")

        return str(output_path)
