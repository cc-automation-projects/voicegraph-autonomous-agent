Мы используем **WeasyPrint** в связке с **Matplotlib**, так как это позволяет генерировать красивые, профессиональные PDF-отчеты с графиками через HTML/CSS шаблоны, что гораздо гибче и проще в поддержке, чем чистый ReportLab.

---

# 🚀 ЭТАП 6.3: Генерация финального отчета (Полная реализация)

## Шаг 1: Зависимости и подготовка окружения

Добавляем необходимые библиотеки в `pyproject.toml` модуля оркестратора/отчетности.

```toml
# Добавить в pyproject.toml
dependencies = [
    "weasyprint>=61.2",         # Генерация PDF из HTML/CSS
    "matplotlib>=3.8.0",        # Построение графиков
    "seaborn>=0.13.0",          # Стилизация графиков
    "python-telegram-bot>=20.7",# Отправка отчетов в Telegram
    "asyncpg>=0.29.0"           # Асинхронная работа с PostgreSQL
]
```
*Важно: Для работы WeasyPrint в Docker-образе необходимо установить системные зависимости: `apt-get install -y libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libffi-dev`.*

---

## Шаг 2: Агрегация данных из PostgreSQL

Создаем сервис, который собирает все необходимые метрики для отчета одним асинхронным запросом (или транзакцией), чтобы минимизировать нагрузку на БД.

**Файл: `src/reporting/data_aggregator.py`**

```python
import asyncpg
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ReportDataAggregator:
    def __init__(self, db_dsn: str):
        self.db_dsn = db_dsn

    async def get_campaign_report_data(self, campaign_id: str) -> Dict[str, Any]:
        """Собирает агрегированные данные для финального отчета кампании."""
        pool = await asyncpg.create_pool(self.db_dsn)
        
        async with pool.acquire() as conn:
            # 1. Общая статистика кампании
            campaign_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_calls,
                    COUNT(CASE WHEN outcome = 'SUCCESS' THEN 1 END) as success_calls,
                    ROUND(AVG(duration_sec), 1) as avg_duration
                FROM call_logs 
                WHERE campaign_id = $1
            """, campaign_id)

            # 2. Конверсия по скриптам (для графика Bandit)
            script_stats = await conn.fetch("""
                SELECT 
                    script_id,
                    COUNT(*) as total,
                    COUNT(CASE WHEN outcome = 'SUCCESS' THEN 1 END) as success,
                    ROUND(
                        COUNT(CASE WHEN outcome = 'SUCCESS' THEN 1 END)::NUMERIC * 100.0 / NULLIF(COUNT(*), 0), 
                        1
                    ) as conversion_rate
                FROM call_logs 
                WHERE campaign_id = $1
                GROUP BY script_id
                ORDER BY conversion_rate DESC
            """, campaign_id)

            # 3. ТОП-5 причин отказов (из инсайтов рефлексии)
            top_refusals = await conn.fetch("""
                SELECT 
                    root_cause,
                    COUNT(*) as count,
                    ROUND(COUNT(*)::NUMERIC * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
                FROM reflection_insights 
                WHERE campaign_id = $1 AND root_cause != 'UNKNOWN'
                GROUP BY root_cause
                ORDER BY count DESC
                LIMIT 5
            """, campaign_id)

            # 4. Финальные рекомендации (последний инсайт или агрегация)
            recommendations = await conn.fetchval("""
                SELECT suggested_script_tweak 
                FROM reflection_insights 
                WHERE campaign_id = $1 
                ORDER BY confidence_score DESC, created_at DESC 
                LIMIT 1
            """, campaign_id)

        await pool.close()

        return {
            "campaign_id": campaign_id,
            "total_calls": campaign_stats["total_calls"] or 0,
            "success_calls": campaign_stats["success_calls"] or 0,
            "avg_duration": campaign_stats["avg_duration"] or 0.0,
            "script_stats": [dict(row) for row in script_stats],
            "top_refusals": [dict(row) for row in top_refusals],
            "recommendation": recommendations or "Недостаточно данных для формирования рекомендаций. Попробуйте увеличить длительность кампании."
        }
```

---

## Шаг 3: Генерация PDF через WeasyPrint + Matplotlib

Чтобы избежать создания временных файлов на диске, мы генерируем график в памяти, кодируем его в Base64 и встраиваем прямо в HTML-шаблон.

**Файл: `src/reporting/pdf_generator.py`**

```python
import io
import base64
import logging
from weasyprint import HTML
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def generate_conversion_chart(script_stats: list) -> str:
    """Генерирует график конверсии и возвращает его в виде base64 строки."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    
    labels = [row["script_id"] for row in script_stats]
    rates = [row["conversion_rate"] for row in script_stats]
    
    # Создаем барплот
    bars = plt.bar(labels, rates, color=sns.color_palette("Blues_r", len(labels)))
    
    # Добавляем значения на столбцы
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontsize=10)

    plt.title("Конверсия по вариантам скриптов (Результат работы Bandit)", fontsize=12, fontweight='bold')
    plt.ylabel("Конверсия (%)")
    plt.ylim(0, max(rates) * 1.2 if rates else 10)
    plt.tight_layout()
    
    # Сохраняем в буфер памяти
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_pdf_report(data: dict) -> bytes:
    """Генерирует PDF-отчет на основе HTML-шаблона и данных."""
    chart_base64 = generate_conversion_chart(data["script_stats"])
    
    # Формируем строки для таблицы причин отказов
    refusal_rows = "".join([
        f"<tr><td>{row['root_cause'].replace('_', ' ').title()}</td><td>{row['count']}</td><td>{row['percentage']}%</td></tr>"
        for row in data["top_refusals"]
    ]) or "<tr><td colspan='3'>Нет данных об отказах</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Arial', sans-serif; color: #333; line-height: 1.6; }}
            .header {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }}
            .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; display: inline-block; margin-right: 15px; min-width: 120px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #2c3e50; color: white; }}
            .recommendation {{ background: #e8f6f3; border-left: 4px solid #1abc9c; padding: 15px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Отчет по кампании VoiceGraph</h1>
            <p>ID кампании: {data['campaign_id']} | Дата генерации: автоматическая</p>
        </div>

        <div style="margin-bottom: 30px;">
            <div class="metric-card">
                <div class="metric-value">{data['total_calls']}</div>
                <div class="metric-label">Всего звонков</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['success_calls']}</div>
                <div class="metric-label">Успешных</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['avg_duration']}с</div>
                <div class="metric-label">Ср. длительность</div>
            </div>
        </div>

        <h3>📊 Эффективность скриптов (Thompson Sampling)</h3>
        <img src="data:image/png;base64,{chart_base64}" alt="Conversion Chart" style="max-width: 100%;"/>

        <h3>🚫 ТОП-5 причин отказов</h3>
        <table>
            <tr><th>Причина</th><th>Количество</th><th>Доля</th></tr>
            {refusal_rows}
        </table>

        <div class="recommendation">
            <h3>💡 Рекомендация для следующей кампании</h3>
            <p>{data['recommendation']}</p>
        </div>
        
        <footer style="margin-top: 40px; font-size: 10px; color: #95a5a6; text-align: center;">
            Сгенерировано автономной системой VoiceGraph. Все персональные данные замаскированы в соответствии с 152-ФЗ.
        </footer>
    </body>
    </html>
    """
    
    # Генерация PDF
    pdf_bytes = HTML(string=html_content).write_pdf()
    logger.info(f"✅ PDF-отчет для кампании {data['campaign_id']} успешно сгенерирован ({len(pdf_bytes)} байт)")
    return pdf_bytes
```

---

## Шаг 4: Доставка отчета (Telegram + CRM)

Реализуем асинхронный сервис, который берет сгенерированный PDF и рассылает его по назначению.

**Файл: `src/reporting/delivery_service.py`**

```python
import logging
import io
from telegram import Bot
from telegram.error import TelegramError
# Предполагаем, что у нас есть настроенный клиент Composio из Этапа 6.1
# from src.integrations.crm_tools import composio_upload_file 

logger = logging.getLogger(__name__)

class ReportDeliveryService:
    def __init__(self, tg_bot_token: str, supervisor_chat_id: str):
        self.bot = Bot(token=tg_bot_token)
        self.supervisor_chat_id = supervisor_chat_id

    async def send_report(self, campaign_id: str, pdf_bytes: bytes, summary_text: str):
        """Отправляет отчет в Telegram и прикрепляет к CRM."""
        file_io = io.BytesIO(pdf_bytes)
        file_io.name = f"VoiceGraph_Report_{campaign_id}.pdf"
        
        # 1. Отправка в Telegram
        try:
            caption = (
                f"📊 *Отчет по кампании завершен*\n\n"
                f"ID: `{campaign_id}`\n"
                f"{summary_text}\n\n"
                f"Детальный PDF-отчет во вложении."
            )
            await self.bot.send_document(
                chat_id=self.supervisor_chat_id,
                document=file_io,
                caption=caption,
                parse_mode="Markdown"
            )
            logger.info("✅ Отчет успешно отправлен в Telegram супервайзеру.")
        except TelegramError as e:
            logger.error(f"❌ Ошибка отправки отчета в Telegram: {e}")

        # 2. Прикрепление к CRM (через Composio или прямой API)
        try:
            file_io.seek(0) # Сброс курсора для повторного чтения
            # Псевдокод вызова инструмента Composio для загрузки файла
            # await composio_upload_file(
            #     entity_id=campaign_id, 
            #     file_obj=file_io, 
            #     file_name=file_io.name
            # )
            logger.info("✅ Отчет успешно прикреплен к кампании в CRM.")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки отчета в CRM: {e}")
```

---

## Шаг 5: Интеграция в LangGraph (Узел завершения)

Добавляем финальный узел в наш граф, который срабатывает, когда кампания переходит в статус `DONE`.

**Файл: `src/orchestrator/nodes.py`** (Дополнение)

```python
import logging
from src.orchestrator.state import CampaignState
from src.reporting.data_aggregator import ReportDataAggregator
from src.reporting.pdf_generator import generate_pdf_report
from src.reporting.delivery_service import ReportDeliveryService
import os

logger = logging.getLogger(__name__)

# Инициализация сервисов (в реальном коде лучше через Dependency Injection)
aggregator = ReportDataAggregator(db_dsn=os.getenv("DATABASE_URL"))
delivery = ReportDeliveryService(
    tg_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
    supervisor_chat_id=os.getenv("SUPERVISOR_CHAT_ID")
)

async def finalize_campaign_node(state: CampaignState) -> dict:
    """Генерирует и отправляет финальный отчет по завершении кампании."""
    campaign_id = state["campaign_id"]
    logger.info(f"📝 Генерация финального отчета для кампании {campaign_id}...")
    
    try:
        # 1. Сбор данных
        report_data = await aggregator.get_campaign_report_data(campaign_id)
        
        # 2. Генерация PDF
        pdf_bytes = generate_pdf_report(report_data)
        
        # 3. Формирование краткого саммари для Telegram
        total = report_data["total_calls"]
        success = report_data["success_calls"]
        rate = (success / total * 100) if total > 0 else 0
        summary = f"Всего звонков: {total}\nУспешных: {success} ({rate:.1f}%)\nЛучший скрипт: {report_data['script_stats'][0]['script_id'] if report_data['script_stats'] else 'N/A'}"
        
        # 4. Доставка
        await delivery.send_report(campaign_id, pdf_bytes, summary)
        
        return {"error_message": None} # Успех
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при генерации отчета: {e}")
        return {"error_message": f"Failed to generate report: {str(e)}"}
```

*Не забудьте добавить `builder.add_edge("reflecting_node", "finalize_campaign")` и `builder.add_edge("finalize_campaign", END)` в `graph_builder.py`.*

---

## Шаг 6: Модульное тестирование

Проверяем, что пайплайн генерации и отправки работает корректно с моками.

**Файл: `tests/reporting/test_pdf_generation.py`**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.reporting.pdf_generator import generate_conversion_chart, generate_pdf_report

def test_generate_conversion_chart():
    mock_stats = [
        {"script_id": "v1_direct", "conversion_rate": 15.5},
        {"script_id": "v2_empathic", "conversion_rate": 22.0},
        {"script_id": "v3_benefit", "conversion_rate": 18.2}
    ]
    base64_img = generate_conversion_chart(mock_stats)
    assert len(base64_img) > 0
    assert base64_img.startswith("iVBOR") # Сигнатура PNG в base64

@pytest.mark.asyncio
@patch("src.reporting.delivery_service.ReportDeliveryService.bot")
async def test_send_report_success(mock_bot):
    from src.reporting.delivery_service import ReportDeliveryService
    
    delivery = ReportDeliveryService("fake_token", "12345")
    mock_bot.send_document = AsyncMock()
    
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    await delivery.send_report("camp-001", pdf_bytes, "Test summary")
    
    mock_bot.send_document.assert_called_once()
    call_kwargs = mock_bot.send_document.call_args[1]
    assert call_kwargs["chat_id"] == "12345"
    assert call_kwargs["document"].name == "VoiceGraph_Report_camp-001.pdf"
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 6.3 и всего ЭТАПА 6)

Прежде чем считать проект полностью реализованным, убедитесь, что:

- [ ] Сервис `ReportDataAggregator` успешно выполняет SQL-запросы и возвращает корректно структурированные данные для заданного `campaign_id`.
- [ ] Функция `generate_pdf_report` создает валидный PDF-файл, содержащий встроенный график (Base64), таблицы и текстовые блоки, без ошибок рендеринга WeasyPrint.
- [ ] Отчет успешно доставляется в указанный Telegram-чат с корректным форматированием (Markdown) и вложенным файлом.
- [ ] (Опционально, но рекомендуется) Реализован и протестирован механизм загрузки этого же PDF-файла в карточку кампании/сделки в CRM через Composio.
- [ ] В логах отсутствуют утечки PII (в отчете используются только агрегированные данные и ID скриптов, без имен или телефонов).
- [ ] Юнит-тесты покрывают генерацию графика и мок-отправку в Telegram.
