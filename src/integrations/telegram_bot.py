from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.reporting.pdf_generator import PDFReportGenerator

logger = logging.getLogger(__name__)


class VoiceGraphTelegramBot:
    def __init__(self, token: str, report_generator: PDFReportGenerator):
        self.token = token
        self.report_generator = report_generator
        self.application = Application.builder().token(token).build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("report", self.report_command))
        self.application.add_handler(CommandHandler("status", self.status_command))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Добро пожаловать в VoiceGraph Bot!\n\n"
            "Команды:\n"
            "/report - сгенерировать PDF-отчёт\n"
            "/status - статус активных кампаний"
        )

    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Генерация отчёта...")
        try:
            pdf_path = await self.report_generator.generate_weekly_report_pdf()
            with open(pdf_path, "rb") as f:
                await update.message.reply_document(f, filename="weekly_report.pdf")
        except Exception as e:
            logger.error(f"Ошибка генерации отчёта: {e}")
            await update.message.reply_text(f"Ошибка: {e}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Статус: VoiceGraph активен и работает.")

    def run(self, webhook_url: str = "", port: int = 8443):
        if webhook_url:
            self.application.run_webhook(
                listen="0.0.0.0",
                port=port,
                webhook_url=webhook_url,
            )
        else:
            self.application.run_polling()
