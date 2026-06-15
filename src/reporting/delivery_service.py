from __future__ import annotations

import logging
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List

import aiosmtplib

logger = logging.getLogger(__name__)


class ReportDeliveryService:
    def __init__(
        self,
        smtp_host: str = "smtp.yandex.ru",
        smtp_port: int = 465,
        smtp_user: str = "",
        smtp_password: str = "",
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password

    async def send_report(self, to_emails: List[str], pdf_path: str) -> bool:
        if not pdf_path or not Path(pdf_path).exists():
            logger.error("PDF-файл не найден")
            return False

        msg = MIMEMultipart("mixed")
        msg["From"] = self.smtp_user
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = "VoiceGraph — Еженедельный отчёт"

        body = MIMEText("Во вложении еженедельный отчёт VoiceGraph.", "plain", "utf-8")
        msg.attach(body)

        with open(pdf_path, "rb") as f:
            attachment = MIMEApplication(f.read(), _subtype="pdf")
            attachment.add_header("Content-Disposition", "attachment", filename=Path(pdf_path).name)
            msg.attach(attachment)

        try:
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                use_tls=True,
            )
            logger.info(f"Отчёт отправлен на {to_emails}")
            return True
        except Exception as e:
            logger.error(f"Ошибка отправки отчёта: {e}")
            return False
