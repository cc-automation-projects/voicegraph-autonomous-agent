from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from composio_langchain import Action, App, ComposioToolSet

logger = logging.getLogger(__name__)


class ComposioIntegration:
    def __init__(self, api_key: str = ""):
        self.toolset = ComposioToolSet(api_key=api_key) if api_key else ComposioToolSet()
        self._tools: List[Any] = []

    def get_available_apps(self) -> List[str]:
        return [app.value for app in App]

    def get_tools(self, app_names: List[str]) -> List[Any]:
        tools = []
        for app_name in app_names:
            try:
                app_enum = App[app_name.upper()]
                app_tools = self.toolset.get_tools(apps=[app_enum])
                tools.extend(app_tools)
                logger.info(f"Загружены инструменты для {app_name}: {len(app_tools)} шт.")
            except (KeyError, Exception) as e:
                logger.warning(f"Не удалось загрузить инструменты для {app_name}: {e}")
        self._tools = tools
        return tools

    async def execute_action(self, action: Action, params: Dict[str, Any]) -> Any:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.toolset.execute_action, action, params)
            return result
        except Exception as e:
            logger.error(f"Ошибка выполнения действия {action}: {e}")
            raise
