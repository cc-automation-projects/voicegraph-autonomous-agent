Это математическое "сердце" адаптивности системы. Алгоритм Томпсона (Thompson Sampling) идеально подходит для нашей задачи, так как он естественным образом балансирует **Exploration** (попытка новых, менее проверенных скриптов) и **Exploitation** (использование скрипта, который статистически показывает наилучшую конверсию), используя сопряженные априорные распределения (Beta-распределение для биномиальных исходов: успех/неудача).

---

# 🚀 ЭТАП 3.3: Реализация Thompson Sampling

## Шаг 1: Зависимости

Добавляем необходимые математические библиотеки в `pyproject.toml` микросервиса оркестратора.

```toml
# Добавить в pyproject.toml orchestrator
dependencies = [
    # ... предыдущие зависимости ...
    "numpy>=1.26.4",
    "scipy>=1.13.0"
]
```

---

## Шаг 2: Ядро алгоритма оптимизации (Bandit Optimizer)

Мы создаем выделенный, потокобезопасный класс, который инкапсулирует логику сэмплирования и обновления весов. Это позволяет легко тестировать математику отдельно от логики LangGraph.

**Файл: `src/orchestrator/bandit_optimizer.py`**

```python
import logging
import numpy as np
from scipy.stats import beta
from typing import Dict, List

logger = logging.getLogger(__name__)

class ThompsonSamplingOptimizer:
    """
    Реализация алгоритма Thompson Sampling (Multi-Armed Bandit) 
    для динамического выбора наилучшего скрипта обзвона.
    
    Использует Beta-распределение, где:
    - alpha = количество успехов + 1 (априорное знание)
    - beta = количество неудач + 1 (априорное знание)
    """
    
    def __init__(self, initial_weights: Dict[str, Dict[str, float]] = None):
        """
        Инициализирует оптимизатор.
        :param initial_weights: Словарь вида {"script_id": {"alpha": 1.0, "beta": 1.0}}
        """
        self.weights = initial_weights or {}

    def select_script(self, available_scripts: List[str]) -> str:
        """
        Выбирает скрипт для следующего звонка путем сэмплирования из Beta-распределения.
        
        :param available_scripts: Список доступных ID скриптов (например, из state["active_scripts"])
        :return: ID выбранного скрипта
        """
        if not available_scripts:
            raise ValueError("Список доступных скриптов пуст")

        samples = {}
        for script_id in available_scripts:
            # Если скрипт новый, инициализируем равномерным распределением (alpha=1, beta=1)
            params = self.weights.get(script_id, {"alpha": 1.0, "beta": 1.0})
            
            # Сэмплируем вероятность успеха theta для данного скрипта
            samples[script_id] = beta.rvs(params["alpha"], params["beta"])
        
        # Выбираем скрипт с максимальным сэмплированным значением theta
        best_script = max(samples, key=samples.get)
        
        logger.debug(f"[Bandit] Сэмплированные значения theta: { {k: round(v, 3) for k, v in samples.items()} }. Выбран: {best_script}")
        return best_script

    def update_weights(self, script_id: str, outcome: str) -> Dict[str, Dict[str, float]]:
        """
        Обновляет параметры Alpha/Beta на основе исхода завершенного звонка.
        
        :param script_id: ID скрипта, который использовался
        :param outcome: Результат звонка ('SUCCESS', 'REFUSAL', 'HANGUP', 'ANGRY', etc.)
        :return: Обновленный словарь весов
        """
        if script_id not in self.weights:
            self.weights[script_id] = {"alpha": 1.0, "beta": 1.0}
        
        # Определяем, считается ли исход "успехом" для данной кампании
        # В проде этот список может быть конфигурируемым (например, для NPS успехом считается 9-10)
        success_outcomes = {"SUCCESS", "POSITIVE_NPS", "APPOINTMENT_SET"}
        
        if outcome in success_outcomes:
            self.weights[script_id]["alpha"] += 1.0
            logger.info(f"[Bandit] ✅ Успех для '{script_id}'. Alpha увеличена до {self.weights[script_id]['alpha']}")
        else:
            self.weights[script_id]["beta"] += 1.0
            logger.info(f"[Bandit] ❌ Неудача для '{script_id}'. Beta увеличена до {self.weights[script_id]['beta']}")
        
        return self.weights
```

---

## Шаг 3: Интеграция в `optimizing_node` LangGraph

Теперь мы модифицируем узел `optimizing_node` (из Подзадачи 3.2), чтобы он использовал этот оптимизатор. 

*Примечание:* В реальном асинхронном потоке исход (`outcome`) предыдущего звонка может прийти с задержкой. Для упрощения архитектуры графа мы предполагаем, что `dialer_node` или фоновый слушатель обновляет состояние `last_call_outcome` и `last_called_script_id` перед передачей управления в `optimizing_node`.

**Файл: `src/orchestrator/nodes.py`** (Обновленный фрагмент)

```python
import logging
from typing import Dict, Any
from src.orchestrator.state import CampaignState
from src.orchestrator.bandit_optimizer import ThompsonSamplingOptimizer

logger = logging.getLogger(__name__)

async def optimizing_node(state: CampaignState) -> Dict[str, Any]:
    """
    Обновляет веса Bandit на основе результатов последних звонков 
    и готовит граф к следующей итерации выбора скрипта.
    """
    logger.info("[optimizing_node] Запуск оптимизации весов скриптов...")
    
    # 1. Извлекаем данные о последнем завершенном звонке из состояния
    # (В реальной системе это может подтягиваться из Redis/PostgreSQL по session_id)
    last_script_id = state.get("last_called_script_id")
    last_outcome = state.get("last_call_outcome", "UNKNOWN")
    
    if not last_script_id or last_outcome == "UNKNOWN":
        logger.warning("[optimizing_node] Нет данных о последнем звонке для оптимизации. Пропуск.")
        return {} # Не меняем состояние

    # 2. Инициализируем оптимизатор текущими весами из состояния
    optimizer = ThompsonSamplingOptimizer(initial_weights=state["bandit_weights"])
    
    # 3. Обновляем веса на основе исхода
    updated_weights = optimizer.update_weights(
        script_id=last_script_id, 
        outcome=last_outcome
    )
    
    # 4. Очищаем временные поля состояния, чтобы не засорять его
    return {
        "bandit_weights": updated_weights,
        "last_called_script_id": None,
        "last_call_outcome": "UNKNOWN"
    }
```

*Для полноты картины, `dialer_node` должен сохранять эти временные данные перед отправкой задачи:*
```python
# Внутри dialer_node (фрагмент):
# ... после выбора скрипта и отправки в очередь ...
return {
    "current_user_index": idx + 1,
    "last_called_script_id": script_id,
    # last_call_outcome будет обновлен асинхронным триггером по завершении звонка
}
```

---

## Шаг 4: Модульное тестирование (Доказательство сходимости)

Критически важно доказать, что алгоритм не просто случайно выбирает скрипты, а действительно **сходится** к лучшему варианту, сохраняя при этом способность к исследованию (Exploration).

**Файл: `tests/orchestrator/test_bandit_optimizer.py`**

```python
import pytest
from collections import Counter
from src.orchestrator.bandit_optimizer import ThompsonSamplingOptimizer

def test_initial_weight_update():
    """Проверка корректности инкремента Alpha и Beta."""
    optimizer = ThompsonSamplingOptimizer()
    
    # Первый успех
    optimizer.update_weights("script_A", "SUCCESS")
    assert optimizer.weights["script_A"]["alpha"] == 2.0  # 1 (initial) + 1
    assert optimizer.weights["script_A"]["beta"] == 1.0   # 1 (initial)
    
    # Первая неудача
    optimizer.update_weights("script_A", "REFUSAL")
    assert optimizer.weights["script_A"]["alpha"] == 2.0
    assert optimizer.weights["script_A"]["beta"] == 2.0   # 1 (initial) + 1

def test_convergence_exploitation_vs_exploration():
    """
    Проверка того, что алгоритм сходится к лучшему скрипту (Exploitation), 
    но иногда пробует худший (Exploration).
    """
    optimizer = ThompsonSamplingOptimizer()
    scripts = ["script_good", "script_bad"]
    
    # "Обучаем" бандита: script_good имеет ~90% успеха, script_bad ~10%
    for _ in range(90):
        optimizer.update_weights("script_good", "SUCCESS")
    for _ in range(10):
        optimizer.update_weights("script_good", "REFUSAL")
        
    for _ in range(10):
        optimizer.update_weights("script_bad", "SUCCESS")
    for _ in range(90):
        optimizer.update_weights("script_bad", "REFUSAL")
    
    # Делаем 1000 выборов
    selections = Counter()
    for _ in range(1000):
        chosen = optimizer.select_script(scripts)
        selections[chosen] += 1
        
    # Ожидаем, что "хороший" скрипт будет выбран подавляющее большинство раз (> 85%)
    # из-за того, что его Beta-распределение сильно смещено вправо (высокая theta)
    assert selections["script_good"] > 850, f"Ожидалось >850, получено {selections['script_good']}"
    
    # Ожидаем, что "плохой" скрипт все еще будет выбран несколько раз (Exploration)
    # Это предотвращает застревание в локальном оптимуме, если условия изменятся
    assert 10 < selections["script_bad"] < 150, f"Ожидалось 10-150, получено {selections['script_bad']}"

def test_new_script_handling():
    """Проверка обработки скрипта, которого еще нет в весах."""
    optimizer = ThompsonSamplingOptimizer(initial_weights={"old_script": {"alpha": 5.0, "beta": 2.0}})
    
    # Выбираем из списка, содержащего новый скрипт
    chosen = optimizer.select_script(["old_script", "brand_new_script"])
    assert chosen in ["old_script", "brand_new_script"]
    
    # Обновляем вес нового скрипта
    updated = optimizer.update_weights("brand_new_script", "SUCCESS")
    assert updated["brand_new_script"]["alpha"] == 2.0
    assert updated["brand_new_script"]["beta"] == 1.0
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 3.3)

Прежде чем переходить к **ЭТАПУ 4 (Интеграция эпизодической памяти)**, убедитесь, что:

- [ ] Класс `ThompsonSamplingOptimizer` корректно инициализирует веса `(1.0, 1.0)` для новых скриптов.
- [ ] Метод `update_weights` строго инкрементирует `alpha` для успехов и `beta` для неудач.
- [ ] Метод `select_script` использует `scipy.stats.beta.rvs` и возвращает ID скрипта с максимальным сэмплированным значением.
- [ ] Юнит-тест `test_convergence_exploitation_vs_exploration` стабильно проходит, доказывая, что алгоритм отдает предпочтение скрипту с исторически высокой конверсией (>85% выборов), но сохраняет элемент случайности.
- [ ] `optimizing_node` успешно интегрирован в граф, принимает временные данные о последнем звонке и возвращает обновленный словарь `bandit_weights` в `CampaignState`.
