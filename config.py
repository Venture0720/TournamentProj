# config.py
# Этот файл сохранён для обратной совместимости.
# Все константы перенесены в data_loader.py (KAZAKHSTAN_REAL_DATA).
# CONFIG используется в app.py — оставлен как пустой импорт-заглушка.

from dataclasses import dataclass


@dataclass
class AppConfig:
    """Минимальный конфиг — только то, что реально используется в app.py."""
    pass


CONFIG = AppConfig()
