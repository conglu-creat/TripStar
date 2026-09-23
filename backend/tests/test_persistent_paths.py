"""回归测试：容器部署下用户配置与偏好记忆必须落在已挂载的数据卷内。

背景
----
``docker-compose.yaml`` 只挂载了 ``trip_data:/app/backend/data``，但两处持久化
路径都在卷之外：

* ``backend/app/config.py``：运行时配置文件固定在
  ``backend/runtime_settings.json``，容器内即
  ``/app/backend/runtime_settings.json``；
* ``backend/app/memory/sqlite_store.py``：默认 ``./memory.db``，而 ``start.sh``
  会先 ``cd /app``，因此落到 ``/app/memory.db``。

后果：``docker compose down && docker compose up -d --build``（README 介绍的
升级方式）会重建容器并丢掉可写层，于是

* 用户在设置页填写的 LLM / 高德 / 小红书等全部 Key 被静默还原为 compose 里的
  默认值——而 README 正是把设置页作为容器部署的配置方式推荐的；
* ``MEMORY_USE_SQLITE=true`` 时，全部偏好记忆被删除。

``restart: unless-stopped`` 会掩盖这个问题：普通重启保留可写层，只有重建才暴露。

修复方式是让两处路径可被环境变量覆盖，并在 compose 中指向卷内。本文件验证
路径解析的优先级，以及 compose 是否真的完成了接线。

完全离线：不访问网络、不需要 API Key，也不创建任何数据库文件。
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app import config
from app.memory.sqlite_store import SqliteMemoryStore

COMPOSE_FILE = REPO_ROOT / "docker-compose.yaml"
VOLUME_MOUNT = "/app/backend/data"
# 历史默认值写成字面量，而不从模块导入：这样在未修复版本上本文件依然可以被
# 导入，compose 接线与「环境变量是否生效」等用例会**有意义地失败**，而不是整个
# 模块因 ImportError 直接报错、掩盖掉真实信号。
LEGACY_DB_PATH = "./memory.db"


def _clean_env(**overrides) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in overrides}
    env.update(overrides)
    return env


class RuntimeSettingsPathTests(unittest.TestCase):
    def test_defaults_to_legacy_location(self) -> None:
        with mock.patch.dict(os.environ, _clean_env(RUNTIME_SETTINGS_PATH=""), clear=True):
            resolved = config._resolve_runtime_settings_file()

        self.assertEqual(resolved, BACKEND_DIR / "runtime_settings.json")

    def test_env_override_wins(self) -> None:
        target = "/app/backend/data/runtime_settings.json"
        with mock.patch.dict(os.environ, _clean_env(RUNTIME_SETTINGS_PATH=target), clear=True):
            resolved = config._resolve_runtime_settings_file()

        self.assertEqual(resolved, Path(target))

    def test_blank_env_falls_back_to_default(self) -> None:
        with mock.patch.dict(os.environ, _clean_env(RUNTIME_SETTINGS_PATH="   "), clear=True):
            resolved = config._resolve_runtime_settings_file()

        self.assertEqual(resolved, BACKEND_DIR / "runtime_settings.json")


class MemoryDbPathTests(unittest.TestCase):
    def test_defaults_to_legacy_location(self) -> None:
        with mock.patch.dict(os.environ, _clean_env(MEMORY_DB_PATH=""), clear=True):
            store = SqliteMemoryStore()

        self.assertEqual(store.db_path, LEGACY_DB_PATH)

    def test_env_override_wins(self) -> None:
        target = "/app/backend/data/memory.db"
        with mock.patch.dict(os.environ, _clean_env(MEMORY_DB_PATH=target), clear=True):
            store = SqliteMemoryStore()

        self.assertEqual(store.db_path, target)

    def test_explicit_argument_beats_env(self) -> None:
        with mock.patch.dict(
            os.environ, _clean_env(MEMORY_DB_PATH="/app/backend/data/memory.db"), clear=True
        ):
            store = SqliteMemoryStore(db_path="./explicit.db")

        self.assertEqual(store.db_path, "./explicit.db")

    def test_blank_env_falls_back_to_default(self) -> None:
        with mock.patch.dict(os.environ, _clean_env(MEMORY_DB_PATH="  "), clear=True):
            store = SqliteMemoryStore()

        self.assertEqual(store.db_path, LEGACY_DB_PATH)


class ComposeWiringTests(unittest.TestCase):
    """compose 必须真的把这两个变量接上，否则代码侧可配置也没用。"""

    def setUp(self) -> None:
        self.compose = COMPOSE_FILE.read_text(encoding="utf-8")

    def test_volume_mounts_the_data_dir(self) -> None:
        self.assertIn(VOLUME_MOUNT, self.compose)

    def test_runtime_settings_path_is_wired_into_the_volume(self) -> None:
        self.assertIn(
            f"RUNTIME_SETTINGS_PATH={VOLUME_MOUNT}/runtime_settings.json", self.compose
        )

    def test_memory_db_path_is_wired_into_the_volume(self) -> None:
        self.assertIn(f"MEMORY_DB_PATH={VOLUME_MOUNT}/memory.db", self.compose)


if __name__ == "__main__":
    unittest.main()
