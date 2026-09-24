"""守卫测试：backend/package.json 声明的依赖必须真的被仓库引用。

背景
----
``backend/package.json`` 声明了两个依赖：``crypto-js`` 与 ``jsdom``。其中
``jsdom`` 在仓库任何地方都没有被 require/import —— 后端 Python、后端 JS、
前端源码全都搜不到它。

它并非无害：``README.md`` 要求贡献者在 ``backend/`` 下执行 ``npm install``，
这一步会把 ``jsdom`` 及其传递依赖拉进来（实测 ``backend/node_modules`` 共
1290 个文件，其中 ``jsdom`` 自身占 500 个）。这些文件随后还会：

* 在 ``.gitignore`` 缺少 ``node_modules`` 规则时泄漏进 git；
* 被 ``Dockerfile`` 的 ``COPY backend/ ./backend/`` 一并送入构建上下文。

因此加一条守卫：**声明了就必须有人用**。这条测试完全离线、零依赖，只读文件。
"""

import json
import os
import re
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PACKAGE_JSON = BACKEND_DIR / "package.json"

# 可能包含 require/import 语句的文件类型
SCAN_SUFFIXES = {".py", ".js", ".mjs", ".cjs", ".ts", ".vue"}
# 不扫描的目录：依赖、构建产物、版本库元数据、本地缓存
SKIP_DIR_NAMES = {
    "node_modules",
    "dist",
    "dist-ssr",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".npm-cache",
    ".uv-cache",
    ".vite",
}
# 这些文件即使出现依赖名也不算「被引用」
SKIP_FILE_NAMES = {"package.json", "package-lock.json"}


def _declared_dependencies() -> list:
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    declared = set(data.get("dependencies") or {}) | set(data.get("devDependencies") or {})
    return sorted(declared)


def _iter_source_text() -> str:
    """拼接所有可能含 require/import 的源文件内容。

    用 os.walk 并原地剪枝 dirnames，而不是 rglob —— 后者仍会深入
    node_modules 等目录遍历上千个文件，实测会慢一个数量级。
    """
    chunks = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix not in SCAN_SUFFIXES or filename in SKIP_FILE_NAMES:
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def _is_referenced(dependency: str, haystack: str) -> bool:
    escaped = re.escape(dependency)
    patterns = (
        rf'require\(\s*["\']{escaped}["\']\s*\)',
        rf'\bfrom\s+["\']{escaped}["\']',
        rf'\bimport\s+["\']{escaped}["\']',
    )
    return any(re.search(pattern, haystack) for pattern in patterns)


class BackendPackageDependencyTests(unittest.TestCase):
    def test_package_json_declares_at_least_one_dependency(self) -> None:
        # 防止上面两个用例因为「依赖列表为空」而变成空转
        self.assertTrue(_declared_dependencies(), "backend/package.json 未声明任何依赖")

    def test_every_declared_dependency_is_referenced_somewhere(self) -> None:
        declared = _declared_dependencies()
        haystack = _iter_source_text()

        unused = [dep for dep in declared if not _is_referenced(dep, haystack)]

        self.assertEqual(
            unused,
            [],
            f"backend/package.json 声明了仓库中无人使用的依赖: {unused}。"
            f"未使用的依赖会让贡献者的 node_modules 与 Docker 构建上下文无谓膨胀。",
        )


if __name__ == "__main__":
    unittest.main()
