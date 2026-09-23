"""回归测试：缺少目的地城市的请求必须在入口被拦截（422）。

背景
----
``TripRequest`` 的 ``city`` 默认空字符串、``cities`` 默认空列表，而
``normalize_cities`` 只在「其中一方有值」时做互补填充，**两边都为空时不做任何
检查**，于是这样的请求能通过校验：

* ``POST /api/trip/plan`` 先返回 ``200 {task_id}``；
* 后台任务 ``_run_trip_planning`` 里 ``_build_planner_query()`` 访问
  ``cities[0]``，抛出 ``IndexError: list index out of range``；
* 用户最终通过 WebSocket 看到的是
  「旅行计划生成失败: list index out of range」——一个既无法定位、
  也不提示真正原因的报错。

而 ``trip_planner_agent.py`` 里已经写着「``List[CityStay]`` — 已由
``normalize_cities`` 保证非空」，说明这本就是预期不变量，只是校验没有落实。

本文件完全离线：不访问网络、不需要任何 API Key。
"""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pydantic import ValidationError

from app.models.schemas import TripRequest


def _payload(**overrides):
    """一个除目的地城市外字段齐全的合法请求体。"""
    payload = {
        "start_date": "2026-05-01",
        "end_date": "2026-05-02",
        "travel_days": 2,
        "transportation": "公共交通",
        "accommodation": "经济型酒店",
        "preferences": ["历史文化"],
        "language": "zh",
    }
    payload.update(overrides)
    return payload


class TripRequestCityValidationTests(unittest.TestCase):
    """单元层：``cities`` 非空是后续流程依赖的不变量。"""

    def test_city_only_is_normalized_to_cities(self) -> None:
        request = TripRequest(**_payload(city="北京"))

        self.assertEqual([(c.city, c.days) for c in request.cities], [("北京", 2)])

    def test_cities_only_is_normalized_to_city(self) -> None:
        request = TripRequest(**_payload(cities=[{"city": "西安", "days": 3}]))

        self.assertEqual(request.city, "西安")

    def test_missing_city_and_cities_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TripRequest(**_payload())

    def test_empty_cities_without_city_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TripRequest(**_payload(cities=[]))

    def test_empty_city_string_without_cities_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TripRequest(**_payload(city=""))

    def test_multi_city_request_is_untouched(self) -> None:
        request = TripRequest(
            **_payload(cities=[{"city": "北京", "days": 2}, {"city": "西安", "days": 3}])
        )

        self.assertEqual([c.city for c in request.cities], ["北京", "西安"])
        self.assertEqual(request.city, "北京")


class PlanEndpointValidationTests(unittest.TestCase):
    """接口层：非法请求应在入口得到 422，而不是 200 之后异步失败。"""

    def _client(self):
        from fastapi.testclient import TestClient

        from app.api.main import app

        return TestClient(app)

    def test_plan_without_city_returns_422(self) -> None:
        response = self._client().post("/api/trip/plan", json=_payload())

        self.assertEqual(
            response.status_code,
            422,
            f"缺少目的地城市应被校验拦截，实际返回 {response.status_code}: {response.text[:200]}",
        )

    def test_validation_error_names_the_city_field(self) -> None:
        response = self._client().post("/api/trip/plan", json=_payload())

        self.assertIn("city", response.text.lower())


if __name__ == "__main__":
    unittest.main()
