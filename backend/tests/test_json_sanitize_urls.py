"""回归测试：JSON 清理不得破坏字符串值中的 URL。

背景
----
``_sanitize_json_str`` 用 ``re.sub(r'//[^\\n]*', '', json_str)`` 删除 JS 风格
行注释。这个正则不区分「JSON 字符串内部」与「JSON 结构位置」，而
``https://...`` 里天然含有 ``//``，因此任何出现在字符串值中的 URL 都会把
该行剩余内容（含结尾引号）一起吞掉，结果是：

* 多行（格式化）输出：字符串变成 ``"…预约入口见 https:`` 后跟换行，
  本地修复全部失败，最终走到 LLM 修复兜底，失败则整个请求以
  「行程 JSON 解析失败」结束；
* 单行（紧凑）输出：整段剩余响应被删除，``_repair_truncated_json`` 会把它
  补成"合法但缺数据"的 JSON —— 静默丢掉行程内容，比直接报错更危险。

而 ``PLANNER_AGENT_PROMPT`` 明确要求把 ``reservation_tips`` 原样透传
（含预约渠道信息），所以 URL 是预期内容，不是异常输入。

本文件完全离线：不访问网络、不需要任何 API Key。
"""

import json
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

from app.agents import trip_planner_agent as tpa
from app.models.schemas import TripRequest

BOOKING_URL = "https://gugong.ktmtech.cn"


def _bare_planner() -> "tpa.MultiAgentTripPlanner":
    """只装配测试所需的最小状态，不触发 __init__ 的外部依赖。"""
    return object.__new__(tpa.MultiAgentTripPlanner)


def _plan_with_url(city: str = "北京") -> str:
    """构造一份合法行程，其中景点预约提示里带有真实预约链接。"""
    return json.dumps(
        {
            "city": city,
            "cities": [city],
            "start_date": "2026-05-01",
            "end_date": "2026-05-01",
            "overall_suggestions": f"{city}行程建议",
            "days": [
                {
                    "date": "2026-05-01",
                    "day_index": 0,
                    "city": city,
                    "description": f"{city}第一天行程",
                    "transportation": "公共交通",
                    "accommodation": "经济型酒店",
                    "attractions": [
                        {
                            "name": "故宫博物院",
                            "address": "北京市东城区景山前街4号",
                            "location": {"longitude": 116.397128, "latitude": 39.916527},
                            "visit_duration": 180,
                            "description": "需提前预约，预约入口见官方渠道",
                            "ticket_price": 60,
                            "reservation_required": True,
                            "reservation_tips": f"请提前 7 天在官网预约：{BOOKING_URL}",
                        }
                    ],
                    "meals": [],
                }
            ],
            "weather_info": [],
            "budget": {
                "total_attractions": 60,
                "total_hotels": 0,
                "total_meals": 0,
                "total_transportation": 0,
                "total_inter_city_transport": 0,
                "total": 60,
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def _trip_request(city: str = "北京") -> TripRequest:
    return TripRequest(
        city=city,
        start_date="2026-05-01",
        end_date="2026-05-01",
        travel_days=1,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
        language="zh",
    )


class SanitizePreservesStringContentTests(unittest.TestCase):
    """单元层：``_sanitize_json_str`` 只应删除字符串之外的注释。"""

    def setUp(self) -> None:
        self.planner = _bare_planner()

    def test_url_in_string_value_is_preserved(self) -> None:
        source = '{"reservation_tips": "请提前预约：' + BOOKING_URL + '"}'

        cleaned = self.planner._sanitize_json_str(source)

        # 断言不依赖第 5 步的全角标点规范化，只关心 URL 是否被完整保留
        self.assertIn(BOOKING_URL, cleaned)
        self.assertTrue(json.loads(cleaned)["reservation_tips"].endswith(BOOKING_URL))

    def test_double_slash_inside_string_is_not_a_comment(self) -> None:
        source = '{"note": "a // b", "next": 1}'

        cleaned = self.planner._sanitize_json_str(source)

        self.assertEqual(json.loads(cleaned), {"note": "a // b", "next": 1})

    def test_block_comment_marker_inside_string_is_preserved(self) -> None:
        source = '{"note": "/* not a comment */", "next": 1}'

        cleaned = self.planner._sanitize_json_str(source)

        self.assertEqual(json.loads(cleaned), {"note": "/* not a comment */", "next": 1})

    def test_escaped_quote_before_url_does_not_confuse_scanner(self) -> None:
        source = '{"note": "他说\\"看这里\\" ' + BOOKING_URL + '"}'

        cleaned = self.planner._sanitize_json_str(source)

        self.assertEqual(json.loads(cleaned)["note"], f'他说"看这里" {BOOKING_URL}')

    def test_trailing_slash_only_url_is_preserved(self) -> None:
        source = '{"url": "https://example.com/a//b"}'

        cleaned = self.planner._sanitize_json_str(source)

        self.assertEqual(json.loads(cleaned)["url"], "https://example.com/a//b")


class SanitizeStillRemovesRealCommentsTests(unittest.TestCase):
    """不该因修复 URL 而丢掉原本的注释清理能力。"""

    def setUp(self) -> None:
        self.planner = _bare_planner()

    def test_line_comment_outside_string_is_removed(self) -> None:
        source = '{\n  "a": 1, // 这是注释\n  "b": 2\n}'

        cleaned = self.planner._sanitize_json_str(source)

        self.assertEqual(json.loads(cleaned), {"a": 1, "b": 2})

    def test_block_comment_outside_string_is_removed(self) -> None:
        source = '{"a": 1 /* 这是注释 */, "b": 2}'

        cleaned = self.planner._sanitize_json_str(source)

        self.assertEqual(json.loads(cleaned), {"a": 1, "b": 2})

    def test_comment_after_url_string_is_still_removed(self) -> None:
        source = '{\n  "reservation_tips": "' + BOOKING_URL + '", // 预约链接\n  "b": 2\n}'

        cleaned = self.planner._sanitize_json_str(source)

        parsed = json.loads(cleaned)
        self.assertEqual(parsed["reservation_tips"], BOOKING_URL)
        self.assertEqual(parsed["b"], 2)


class ParseResponseWithUrlTests(unittest.TestCase):
    """端到端：带预约链接的响应必须能被本地修复链解析，不需要 LLM 兜底。"""

    def test_plan_with_booking_url_parses_without_llm_fallback(self) -> None:
        planner = _bare_planner()

        def _forbidden_llm_repair(broken_json: str) -> str:
            raise RuntimeError("本地修复链已失败，触发了 LLM 修复兜底（离线测试禁用）")

        planner._llm_repair_json = _forbidden_llm_repair  # type: ignore[method-assign]

        plan = planner._parse_response(_plan_with_url("北京"), _trip_request("北京"))

        attraction = plan.days[0].attractions[0]
        self.assertIn(BOOKING_URL, attraction.reservation_tips)
        self.assertTrue(attraction.reservation_required)

    def test_compact_single_line_response_keeps_all_days(self) -> None:
        """紧凑输出曾整行被删除，导致内容静默丢失。"""
        planner = _bare_planner()
        planner._llm_repair_json = lambda broken: (_ for _ in ()).throw(
            RuntimeError("本地修复链已失败，触发了 LLM 修复兜底（离线测试禁用）")
        )
        compact = json.dumps(json.loads(_plan_with_url("上海")), ensure_ascii=False)

        plan = planner._parse_response(compact, _trip_request("上海"))

        self.assertEqual(len(plan.days), 1)
        self.assertIn(BOOKING_URL, plan.days[0].attractions[0].reservation_tips)


if __name__ == "__main__":
    unittest.main()
