"""回归测试：行程规划的 Agent 对话历史必须按请求隔离。

背景
----
``get_trip_planner_agent()`` 返回进程级单例 ``MultiAgentTripPlanner``。
hello_agents 的 ``SimpleAgent.run()`` 会把本次输入与回复追加进
``Agent._history``，并在下一次 ``run()`` 时把整段历史重新注入 messages。
因此只要规划流程复用同一个 Agent 实例，第 N 次请求就会带上前 N-1 次请求
的城市、偏好和已经生成的行程 JSON：

* 跨请求上下文污染——多用户部署时等同于串号；
* prompt 随请求数线性膨胀，最终触发模型上下文超限，之后所有规划请求都会
  失败，只能重启进程才能恢复。

本文件中的用例完全离线：不访问网络、不需要任何 API Key，也不接触
小红书 / 高德 / Google / LLM 真实服务。
"""

import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# hello_agents 会向 stdout 打印 emoji（✅ 等）。Windows 默认的 GBK 控制台会因此
# 抛 UnicodeEncodeError。服务进程在 app/api/main.py 里做了同样的兜底，测试进程
# 不经过那里，所以这里显式保持一致。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from hello_agents import SimpleAgent

from app.agents import trip_planner_agent as tpa
from app.models.schemas import TripRequest


# ============ 测试替身 ============


class _FakeTool:
    """满足 ToolRegistry.register_tool 所需最小接口的工具替身。"""

    def __init__(self, name: str = "fake_map_tool") -> None:
        self.name = name
        self.description = "离线测试用的地图工具替身"


class _RecordingLLM:
    """鸭子类型的 LLM：记录每次 invoke 收到的 messages。

    规划 Agent 与天气/酒店 Agent 共用同一个 LLM 实例，因此这里按 system
    prompt 区分调用来源：只有 system prompt 等于 ``PLANNER_AGENT_PROMPT``
    的调用才被记为“规划调用”，并返回 ``planner_response``。
    """

    provider = "recording"

    def __init__(self) -> None:
        self.planner_response: str = "{}"
        self.planner_calls: List[List[Dict[str, str]]] = []
        self.all_calls: List[List[Dict[str, str]]] = []

    def invoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        recorded = [dict(message) for message in messages]
        self.all_calls.append(recorded)
        system_prompt = recorded[0].get("content", "") if recorded else ""
        if system_prompt == tpa.PLANNER_AGENT_PROMPT:
            self.planner_calls.append(recorded)
            return self.planner_response
        return "离线占位：天气 / 酒店信息"


# ============ 用例辅助 ============


def _plan_payload(city: str) -> str:
    """构造一份最小但合法的 TripPlan JSON，并在内容里带上城市标记。"""
    return json.dumps(
        {
            "city": city,
            "cities": [city],
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "overall_suggestions": f"{city}行程建议",
            "days": [
                {
                    "date": "2026-05-01",
                    "day_index": 0,
                    "city": city,
                    "description": f"{city}第一天行程",
                    "transportation": "公共交通",
                    "accommodation": "经济型酒店",
                    "attractions": [],
                    "meals": [],
                },
                {
                    "date": "2026-05-02",
                    "day_index": 1,
                    "city": city,
                    "description": f"{city}第二天行程",
                    "transportation": "公共交通",
                    "accommodation": "经济型酒店",
                    "attractions": [],
                    "meals": [],
                },
            ],
            "weather_info": [],
            "budget": {
                "total_attractions": 0,
                "total_hotels": 0,
                "total_meals": 0,
                "total_transportation": 0,
                "total_inter_city_transport": 0,
                "total": 0,
            },
        },
        ensure_ascii=False,
    )


def _trip_request(city: str) -> TripRequest:
    return TripRequest(
        city=city,
        start_date="2026-05-01",
        end_date="2026-05-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
        language="zh",
    )


def _build_planner(llm: _RecordingLLM) -> "tpa.MultiAgentTripPlanner":
    """绕过依赖外部服务的 ``__init__``，直接装出一个可用的规划器。

    同时装配好实例属性（未修复版本的复用路径）与提示词属性（修复版本的
    每次新建路径），使同一份用例在修复前后都能运行。
    """
    planner = object.__new__(tpa.MultiAgentTripPlanner)
    planner.llm = llm
    planner.map_provider = "amap"
    planner._active_tool = _FakeTool()
    planner._weather_prompt = "天气查询系统提示词"
    planner._hotel_prompt = "酒店推荐系统提示词"

    weather_agent = SimpleAgent(name="天气查询专家", llm=llm, system_prompt=planner._weather_prompt)
    weather_agent.add_tool(planner._active_tool)
    hotel_agent = SimpleAgent(name="酒店推荐专家", llm=llm, system_prompt=planner._hotel_prompt)
    hotel_agent.add_tool(planner._active_tool)
    planner_agent = SimpleAgent(name="行程规划专家", llm=llm, system_prompt=tpa.PLANNER_AGENT_PROMPT)

    planner.weather_agent = weather_agent
    planner.hotel_agent = hotel_agent
    planner.planner_agent = planner_agent
    return planner


class _StubXhsModule:
    """上下文管理器：用假的 search_xhs_attractions 替换小红书服务模块。"""

    def __init__(self) -> None:
        self._original = sys.modules.get("app.services.xhs_service")

    def __enter__(self) -> None:
        module = types.ModuleType("app.services.xhs_service")

        def search_xhs_attractions(city: str, keywords: str, lang: str = "zh") -> str:
            return f"{city}景点候选数据（离线替身）"

        module.search_xhs_attractions = search_xhs_attractions  # type: ignore[attr-defined]
        sys.modules["app.services.xhs_service"] = module

    def __exit__(self, *exc_info: Any) -> None:
        if self._original is None:
            sys.modules.pop("app.services.xhs_service", None)
        else:
            sys.modules["app.services.xhs_service"] = self._original


# ============ 用例 ============


class SimpleAgentHistoryTests(unittest.TestCase):
    """固化 hello_agents 的行为，说明为什么 Agent 实例不能跨请求复用。"""

    def test_simple_agent_replays_previous_turns_into_next_call(self) -> None:
        llm = _RecordingLLM()
        agent = SimpleAgent(name="探针", llm=llm, system_prompt="探针系统提示词")

        agent.run("第一次提问")
        agent.run("第二次提问")

        second_call_text = json.dumps(llm.all_calls[-1], ensure_ascii=False)
        self.assertIn("第一次提问", second_call_text)
        self.assertEqual(len(agent.get_history()), 4)

    def test_clear_history_stops_the_replay(self) -> None:
        llm = _RecordingLLM()
        agent = SimpleAgent(name="探针", llm=llm, system_prompt="探针系统提示词")

        agent.run("第一次提问")
        agent.clear_history()
        agent.run("第二次提问")

        second_call_text = json.dumps(llm.all_calls[-1], ensure_ascii=False)
        self.assertNotIn("第一次提问", second_call_text)


class TripPlanningAgentIsolationTests(unittest.TestCase):
    """核心回归：连续两次规划之间不得有任何对话历史泄漏。"""

    def test_second_plan_does_not_replay_first_plan(self) -> None:
        llm = _RecordingLLM()
        planner = _build_planner(llm)

        with _StubXhsModule():
            llm.planner_response = _plan_payload("北京")
            asyncio.run(planner.plan_trip(_trip_request("北京")))
            boundary = len(llm.all_calls)

            llm.planner_response = _plan_payload("上海")
            asyncio.run(planner.plan_trip(_trip_request("上海")))

        self.assertEqual(len(llm.planner_calls), 2, "两次请求应各产生一次规划调用")

        # 断言覆盖第二次请求的**全部**模型调用（天气 / 酒店 / 规划三个 Agent），
        # 而不只是规划调用。否则「只把规划 Agent 换成新实例、天气与酒店仍共享」
        # 这种部分修复会带着泄漏溜过去。
        second_request_calls = json.dumps(llm.all_calls[boundary:], ensure_ascii=False)
        self.assertNotIn(
            "北京",
            second_request_calls,
            "第二次请求的模型调用中仍能看到第一次请求的城市，说明对话历史跨请求泄漏了",
        )

    def test_later_plans_never_see_earlier_cities(self) -> None:
        """连续多次规划时，每次请求都不得看到此前任何一次请求的城市。

        原实现比较各次规划请求的字节长度是否相等，是个脆弱的代理指标——它之所
        以长期为真，只是因为所选城市名恰好都是 2 个字（实测把城市换成
        「呼和浩特」「新疆维吾尔自治区」时长度即不相等）。改为直接断言城市名
        不出现，既不依赖输入长度，也能覆盖三个 Agent。
        """
        llm = _RecordingLLM()
        planner = _build_planner(llm)

        cities = ("北京", "上海", "广州")
        with _StubXhsModule():
            for index, city in enumerate(cities):
                llm.planner_response = _plan_payload(city)
                boundary = len(llm.all_calls)
                asyncio.run(planner.plan_trip(_trip_request(city)))

                current_calls = json.dumps(llm.all_calls[boundary:], ensure_ascii=False)
                for earlier in cities[:index]:
                    self.assertNotIn(
                        earlier,
                        current_calls,
                        f"第 {index + 1} 次请求（{city}）的模型调用中出现了此前的城市 {earlier}",
                    )

    def test_singleton_agent_attributes_are_untouched_by_planning(self) -> None:
        """规划流程不应把对话历史写回单例上的 Agent 属性。"""
        llm = _RecordingLLM()
        planner = _build_planner(llm)

        with _StubXhsModule():
            llm.planner_response = _plan_payload("北京")
            asyncio.run(planner.plan_trip(_trip_request("北京")))

        self.assertEqual(planner.weather_agent.get_history(), [])
        self.assertEqual(planner.hotel_agent.get_history(), [])
        self.assertEqual(planner.planner_agent.get_history(), [])


if __name__ == "__main__":
    unittest.main()
