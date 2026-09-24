"""高德官方 Web 服务（REST）直连客户端。

为什么不复用 `amap_service.AmapService`（它走 MCP）：
`uvx amap-mcp-server` 的 `maps_text_search` 只返回
`id / name / address / typecode` 四个字段，把 `photos`（官方图库）、
`location`（坐标）、`business`（评分、营业时间、电话）全部裁掉了。
要拿到这些字段必须直接调高德 Web 服务 API，而不是经过 MCP 封装。

本模块只负责「取数」，不做业务编排；调用方负责在异步路由里用
`asyncio.to_thread` 派发（`httpx.get` 是阻塞调用）。
"""

import threading
import time
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings

_PLACE_TEXT_URL = "https://restapi.amap.com/v5/place/text"
_TIMEOUT_SECONDS = 10
# v5 接口默认只返回基础字段，需要显式声明才能拿回图片与营业信息
_SHOW_FIELDS = "photos,business"

# 高德个人 Key 默认限 3 次/秒，超了会返回 CUQPS_HAS_EXCEEDED_THE_LIMIT。
# 前端加载图片时是并发 4 的，不主动限流必然触发。这里与
# `xhs_service._wait_for_amap_geocode_slot` 用同一套「滑动窗口」做法
# （那边限的是地理编码，本模块限的是 POI 搜索，两者共用同一个 Key 的配额）。
_AMAP_RATE_LIMIT = 3
_AMAP_RATE_WINDOW = 1.0
_QPS_RETRY_DELAY_SECONDS = 1.2

_rate_lock = threading.Lock()
_request_times: deque = deque()


def _wait_for_slot() -> None:
    """限制高德 POI 搜索的启动速率，避免超过官方 3 次/秒上限。"""
    while True:
        wait_seconds = 0.0
        now = time.monotonic()
        with _rate_lock:
            while _request_times and now - _request_times[0] >= _AMAP_RATE_WINDOW:
                _request_times.popleft()

            if len(_request_times) < _AMAP_RATE_LIMIT:
                _request_times.append(now)
                return

            wait_seconds = _AMAP_RATE_WINDOW - (now - _request_times[0])

        if wait_seconds > 0:
            time.sleep(wait_seconds)


# 名称+城市 → 图片直链的内存缓存。
#
# 两条 TTL 刻意不对称：高德官方图库直链是**稳定**的（路径里没有时效签名、
# 也不校验 Referer），所以命中可以放心长期复用；而未取到图只缓存很短时间，
# 避免把一次瞬时失败放大成「接下来很久都没有图」——那正是小红书旧实现
# 用 600 秒负向缓存造成过的故障。
_HIT_TTL_SECONDS = 7 * 24 * 3600
_MISS_TTL_SECONDS = 60
_CACHE_MAXSIZE = 1024

_photo_cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
_photo_cache_lock = threading.Lock()


def _cache_key(name: str, city: str) -> str:
    return f"{city.strip()}\x00{name.strip()}"


def _cache_get(key: str) -> Optional[str]:
    with _photo_cache_lock:
        entry = _photo_cache.get(key)
        if entry is None:
            return None
        url, expires_at = entry
        if time.monotonic() >= expires_at:
            del _photo_cache[key]
            return None
        _photo_cache.move_to_end(key)
        return url


def _cache_put(key: str, url: str) -> None:
    ttl = _HIT_TTL_SECONDS if url else _MISS_TTL_SECONDS
    with _photo_cache_lock:
        _photo_cache[key] = (url, time.monotonic() + ttl)
        _photo_cache.move_to_end(key)
        while len(_photo_cache) > _CACHE_MAXSIZE:
            _photo_cache.popitem(last=False)


def _photo_urls(poi: Dict[str, Any]) -> List[str]:
    """取出一条 POI 的图片直链列表（过滤掉缺失项）。"""
    photos = poi.get("photos") or []
    if not isinstance(photos, list):
        return []
    return [
        str(p["url"])
        for p in photos
        if isinstance(p, dict) and p.get("url")
    ]


def _pick_photo(pois: List[Dict[str, Any]], name: str) -> str:
    """在候选 POI 中挑一张图。

    优先「名称完全一致」，其次「名称包含关键词」，最后退化到第一个有图的条目。
    高德文本搜索会返回一批相关 POI，直接取第一个容易拿到同名酒店、分店或
    公交站；而终点兜底又很必要——行程里的名字常与高德官方名略有出入
    （「飞天山」在库里是「飞天山景区」，「郴州博物馆」是「郴州市博物馆」）。
    """
    exact = ""
    partial = ""
    fallback = ""
    for poi in pois:
        if not isinstance(poi, dict):
            continue
        urls = _photo_urls(poi)
        if not urls:
            continue
        poi_name = str(poi.get("name") or "").strip()
        if not fallback:
            fallback = urls[0]
        if name and poi_name == name and not exact:
            exact = urls[0]
        elif name and name in poi_name and not partial:
            partial = urls[0]
    return exact or partial or fallback


def search_place(keywords: str, city: str = "", page_size: int = 5) -> Optional[List[Dict[str, Any]]]:
    """调用高德 POI 文本搜索，返回原始 POI 列表（含 photos / business / location）。

    返回值刻意区分两种「空」：
    - `[]`   —— 接口正常，但确实搜不到
    - `None` —— 接口调用失败（网络异常、Key 无效、配额用尽等）
    调用方据此决定要不要把「空结果」写进缓存：失败绝不能当成「此处无图」，
    否则一次限流就会变成一段时间的持续缺图。
    """
    key = get_settings().vite_amap_web_key
    if not key or not keywords:
        return None

    params: Dict[str, Any] = {
        "key": key,
        "keywords": keywords,
        "page_size": page_size,
        "show_fields": _SHOW_FIELDS,
    }
    if city:
        params["region"] = city

    for attempt in (1, 2):
        _wait_for_slot()
        try:
            resp = httpx.get(
                _PLACE_TEXT_URL,
                params=params,
                timeout=_TIMEOUT_SECONDS,
                trust_env=False,
            )
            data = resp.json()
        except Exception as e:  # noqa: BLE001 — 网络异常统一降级，不打断上层
            print(f"⚠️  [AMapREST] POI 搜索失败 ({keywords}@{city}): {e}")
            return None

        if str(data.get("status")) == "1":
            pois = data.get("pois")
            return pois if isinstance(pois, list) else []

        info = str(data.get("info") or "")
        # 被限流是可以自愈的，等一个窗口再试一次，别急着上报失败
        if attempt == 1 and "QPS" in info.upper():
            print(f"⚠️  [AMapREST] 触发 QPS 限流，{_QPS_RETRY_DELAY_SECONDS}s 后重试 ({keywords})")
            time.sleep(_QPS_RETRY_DELAY_SECONDS)
            continue

        print(f"⚠️  [AMapREST] POI 搜索返回异常 ({keywords}@{city}): {info}")
        return None

    return None


def get_poi_photo_url(name: str, city: str = "") -> str:
    """按景点名取一张高德官方图库直链，取不到返回空串（带缓存）。

    该直链可直接被浏览器引用：实测带不带 Referer 都返回 200，
    因此页面展示不需要后端代理（导出场景除外，见 xhs_service 的白名单注释）。
    """
    name = (name or "").strip()
    city = (city or "").strip()
    if not name:
        return ""

    cache_key = _cache_key(name, city)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = ""
    api_failed = False
    # 先限定城市搜；不带城市再兜一次，覆盖「城市名与高德行政区划不一致」的情况
    for region in ([city] if city else []) + [""]:
        pois = search_place(name, region)
        if pois is None:
            api_failed = True
            continue
        url = _pick_photo(pois, name)
        if url:
            break

    if url:
        _cache_put(cache_key, url)
    elif api_failed:
        # 接口失败不代表没有图：不写缓存，下次请求立刻重试
        print(f"⚠️  [AMapREST] 「{name}」查询失败，不缓存以便重试 (city={city or '-'})")
    else:
        _cache_put(cache_key, "")
        print(f"⚠️  [AMapREST] 高德图库确认无「{name}」的图片 (city={city or '-'})")
    return url
