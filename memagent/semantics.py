"""Shared semantic slots for memory writing, retrieval, advice, and forgetting."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re

import jieba.posseg as pseg


@dataclass(frozen=True)
class SemanticFrame:
    topic: str = "生活"
    entities: tuple[str, ...] = ()
    facets: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    broad: bool = False
    scope: str = "profile"
    normalized_query: str = ""
    personal: bool | None = None
    excluded_entities: tuple[str, ...] = ()


TOPIC_TERMS = {
    "健身": ("跑步", "健身", "公里", "慢跑", "晨跑"),
    "运动": ("羽毛球", "游泳", "篮球", "足球", "骑行", "运动", "锻炼", "活动", "宅着", "不动"),
    "饮食": (
        "饮食", "吃", "喝", "口味", "咖啡", "奶茶", "茶", "麻辣烫", "火锅", "早餐",
        "午餐", "晚餐", "清淡", "辣", "甜", "蒜香", "孜然", "香菜",
    ),
    "作息": ("睡觉", "睡眠", "起床", "早睡", "熬夜", "午睡", "作息"),
    "课程": ("课程", "项目", "实验", "论文", "答辩", "考试", "报告", "作业"),
    "阅读": ("读书", "在读", "读《", "看《", "阅读", "小说", "纸质书", "看书", "书", "Kindle"),
    "旅行": ("旅行", "旅游", "行程", "酒店", "景点"),
    "理财": ("基金", "定投", "理财", "记账", "存款"),
    "健康": ("体检", "复诊", "牙医", "体重", "健康"),
    "社交": ("朋友", "室友", "同事", "聚会", "社交"),
}

ENTITY_ALIASES = {
    "咖啡": ("咖啡", "咖啡豆", "豆子", "手冲", "速溶", "浅烘", "深烘"),
    "奶茶": ("奶茶",),
    "茶": ("茶", "茶饮", "红茶", "绿茶", "乌龙茶"),
    "麻辣烫": ("麻辣烫",),
    "火锅": ("火锅",),
    "烧烤": ("烧烤",),
    "跑步": ("跑步", "慢跑", "晨跑", "夜跑"),
    "羽毛球": ("羽毛球",),
    "游泳": ("游泳",),
    "睡眠": ("睡觉", "睡眠", "早睡", "熬夜", "午睡"),
    "课程项目": ("课程项目", "期末项目", "课程大作业", "课程表助手", "开题", "选题"),
    "论文": ("论文",),
    "答辩": ("答辩",),
    "阅读": ("读书", "在读", "阅读", "小说", "书籍", "看书", "读", "书", "科幻"),
    "定投": ("定投", "基金"),
    "实习": ("实习",),
    "复诊": ("复诊", "牙医"),
    "汽车": ("汽车", "买车", "车身", "车辆"),
    "机械键盘": ("机械键盘", "键盘", "轴体", "青轴", "红轴"),
    "钢笔": ("钢笔", "笔尖", "墨水", "EF 尖", "EF尖"),
    "胶片": ("胶片", "拍街景", "ISO"),
    "望远镜": ("望远镜", "观星", "目镜"),
    "鱼缸": ("鱼缸", "换水"),
    "蝴蝶兰": ("蝴蝶兰", "兰花", "浇水"),
    "酵种": ("酵种", "酸面包", "喂养"),
    "组会": ("组会",),
    "音乐": ("音乐", "听歌", "巴赫", "变奏曲"),
    "日语单词": ("日语单词", "日语", "Anki", "单词"),
    "通勤": ("通勤", "自行车", "骑自行车", "骑车", "单车", "地铁", "公交"),
    "猫咪": ("猫咪", "猫", "豆包", "喂猫"),
    "空调": ("空调", "睡眠模式"),
    "大扫除": ("大扫除", "打扫卫生", "清洁"),
    "高铁座位": ("高铁", "动车", "火车", "靠窗座位"),
    "飞机座位": ("飞机", "航班", "靠过道座位"),
}

FACET_VALUES = {
    "口味": (
        "清淡", "重口", "少油", "多油", "油腻", "咸", "甜", "酸", "苦",
        "蒜香", "香辣", "孜然", "泡椒", "酸菜", "麻酱", "番茄", "菌汤",
        "香菜", "葱", "蒜",
    ),
    "辣度": ("不辣", "微辣", "中辣", "重辣", "特辣", "麻辣", "能吃辣", "不能吃辣"),
    "甜度": ("无糖", "少糖", "半糖", "全糖", "低糖"),
    "温度": ("热饮", "冷饮", "常温", "加冰", "少冰", "去冰"),
    "烘焙": ("浅烘", "中烘", "深烘"),
    "冲泡方式": ("手冲", "速溶", "冷萃", "意式"),
    "颜色": (
        "红色", "橙色", "黄色", "绿色", "蓝色", "紫色", "黑色", "白色",
        "灰色", "粉色", "棕色",
    ),
    "风格": ("简约", "复古", "现代", "正式", "休闲", "运动风", "极简"),
    "频率": ("每天", "每周", "每月", "经常", "偶尔", "隔天"),
    "时间": (
        "早上", "上午", "中午", "下午", "晚上", "凌晨", "睡前", "周末",
        "周一", "周二", "周三", "周四", "周五", "周六", "周日",
    ),
    "强度": ("轻度", "中等", "高强度", "低强度"),
    "距离": ("三公里", "五公里", "十公里"),
    "状态": ("开始", "进行中", "完成", "暂停", "放弃", "延期"),
    "设备": ("Kindle", "纸质书", "电子书", "手机", "平板"),
}

FACET_MARKERS = {
    "口味": ("口味", "味道", "风味"),
    "辣度": ("辣度", "辣不辣", "吃辣", "能不能吃辣", "多辣", "辣"),
    "甜度": ("甜度", "几分糖", "糖度"),
    "温度": ("温度", "水温", "多少度", "冰不冰", "热饮", "冷饮"),
    "烘焙": ("烘焙", "烘焙度"),
    "冲泡方式": ("冲泡", "怎么冲", "冲法"),
    "颜色": ("颜色", "什么色", "哪种色"),
    "风格": ("风格", "款式"),
    "频率": ("频率", "多久一次", "几次"),
    "时间": ("时间", "几点", "什么时候"),
    "强度": ("强度",),
    "距离": ("距离", "几公里"),
    "状态": ("状态", "进度", "做到哪", "还在"),
    "配置": ("配置", "设置", "设多少", "什么轴", "轴体", "轴", "模式"),
    "设备": ("设备", "用什么看", "工具"),
    "类型": ("类型", "哪类", "什么书", "什么音乐"),
    "比例": ("比例", "多少水", "换水量"),
    "地点": ("地点", "城市", "哪里", "哪儿", "在哪", "什么地方"),
    "数量": ("数量", "多少本", "几本", "多少个", "几个"),
    "金额": ("金额", "多少钱", "多少元", "几元", "预算", "费用", "花费", "价格"),
}

_GENERIC_ENTITY_WORDS = {
    "偏好", "习惯", "口味", "味道", "风味", "类型", "方式", "时间", "频率",
    "状态", "颜色", "风格", "东西", "事情", "内容", "记录", "记忆", "计划",
    "今天", "昨天", "明天", "最近", "目前", "现在", "什么", "哪些", "哪个",
    "本人", "用户", "一次", "长期", "饮食", "健身", "运动", "作息", "课程",
    "旅行", "理财", "健康", "社交", "生活", "项目", "合口味",
    "程度", "大概", "左右", "一般", "通常", "固定", "安排", "设备", "类型",
    "方向", "题目", "比例", "模式", "工具", "偏向", "选择", "时候", "周日",
    "周几", "晚上", "上午", "座位", "原来", "后来", "最初", "最终", "最后",
    "总结", "归纳", "概括", "回顾", "复盘", "关系", "原因", "城市", "地点",
    "数量", "结果", "情况", "进展", "目标", "本数", "变化", "演变", "历程",
    "过程", "历史", "前后", "轨迹", "实际", "个人", "整体", "综合",
}
_BROAD_MARKERS = ("总结", "整体", "所有", "全部", "有哪些偏好", "有什么习惯")
_CURRENT_MARKERS = (
    "现在", "目前", "当前", "如今", "最近", "已经", "最终", "最后定",
    "还在", "还是", "是不是还",
)
_HISTORY_MARKERS = ("原来", "以前", "最初", "一开始", "当时", "曾经", "过去")
_EVOLUTION_MARKERS = ("后来", "改成", "换成", "变化", "演变", "时间线", "分别")
_PERSONAL_MARKERS = (
    "我", "我的", "本人", "平时", "习惯", "偏好", "固定",
    "原来", "以前", "现在", "目前", "最终", "一开始",
)
_GENERAL_MARKERS = (
    "一般来说", "通常来说", "科普", "原理", "为什么", "什么是", "如何制作",
    "怎么学习", "教程", "定义",
)
_GENERAL_KNOWLEDGE_RE = re.compile(
    r"区别|差异|优缺点|好处|意味着|原理|作用|为什么|"
    r"有什么不同|有什么取舍|差多少|对普通.*来说|如果一个人|"
    r"省电|耗电|兼容性|具体指什么|怎么理解|表示什么|指哪|"
    r"会怎样影响|影响.*速度|什么原因造成|差别|成年人|常用表达|每周建议|"
    r"怎么(?:变化|调整|热身|取舍|养成|制作|判断|安排|区分|补救|预防|设计)|"
    r"如何(?:选择|比较|判断|制作)|要注意什么|需要注意什么|注意事项|"
    r"哪些(?:建议|方法|因素)|受哪些因素影响|训练方法|比较合理|应该先|"
    r"一般建议|一般是为了|可以怎么写|在邮件里|运动量算(?:大|小|合适)"
)
_VALID_SCOPES = {"current", "history", "evolution", "profile", "advice", "general"}
_FACET_COMPATIBILITY_GROUPS = (
    frozenset({"配置", "类型", "设备"}),
)
_LLM_GENERIC_ENTITIES = {
    "安排", "习惯", "偏好", "配置", "设置", "状态", "时间", "频率", "方式",
    "类型", "程度", "选择", "建议", "运动", "饮食", "生活", "项目",
    "实际", "个人", "整体", "综合",
}


def infer_topic(text: str, hint: str = "") -> str:
    if hint and hint != "生活":
        return hint
    ranked = []
    for index, (topic, terms) in enumerate(TOPIC_TERMS.items()):
        hits = [term for term in terms if term in text]
        if not hits:
            continue
        score = sum(max(len(term) - 1, 1) for term in set(hits))
        ranked.append((score, max(map(len, hits)), -index, topic))
    return max(ranked)[-1] if ranked else (hint or "生活")


def _facet_values(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    facets = []
    values = []
    for facet, markers in FACET_MARKERS.items():
        if any(marker in text for marker in markers):
            facets.append(facet)
    for facet, candidates in FACET_VALUES.items():
        hits = [(text.rfind(value), value) for value in candidates if value in text]
        if hits:
            if facet not in facets:
                facets.append(facet)
            values.extend(value for _position, value in sorted(hits))
    degree = re.search(r"(?:\d{1,3}|[一二三四五六七八九十百]{1,4})\s*度", text)
    if degree:
        if "温度" not in facets:
            facets.append("温度")
        values.append(degree.group(0).replace(" ", ""))
    quantity = re.search(
        r"(?:第?[一二三四五六七八九十百\d]+本|[一二三四五六七八九十百\d]+个)",
        text,
    )
    if quantity:
        if "数量" not in facets:
            facets.append("数量")
        values.append(quantity.group(0))
    frequency = re.search(
        r"(?:多久|多长时间).{0,8}(?:一次|一回)|"
        r"隔(?:多久|几天|几周|几个月|多长时间)|"
        r"(?:每|隔)\s*(?:\d+|[一二两三四五六七八九十半]+)?\s*"
        r"(?:天|周|星期|个月|月|年)(?:\s*(?:一次|一回))?",
        text,
    )
    if frequency:
        if "频率" not in facets:
            facets.append("频率")
        frequency_value = frequency.group(0).replace(" ", "")
        if not re.search(r"多久|多长时间|几天|几周|几个月", frequency_value):
            values.append(frequency_value)
    money = re.search(
        r"(?:人民币\s*)?\d+(?:\.\d+)?\s*(?:元|块|万元)|"
        r"(?:每天|每周|每月|每次)\s*\d+(?:\.\d+)?(?:\s*(?:元|块))?",
        text,
    )
    if money and "金额" not in facets:
        facets.append("金额")
    titles = [title.strip() for title in re.findall(r"《([^》]{1,30})》", text) if title.strip()]
    if titles:
        if "类型" not in facets:
            facets.append("类型")
        values.extend(titles)
    # Jieba occasionally tags ordinary nouns as place names (for example
    # "墨水"). Require an explicit spatial context before turning an `ns`
    # token into a location facet, otherwise profile/configuration queries can
    # be filtered out as if they were asking for a place.
    has_spatial_context = bool(re.search(
        r"去(?:了|过|往)?|到(?:了|过|达)?|住在|位于|来自|"
        r"地点|城市|哪里|哪儿|何处|在哪(?:里|儿)?",
        text,
    ))
    if has_spatial_context and any(pair.flag == "ns" for pair in pseg.cut(text)):
        if "地点" not in facets:
            facets.append("地点")
    return tuple(dict.fromkeys(facets)), tuple(dict.fromkeys(values))


def _entities(text: str, values: tuple[str, ...]) -> tuple[str, ...]:
    found = []
    matched_aliases = []
    for entity, aliases in ENTITY_ALIASES.items():
        hits = [alias for alias in aliases if alias in text]
        if entity == "茶" and hits == ["茶"] and "奶茶" in text:
            hits = []
        if hits:
            found.append(entity)
            matched_aliases.extend(hits)
    excluded = _GENERIC_ENTITY_WORDS | set(values)
    for pair in pseg.cut(text):
        word = pair.word.strip()
        if (
            len(word) < 2
            or word in excluded
            or any(word != alias and word in alias for alias in matched_aliases)
            or any(word in markers for markers in FACET_MARKERS.values())
            or any(word in candidates for candidates in FACET_VALUES.values())
        ):
            continue
        if pair.flag.startswith(("n", "vn")) and not any(
            word in aliases for aliases in ENTITY_ALIASES.values()
        ):
            found.append(word)
    found.extend(
        title.strip()
        for title in re.findall(r"《([^》]{1,30})》", text)
        if title.strip()
    )
    return tuple(dict.fromkeys(found))


def infer_scope(text: str) -> str:
    has_history = any(marker in text for marker in _HISTORY_MARKERS) or bool(
        re.search(r"(?:做|去|吃|喝|买|看|跑|放弃|开始|完成)过", text)
    ) or bool(re.search(r"什么时候|哪天|哪一年|哪一月|几月几日", text))
    has_current = any(marker in text for marker in _CURRENT_MARKERS) or (
        "最后" in text and "最后一次" not in text
    )
    has_evolution = any(marker in text for marker in _EVOLUTION_MARKERS)
    explicit_evolution = bool(re.search(
        r"经历(?:了|过)?.{0,10}(?:变化|演变)|"
        r"(?:怎么|如何|怎样)(?:变化|演变)|"
        r"(?:变化|演变)(?:过程|历程)|"
        r"前后.{0,6}(?:变化|调整)|"
        r"(?:安排|计划|状态|配置|金额|频率|时间).{0,6}(?:怎么|如何|怎样)(?:变化|调整)|"
        r"时间线",
        text,
    ))
    if (
        explicit_evolution
        or (has_history and (has_current or has_evolution))
        or ("从" in text and "到" in text)
    ):
        return "evolution"
    if has_history:
        return "history"
    if has_current:
        return "current"
    if re.search(r"应该|该怎么|怎么办|推荐|建议|怎么选|选什么|哪种更合适", text):
        return "advice"
    return "profile"


def _locally_personal(text: str, frame: SemanticFrame) -> bool | None:
    has_personal_subject = bool(re.search(r"我|我的|本人", text))
    if not has_personal_subject and any(marker in text for marker in _GENERAL_MARKERS):
        return False
    if (
        not has_personal_subject
        and frame.scope == "evolution"
        and frame.entities
        and not re.search(r"市场|行业|全国|全球|社会|历史上|一般|通常", text)
    ):
        # “陶艺课的安排怎么变化”既可能指个人记录，也可能是一般讨论。
        # 保留为歧义，让查询路由结合实际记忆证据决定，而不是直接判成通用知识。
        return None
    if not has_personal_subject and _GENERAL_KNOWLEDGE_RE.search(text):
        return False
    if re.search(r"(我|我的).{0,8}(一般|通常)", text):
        return True
    if any(marker in text for marker in _PERSONAL_MARKERS):
        return True
    if frame.scope in ("current", "history", "evolution"):
        return True
    return None


def analyze_semantics(
    text: str,
    *,
    topic_hint: str = "",
    preference_value: str = "",
    is_query: bool = False,
) -> SemanticFrame:
    value = " ".join((text or "").split())
    combined = f"{value} {preference_value}".strip()
    facets, values = _facet_values(combined)
    entities = _entities(combined, values)
    excluded_entities: tuple[str, ...] = ()
    if is_query:
        exclusion = re.search(r"除了(.+?)(?:，|,|还有|我还|还会)", value)
        if exclusion:
            excluded_entities = _entities(exclusion.group(1), values)
            entities = tuple(
                entity for entity in entities
                if not any(_entity_related(entity, excluded) for excluded in excluded_entities)
            )
        contrast = re.search(r"(.+?)是不是(?:跟|和|像)", value)
        if contrast:
            primary_entities = _entities(contrast.group(1), values)
            if primary_entities:
                entities = primary_entities
    topic = infer_topic(combined, topic_hint)
    broad = is_query and (
        any(marker in value for marker in _BROAD_MARKERS)
        or bool(re.search(r"(?:有什么|有哪些).{0,8}(?:偏好|习惯)", value))
    )
    scope = infer_scope(value) if is_query else "profile"
    frame = SemanticFrame(
        topic=topic,
        entities=entities,
        facets=facets,
        values=values,
        broad=broad,
        scope=scope,
        normalized_query=value,
        excluded_entities=excluded_entities,
    )
    if is_query:
        return SemanticFrame(
            topic=frame.topic,
            entities=frame.entities,
            facets=frame.facets,
            values=frame.values,
            broad=frame.broad,
            scope=frame.scope,
            normalized_query=frame.normalized_query,
            personal=_locally_personal(value, frame),
            excluded_entities=frame.excluded_entities,
        )
    return frame


_QUERY_ANALYSIS_SYSTEM = (
    "你是个人记忆系统的查询分析器。只做语义归一化，不回答问题，也不选择记忆。"
    "输出 JSON："
    '{"personal":true/false,"scope":"current|history|evolution|profile|advice|general",'
    '"topic":"简短主题","entities":["具体对象"],"facets":["被询问的属性"],'
    '"broad":true/false,"normalized_query":"不改变原意的简短检索表达"}。'
    "personal=true 表示答案依赖这个用户自己的既定事实、习惯、配置、安排或经历；"
    "省略“我的”但在问既定配置也属于 personal，例如“鱼缸多久换一次水”“组会现在周几”。"
    "若是在问普通知识或通用做法则 personal=false，例如“鱼缸一般多久换水”。"
    "scope=current 只问现状；history 只问旧状态；evolution 同时问前后变化；"
    "profile 问稳定习惯或配置；advice 是需要结合个人情况的建议；general 是通用问题。"
    "entities 只写核心对象，不写‘时间、频率、方式、偏好、安排’等属性词；"
    "否定句中被用户排除的错误选项不是查询对象，例如“不是刷手机，那看什么”不能把手机作为对象；"
    "facets 写时间、频率、颜色、口味、配置、方法、状态等属性。只输出 JSON。"
)


def _merge_frame(local: SemanticFrame, payload: dict, text: str) -> SemanticFrame:
    def clean_list(name: str, fallback: tuple[str, ...], *, entities=False) -> tuple[str, ...]:
        values = payload.get(name)
        if not isinstance(values, list):
            return fallback
        cleaned = [
            str(value).strip() for value in values
            if isinstance(value, str) and 1 < len(value.strip()) <= 20
        ]
        if entities:
            cleaned = [
                value for value in cleaned
                if value not in _LLM_GENERIC_ENTITIES
                and not any(value.endswith(suffix) for suffix in ("安排", "习惯", "偏好", "情况"))
            ]
            if any(value in ENTITY_ALIASES for value in fallback):
                return fallback
        elif fallback:
            return fallback
        return tuple(dict.fromkeys((*fallback, *cleaned)))

    scope = str(payload.get("scope", "")).strip()
    if scope not in _VALID_SCOPES:
        scope = local.scope
    if local.scope != "profile":
        scope = local.scope
    topic = str(payload.get("topic", "")).strip()
    if not topic or len(topic) > 12:
        topic = local.topic
    if local.topic != "生活":
        topic = local.topic
    personal = payload.get("personal")
    if not isinstance(personal, bool):
        legacy_intent = str(payload.get("intent", "")).strip().lower()
        if legacy_intent in ("memory", "general"):
            personal = legacy_intent == "memory"
        else:
            personal = local.personal
    if local.personal is not None:
        personal = local.personal
    normalized = str(payload.get("normalized_query", "")).strip()
    if not normalized or len(normalized) > max(120, len(text) * 3):
        normalized = local.normalized_query or text
    elif normalized != text:
        normalized = f"{text} {normalized}"
    return SemanticFrame(
        topic=topic,
        entities=clean_list("entities", local.entities, entities=True),
        facets=clean_list("facets", local.facets),
        values=local.values,
        broad=local.broad,
        scope=scope,
        normalized_query=normalized,
        personal=personal,
        excluded_entities=local.excluded_entities,
    )


def _needs_online_completion(frame: SemanticFrame, text: str) -> bool:
    if frame.personal is None or not frame.entities:
        return True
    if frame.scope in ("history", "current", "evolution") and len(frame.entities) > 2:
        return True
    return bool(re.search(r"那|这个|那个|它|怎么定|怎么样|什么样", text))


@lru_cache(maxsize=512)
def analyze_query(text: str, use_llm: bool = True) -> SemanticFrame:
    """Build one cached query frame shared by routing and retrieval."""
    local = analyze_semantics(text, is_query=True)
    if not use_llm or not _needs_online_completion(local, text):
        return local
    try:
        from . import llm
        if not llm.is_enabled():
            return local
        payload = llm.chat_json([
            {"role": "system", "content": _QUERY_ANALYSIS_SYSTEM},
            {"role": "user", "content": text},
        ])
        if isinstance(payload, dict):
            return _merge_frame(local, payload, text)
    except Exception:
        pass
    return local


@lru_cache(maxsize=256)
def ground_query_to_catalog(
    text: str,
    catalog: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...],
) -> SemanticFrame | None:
    """Resolve an omitted or unusual expression against known schema labels.

    The model sees only topics/entities/facets, not memory values or IDs. It
    therefore supplies a constrained rewrite while local retrieval still
    decides which concrete records are evidence.
    """
    try:
        from . import llm
        if not llm.is_enabled() or not catalog:
            return None
        payload = llm.chat_json([
            {
                "role": "system",
                "content": (
                    "把含糊问题映射到给定的记忆模式词表。每项包含主题、对象、属性和一条"
                    "用于消歧的原始描述。被否定或排除的对象不能作为目标。"
                    "只有能可靠对应时 matched=true；否则 false。只输出 JSON："
                    '{"matched":true/false,"topic":"词表中的主题",'
                    '"entities":["词表中的对象"],"facets":["词表中的属性"],'
                    '"scope":"current|history|evolution|profile|advice"}。'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{text}\n"
                    f"可选模式：{json.dumps(catalog, ensure_ascii=False)}"
                ),
            },
        ])
    except Exception:
        return None
    if payload.get("matched") is not True:
        return None
    allowed_topics = {item[0] for item in catalog}
    allowed_entities = {value for item in catalog for value in item[1]}
    allowed_facets = {value for item in catalog for value in item[2]}
    topic = str(payload.get("topic", "")).strip()
    entities = tuple(
        value for value in payload.get("entities", [])
        if isinstance(value, str) and value in allowed_entities
    )
    facets = tuple(
        value for value in payload.get("facets", [])
        if isinstance(value, str) and value in allowed_facets
    )
    scope = str(payload.get("scope", "")).strip()
    if topic not in allowed_topics or scope not in _VALID_SCOPES - {"general"}:
        return None
    if not entities and not facets:
        return None
    return SemanticFrame(
        topic=topic,
        entities=tuple(dict.fromkeys(entities)),
        facets=tuple(dict.fromkeys(facets)),
        scope=scope,
        normalized_query=text,
        personal=True,
    )


def _entity_related(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) >= 2 and (left in right or right in left):
        return True
    left_pairs = {left[i:i + 2] for i in range(max(0, len(left) - 1))}
    right_pairs = {right[i:i + 2] for i in range(max(0, len(right) - 1))}
    if not left_pairs or not right_pairs:
        return False
    return len(left_pairs & right_pairs) / len(left_pairs | right_pairs) >= 0.5


def entity_overlap_count(query_entities, memory_entities) -> int:
    return sum(
        1 for query_entity in query_entities
        if any(_entity_related(query_entity, memory_entity) for memory_entity in memory_entities)
    )


def frame_for_memory(memory, parent_text: str = "") -> SemanticFrame:
    stored_entities = tuple(getattr(memory, "entities", ()) or ())
    stored_facets = tuple(getattr(memory, "facets", ()) or ())
    stored_values = tuple(getattr(memory, "semantic_values", ()) or ())
    if stored_entities or stored_facets or stored_values:
        return SemanticFrame(
            topic=memory.topic,
            entities=stored_entities,
            facets=stored_facets,
            values=stored_values,
            normalized_query=memory.detail,
        )
    return analyze_semantics(
        f"{memory.detail} {parent_text}",
        topic_hint=memory.topic,
        preference_value=getattr(memory, "preference_key", ""),
    )


def semantic_match_score(
    query: SemanticFrame,
    memory: SemanticFrame,
    *,
    strict_facet: bool = True,
    require_value: bool = False,
) -> int:
    query_entities = set(query.entities)
    memory_entities = set(memory.entities)
    if entity_overlap_count(query.excluded_entities, memory_entities):
        return 0
    entity_overlap = entity_overlap_count(query_entities, memory_entities)
    related_topics = (
        {query.topic, memory.topic} <= {"健身", "运动"}
    )
    if (
        query.topic
        and memory.topic
        and query.topic != "生活"
        and memory.topic != "生活"
        and query.topic != memory.topic
        and not related_topics
        and not entity_overlap
    ):
        return 0

    score = 10
    query_facets = set(query.facets)
    memory_facets = set(memory.facets)
    query_values = set(query.values)
    memory_values = set(memory.values)
    facet_overlap = query_facets & memory_facets
    for group in _FACET_COMPATIBILITY_GROUPS:
        if query_facets & group and memory_facets & group:
            facet_overlap.add(next(iter(query_facets & group)))
    value_overlap = query_values & memory_values
    if "周末" in query_values:
        value_overlap |= memory_values & {"周六", "周日", "周末"}

    # A specific-domain query must not fall through to an unrelated generic
    # "生活" memory merely because both omit an entity.
    if (
        query.topic != "生活"
        and memory.topic == "生活"
        and not entity_overlap
        and not facet_overlap
    ):
        return 0

    if query_entities:
        if memory_entities:
            if not entity_overlap:
                return 0
            score += 80 + 30 * entity_overlap
        elif query_facets and query_facets & memory_facets:
            # A global attribute can inform a concrete object:
            # "I like blue" may inform a car colour choice.
            score += 35
        else:
            return 0
    elif memory_entities:
        if query_facets and not query.broad and not query.excluded_entities:
            # The user may omit the concrete object while still identifying it
            # through a topic, attribute and value: "周日有什么运动" -> 游泳.
            if not facet_overlap:
                return 0
            # Without a concrete value this is a broad attribute question such
            # as "我喜欢什么口味", where object-specific coffee preferences
            # must not leak into a global answer.
            evidence_facets = {"地点", "数量", "类型", "设备", "金额"}
            if not query_values and query_facets & evidence_facets:
                score += 35
            elif not query_values or not memory_values or not value_overlap:
                return 0
            else:
                score += 45 if query.topic == memory.topic else 20
        else:
            score += 15 if query.topic == memory.topic else 5

    if "地点" in query_facets and memory.topic == "旅行" and memory.entities:
        facet_overlap.add("地点")

    if query_facets:
        if facet_overlap:
            score += 40 + 5 * len(facet_overlap)
        elif query_facets & {
            "地点", "数量", "设备", "配置", "颜色", "风格", "比例",
            "甜度", "温度", "类型", "金额",
        }:
            return 0
        elif memory_facets and strict_facet and not query_facets <= {"时间", "状态"}:
            return 0

    if query_values:
        if value_overlap:
            score += 30 + 5 * len(value_overlap)
        elif (
            "时间" in query_facets
            and "时间" in memory_facets
            and memory_values
            and not query.broad
            and not re.search(r"是不是|不是|难道|吗|吧", query.normalized_query)
        ):
            return 0
        elif require_value:
            return 0

    if query.scope == "current":
        score += 12
    elif query.scope == "history":
        score += 8
    elif query.scope == "evolution":
        score += 6

    return score


def semantic_label(frame: SemanticFrame) -> str:
    """Return a compact, domain-neutral label for user-facing answers."""
    if frame.entities and frame.facets:
        return f"{frame.entities[0]}的{frame.facets[0]}"
    if frame.entities:
        return frame.entities[0]
    if frame.facets:
        return frame.facets[0]
    if frame.topic != "生活":
        return frame.topic
    return "相关"


def semantic_signature(frame: SemanticFrame, fallback: str = "") -> tuple:
    return (
        frame.topic,
        frame.entities,
        frame.facets,
        frame.values or (fallback,),
    )


def enrich_memory_semantics(store) -> bool:
    changed = False
    for memory in store.memories:
        parent = store.memory_by_id(memory.parent_id) if memory.parent_id else None
        frame = analyze_semantics(
            f"{memory.detail} {parent.detail if parent else ''}",
            topic_hint=memory.topic,
            preference_value=memory.preference_key,
        )
        if list(getattr(memory, "entities", ()) or ()) != list(frame.entities):
            memory.entities = list(frame.entities)
            changed = True
        if list(getattr(memory, "facets", ()) or ()) != list(frame.facets):
            memory.facets = list(frame.facets)
            changed = True
        if list(getattr(memory, "semantic_values", ()) or ()) != list(frame.values):
            memory.semantic_values = list(frame.values)
            changed = True
    if changed:
        store.save()
    return changed
