# 记忆层：抽取 -> 冲突检测 -> 提炼结论 -> 决定去留（遗忘是主动设计）
import math
import re
from datetime import date as _date

from . import config, llm
from .store import MemoryStore, MemoryItem, Conflict, Conclusion

# 主题词表：命中即归类（本地规则路径的“语义理解”下位替代）
TOPIC_RULES = [
    (("跑步", "健身", "三公里", "五公里"), "健身"),
    (("羽毛球", "运动", "球馆"), "运动"),
    (("吃", "喝", "早餐", "午餐", "晚餐", "饭", "餐", "饮食", "咖啡", "手冲",
      "速溶", "豆子", "浅烘", "深烘", "奶茶", "麻辣烫", "微辣", "中辣", "辣",
      "清淡", "重口", "蒜香", "香辣", "孜然", "泡椒", "酸菜", "麻酱",
      "番茄", "菌汤", "香菜", "少糖", "无糖", "加冰", "少冰", "去冰"), "饮食"),
    (("熬夜", "早睡", "作息", "刷手机"), "作息"),
    (("课程", "项目", "选题", "实验", "期末", "选课"), "课程"),
    (("室友", "大扫除", "打扫"), "社交"),
    (("读书", "在读", "读《", "看《", "小说", " Kindle", "Kindle", "三体", "书"), "阅读"),
    (("旅行", "旅游", "古城", "泉州", "西安", "行程", "古镇"), "旅行"),
    (("基金", "定投", "理财", "记账", "存款"), "理财"),
    (("体检", "体重", "戒", "健康"), "健康"),
]

# 泛化问法 -> 一组主题：问"兴趣爱好"没有单一主题，把爱好类主题的时间线都拉出来
BROAD_TOPICS = {
    ("爱好", "兴趣"): ("运动", "健身", "饮食", "作息"),
    ("运动", "锻炼"): ("运动", "健身"),
    ("生活", "日常"): ("作息", "社交", "生活"),
}

ACTION_RULES = [
    (("放弃", "停止", "摆烂", "中断", "不再", "再也没", "停用"), "放弃"),
    (("重新开始", "开始", "下决心", "下定决心"), "开始"),
    (("改成", "改选", "换", "改用", "改为", "改回", "改去"), "更改"),
    (("完成", "提交", "做完"), "完成"),
    (("约定", "说好", "约好", "定了", "确定", "选了"), "约定"),
    (("喜欢", "更爱", "合口味", "偏好"), "偏好"),
]

# 判定为一对矛盾的动作组合（同主题、不同时间出现即记为冲突）
OPPOSITE = {("放弃", "开始"), ("开始", "放弃"), ("更改", "更改"),
            ("约定", "更改"), ("约定", "放弃")}

IMPORTANCE = {"放弃": 0.9, "更改": 0.8, "完成": 0.8, "约定": 0.8, "开始": 0.7,
              "选择": 0.7, "偏好": 0.6, "记录": 0.4}
_ALLOWED_TOPICS = {label for _keys, label in TOPIC_RULES} | {"生活"}
_ALLOWED_ACTIONS = set(IMPORTANCE)


def _grounded_llm_items(output, source_text: str) -> list[dict] | None:
    """Reject malformed or materially unsupported model extractions as a whole."""
    import json

    try:
        rows = json.loads(output) if isinstance(output, str) else output
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(rows, list) or not rows:
        return None

    source_grams = {
        source_text[i:i + 2]
        for i in range(max(0, len(source_text) - 1))
        if not source_text[i:i + 2].isspace()
    }
    grounded = []
    for item in rows:
        if not isinstance(item, dict):
            return None
        topic = str(item.get("topic", "")).strip()
        action = str(item.get("action", "")).strip()
        detail = str(item.get("detail", "")).strip()
        if topic not in _ALLOWED_TOPICS or action not in _ALLOWED_ACTIONS:
            return None
        if not detail or len(detail) > min(max(len(source_text) * 2, 80), 500):
            return None
        numbers = re.findall(r"\d+(?:年|月|日|号|点|次|个|公里|元|％|%)?", detail)
        if any(number not in source_text for number in numbers):
            return None
        detail_grams = {
            detail[i:i + 2]
            for i in range(max(0, len(detail) - 1))
            if not detail[i:i + 2].isspace()
        }
        if detail_grams and len(detail_grams & source_grams) / len(detail_grams) < 0.35:
            return None
        grounded.append({"topic": topic, "action": action, "detail": detail})
    return grounded


def _match(rules, text):
    for keys, label in rules:
        if any(k in text for k in keys):
            return label
    return None


def _match_topic(text: str) -> str | None:
    """Choose the most specific topic instead of the first single-word hit."""
    ranked = []
    for index, (keys, label) in enumerate(TOPIC_RULES):
        hits = [key for key in keys if key in text]
        if not hits:
            continue
        score = sum(max(len(key) - 1, 1) for key in set(hits))
        ranked.append((score, max(map(len, hits)), -index, label))
    return max(ranked)[-1] if ranked else None


def _split_sentences(text: str) -> list[str]:
    for sep in "。！？；\n":
        text = text.replace(sep, "|")
    return [s.strip() for s in text.split("|") if len(s.strip()) >= 4]


def _days_between(d1: str, d2: str) -> float:
    a = _date.fromisoformat(d1)
    b = _date.fromisoformat(d2)
    return abs((b - a).days)


# ---------- 1. 抽取并标注 ----------
def extract(store: MemoryStore, entry_text: str, entry_date: str, entry_id: str) -> list[MemoryItem]:
    """把一段原始记录抽成结构化记忆。配置了 LLM 时走 LLM，否则走规则。"""
    items: list[MemoryItem] = []

    if llm.is_enabled():
        try:
            out = llm.chat([
                {"role": "system", "content":
                    "从日记里抽取记忆条目，输出 JSON 数组，每条含 "
                    "topic(健身/运动/饮食/作息/课程/社交/阅读/旅行/理财/健康/生活)、"
                    "action(开始/放弃/更改/完成/约定/偏好/记录)、detail(一句话记忆)。"
                    "只能抽取原文明确支持、主体是用户本人的信息；不能推断、补写、合并第三方事实。只输出 JSON。"},
                {"role": "user", "content": f"日期：{entry_date}\n内容：{entry_text}"},
            ])
            extracted = _grounded_llm_items(out, entry_text)
            if extracted is None:
                raise ValueError("LLM extraction was malformed or unsupported by source")
            for it in extracted:
                items.append(store.add_memory(
                    date=entry_date, topic=it["topic"],
                    action=it["action"], detail=it["detail"],
                    importance=IMPORTANCE.get(it["action"], 0.5),
                    evidence=entry_id))
            return items
        except Exception:
            pass  # LLM 失败时回退到本地规则路径

    for sent in _split_sentences(entry_text):
        topic = _match_topic(sent) or "生活"
        action = _match(ACTION_RULES, sent) or "记录"
        items.append(store.add_memory(
            date=entry_date, topic=topic, action=action, detail=sent,
            importance=IMPORTANCE[action], evidence=entry_id))
    return items


# ---------- 2. 检测冲突 ----------
def detect_conflicts(store: MemoryStore, new_items: list[MemoryItem]) -> list[Conflict]:
    """同一主题上先后矛盾（如"放弃跑步" vs "重新开始跑步"）-> 记冲突，最新为准。"""
    found = []
    order = {m.id: i for i, m in enumerate(store.memories)}   # 写入顺序（同日改口以此判定先后）
    for ni in new_items:
        if ni.action not in ("开始", "放弃", "更改", "约定"):
            continue
        candidates = [
            m for m in store.memories
            if m.topic == ni.topic and m.action in ("开始", "放弃", "更改", "约定")
        ]
        for old in candidates:
            if old.id == ni.id or old.topic != ni.topic:
                continue
            if (old.action, ni.action) in OPPOSITE and order.get(old.id, 0) < order.get(ni.id, 0) \
                    and old.date <= ni.date:
                c = Conflict(id="c" + old.id[1:] + ni.id[1:], topic=ni.topic, action=ni.action,
                             older_id=old.id, newer_id=ni.id)
                store.conflicts.append(c)
                old.status = "superseded"   # 被更新的记忆取代，但保留可追溯
                c.resolved = True
                c.winner_id = ni.id
                found.append(c)
    return found


# ---------- 3. 提炼成结论 ----------
# 演变骨架动作（决定主题现状）与佐证动作（偏好/完成）分开取材
_SKELETON_ACTIONS = ("更改", "放弃", "开始", "约定")


def _evolution_text(topic: str, items: list, key_events: list) -> str:
    """结论正文：按时间轴呈现主题演变（状态变更类事件为骨架），
    末尾保留最近一条佐证事件（偏好/完成），没有关键事件的主题退回概括式。"""
    if key_events:
        skeleton = sorted((m for m in key_events if m.action in _SKELETON_ACTIONS),
                          key=lambda x: x.date)
        backup = [m for m in key_events if m.action not in _SKELETON_ACTIONS]
        if not skeleton:
            skeleton = sorted(backup, key=lambda x: x.date)
            backup = []
        if len(skeleton) > 4:   # 演变过长时保留首尾，中间以省略号带过
            skeleton = skeleton[:2] + skeleton[-2:]
        chain = "；".join(f"{m.date[5:]} {m.detail}" for m in skeleton)
        text = f"{topic}主题演变：{chain}"
        if backup:
            latest = max(backup, key=lambda x: x.date)
            text += f"。近期：{latest.detail}"
        else:
            recent = sorted(items, key=lambda x: x.date)[-1]
            text += f"。近期：{recent.detail}"
        return text
    sample = "；".join(m.detail for m in sorted(items, key=lambda x: x.date)[-2:])
    return f"长期反复出现：{topic}相关的记录有 {len(items)} 条（{sample}）"


def consolidate(store: MemoryStore) -> list[Conclusion]:
    """同主题反复出现（>=3 条，或含关键事件且 >=2 条）-> 升级为结论。
    结论正文按主题演变组织：关键事件（开始/放弃/更改/约定）是时间线骨架，
    保证"总结"类问题能引用到'从什么改成什么'的完整变化过程。"""
    made = []
    groups: dict[str, list[MemoryItem]] = {}
    for m in store.active_memories():
        groups.setdefault(m.topic, []).append(m)
    all_by_topic: dict[str, list[MemoryItem]] = {}
    for m in store.memories:
        all_by_topic.setdefault(m.topic, []).append(m)
    for topic, items in groups.items():
        repeated = [m for m in items if m.action == "记录"]
        # 关键事件从全库取（含档案层）：关键事件随冲突进入档案层后仍是演变骨架
        key_events = [m for m in all_by_topic.get(topic, []) if m.action != "记录"]
        if not (len(items) >= 3 or (key_events and len(items) >= 2)):
            continue
        if not (len(repeated) >= 3 or len(items) >= 5 or key_events):
            continue
        text = _evolution_text(topic, items, key_events)
        exist = [c for c in store.conclusions if c.topic == topic]
        if exist:
            exist[0].text = text
            exist[0].support = [m.id for m in items]
            exist[0].updated = max(m.date for m in items)
            made.append(exist[0])
        else:
            c = Conclusion(id="s" + topic[:2], topic=topic, text=text,
                           support=[m.id for m in items],
                           updated=max(m.date for m in items))
            store.conclusions.append(c)
            made.append(c)
        # 折叠：结论生成后，支撑它的"流水账记录"升华进结论，让位出活跃层
        #（关键事件类记忆不折叠，它们是不可压缩的事实节点）
        # 例外：泛化的"生活"主题是日常长尾，折叠后会让普通回忆类问题在
        # 活跃层扑空，因此只折叠具体主题（健身/饮食/作息/课程/运动/社交）
        for mid in (made[-1].support if made else []):
            m = store.memory_by_id(mid)
            if (
                m
                and m.action == "记录"
                and m.topic != "生活"
                and m.memory_type == "long_term"
            ):
                m.status = "compressed"
    return made


# ---------- 4. 决定去留（主动遗忘） ----------
def retention_score(m: MemoryItem, now: str) -> float:
    r = config.RETENTION
    recency = math.exp(-_days_between(m.date, now) / r["tau_days"])
    return (r["w_recency"] * recency
            + r["w_importance"] * m.importance
            + r["w_freq"] * min(m.frequency, 5) / 5.0)


def apply_retention(store: MemoryStore, now: str) -> list[dict]:
    """给每条活跃记忆打分，按阈值决定保留/压缩/丢弃，并记审计日志。"""
    decisions = []
    protected = set()
    for c in store.conflicts:
        protected.add(c.older_id)
        protected.add(c.newer_id)
    for con in store.conclusions:
        # 只有"关键事件类"支撑受结论保护；流水账已折叠进结论，不再豁免
        for mid in con.support:
            m = store.memory_by_id(mid)
            if m and m.action != "记录":
                protected.add(mid)

    for m in store.active_memories():
        if m.id in protected:
            continue
        s = retention_score(m, now)
        if s < config.RETENTION["discard_below"]:
            m.status = "discarded"
            decision = "discard"
        elif s < config.RETENTION["compress_below"]:
            m.status = "compressed"
            decision = "compress"
        else:
            decision = "keep"
        if decision != "keep":
            decisions.append({"memory_id": m.id, "date": m.date, "topic": m.topic,
                              "detail": m.detail, "score": round(s, 3), "decision": decision})
    store.decisions.extend(decisions)
    return decisions
