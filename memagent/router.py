"""统一查询路由：先判断用户意图，再决定是否进入个人记忆系统。"""
from __future__ import annotations

import re

from . import llm
from .memory_policy import looks_like_question
from .retrieve import is_temporal_activity_query
from .semantics import analyze_query, analyze_semantics

FOLLOW_UP_MARKERS = ("那", "后来", "第二天", "继续", "呢", "它", "当时", "再")
_TEMPORAL_REFERENCE = (
    r"(?:20\d{2}\s*年\s*)?"
    r"(?:1[0-2]|0?[1-9]|十一|十二|十|一|二|三|四|五|六|七|八|九)\s*月"
    r"(?:\s*[0-3]?\d\s*[日号])?|"
    r"最近\s*\d+\s*天|最近|近期|近来|近几天|这几天|最近一段时间|最近一阵子|"
    r"今天|昨天|前天|上周|本周|这周|上个月|本月|这个月|"
    r"国庆|春节|暑假|五一|开学|期中|期末"
)

_RECORD_RE = re.compile(
    r"(帮我|替我|请).{0,3}(记一下|记着|记录(?:一下)?|记住|保存(?:一下)?|存进记忆|归档(?:一下)?)|"
    r"(记一下|记录一下|备忘|提醒我)"
)
_REMINDER_RE = re.compile(r"(不要|别)忘(?:记|了|掉).{1,}")
_RECORD_NEGATION_RE = re.compile(
    r"(不要|不用|无需|别).{0,8}(记录|记着|记住|保存|存进|写入|归档|提醒)"
)
_RECORD_PREFIX_RE = re.compile(
    r"^\s*(?:请)?(?:帮我|替我)?\s*"
    r"(?:记一下|记着|记录(?:一下)?|记住|保存(?:一下)?|存进记忆|写入记忆|归档(?:一下)?|备忘|提醒我(?:一下)?)"
)
_FORGET_RE = re.compile(
    r"^(帮我|请|我想|给我)?(忘掉|忘记掉|忘记|删掉|删除|清除|移除).+|"
    r"(从记忆库|从记录里).{0,6}(删掉|删除|清除|移除)"
)
_REVIEW_RE = re.compile(
    r"回顾|总结.{0,8}(最近|上周|本周|上个月|过去)|"
    r"(我)?(最近一周|上周|本周|上个月).{0,8}(做了什么|干了什么|忙什么)|"
    rf"(回顾|复盘).{{0,8}}({_TEMPORAL_REFERENCE})"
)
_UTILITY_RE = re.compile(
    r"^\s*(?:请问|帮我查查|查一下|告诉我)?(?:今天|现在|当前)(?:是|为)?"
    r"(?:星期几|周几|几号|什么日期|几点|什么时间)(?:了)?[？?]?\s*$|"
    r"^\s*(?:请问|帮我查查|查一下|告诉我)?"
    r"(?:今天几号|今天星期几|今天周几|现在几点|当前时间)(?:了)?[？?]?\s*$"
)
_ADVICE_RE = re.compile(
    r"应该(?:怎么|如何|做|选|注意|准备)|该怎么|怎么办|如何做|给.*建议|推荐|"
    r"帮我写|改写|润色|翻译|什么好|哪个好|怎么选"
)
_GENERAL_STRONG_RE = re.compile(
    r"^(请)?(用.{0,12})?(讲讲|解释|介绍|科普)|"
    r"什么是|是什么意思|原理是什么|为什么会|"
    r"^(请)?帮我(写|改写|润色|翻译|生成|起草)|^(请)?推荐"
)
_PERSONAL_ADVICE_RE = re.compile(
    r"(?:根据|结合|参考|按照|按).{0,8}"
    r"(?:我的|本人).{0,8}(?:偏好|口味|习惯|情况|记录|经历|需求|预算|安排|计划)|"
    r"(?:适合|符合).{0,4}(?:我的|本人).{0,6}(?:偏好|口味|习惯|情况|需求|预算)"
)
_CATALOG_FACETS = (
    r"种类|类型|类别|分类|口味|风味|做法|方式|形式|款式|风格|"
    r"型号|规格|功能|用途|选项|流派|品种|版本|烘焙程度|档位|材质|模式"
)
_GENERAL_CATALOG_RE = re.compile(
    rf"(?:有哪些|有什么|包括哪些|有哪几种|有哪几类).{{0,8}}(?:{_CATALOG_FACETS})|"
    rf"(?:{_CATALOG_FACETS}).{{0,8}}(?:有哪些|有什么|包括哪些|有哪几种|有哪几类)|"
    r"(?:分为哪些|分哪几种|分哪几类|分为哪几种|分为哪几类)"
)
_MEMORY_REASONING_RE = re.compile(
    r"建议|推荐|该不该|是否适合|合适吗|合理吗|怎么办|"
    r"怎么(?:改善|调整|优化|安排|选择|取舍)|"
    r"如何(?:改善|调整|优化|安排|选择|规划|处理)|"
    r"为什么|原因|分析|评估|比较|对比|取舍|优先级|"
    r"哪个更|哪种更|最适合|更适合|"
    r"总结|归纳|概括|复盘|制定|规划|预测|推断|风险|"
    r"有什么问题|问题在哪里|写成|整理成|生成"
)
_STRONG_MEMORY_RULES = (
    re.compile(r"记忆库|记录里|日记里|历书里|你记得|还记得"),
    re.compile(r"(我|我的).*(最近一次|上一次|最后一次|上次).*(什么|什么时候|哪天|是|做|吃|喝|去|买|看|跑)"),
    re.compile(r"(我|我的|我们).*(之前|以前|当时|过去|国庆|春节|暑假|上周|本周|上个月).*(做|去|买|看|喝|吃|发生|忙|经历)"),
    re.compile(r"(我|我的).*(今天|昨天|前天).*(吃了|喝了|做了|去了|买了|看了|发生了什么)"),
    re.compile(rf"(我|我的|我们).*({_TEMPORAL_REFERENCE}).*(做|去|买|看|喝|吃|发生|忙|经历|记录)"),
    re.compile(r"(我|我的).*(现在|目前|当前|还在|最后).*(喝|吃|用|住|做|学|喜欢|偏好|坚持|计划|项目|安排|约定)"),
    re.compile(r"(我的|我有).*(兴趣|爱好|习惯|偏好|计划|项目|安排|约定|经历)"),
)
_PERSONAL_CONTEXT_RE = re.compile(
    r"(根据|结合|参考|看看).{0,8}(我的|我之前|我最近).{0,8}(记录|记忆|经历|情况)|"
    r"(我的|我).{0,8}(记录|记忆|经历).{0,8}(里|中|显示|来看)"
)
_MEMORY_RULES = (
    re.compile(r"(记忆库|记录里|日记里|历书里|你记得|还记得).*(我|我的|什么|吗|？|\?)?"),
    re.compile(r"(我|我的|我们).*(之前|以前|最近|上次|当时|后来|过去|国庆|春节|暑假|上周|本周|上个月).*(做|去|买|看|喝|吃|发生|忙|经历|记录)"),
    re.compile(r"(我|我的).*(现在|目前|当前|还在|最后).*(喝|吃|用|住|做|学|喜欢|偏好|坚持|计划|项目|安排|约定)"),
    re.compile(r"(我|我的).*(什么时候|哪天|几月|多久|最近一次|做过|去过|买过|看过|喝过|吃过)"),
    re.compile(r"(我|我的).*(今天|昨天|前天|最近).*(吃了|喝了|做了|去了|买了|看了|发生了什么)"),
    re.compile(rf"(我|我的|我们).*({_TEMPORAL_REFERENCE}).*(做|去|买|看|喝|吃|发生|忙|经历|记录)"),
    re.compile(r"(我的|我有).*(兴趣|爱好|习惯|偏好|计划|项目|安排|约定|经历)"),
    re.compile(r"(总结|梳理).*(我|我的|最近|过去)"),
)

_ROUTER_SYSTEM = (
    "你是查询路由器。你的唯一任务是判断：回答当前问题是否需要先检索用户的个人记忆库。"
    "memory=需要用户自己的历史事实才能回答，包括过去经历、今天或最近做过什么、"
    "最近一次事件、当前个人状态、偏好、习惯、计划、约定、项目进展，以及基于个人记录提出的建议。"
    "general=不需要个人历史即可回答，包括世界知识、写作改写、通用推荐、普通建议和计算。"
    "第一人称本身不代表 memory，例如‘我该怎么学 Python’是 general；"
    "但‘根据我最近的学习记录，我该怎么调整计划’是 memory。"
    "日期时间由上层工具处理，不会交给你。"
    "示例："
    "‘我今天吃了什么’=memory；"
    "‘我最近一次跑步是什么时候’=memory；"
    "‘我现在喝什么咖啡’=memory；"
    "‘我国庆做了什么’=memory；"
    "‘咖啡有哪些烘焙程度’=general；"
    "‘如何制定跑步计划’=general；"
    "‘用一句话讲讲什么是黑洞’=general。"
    '只输出 JSON：{"intent":"memory"或"general","reason":"一句话","confidence":0到1的小数}。'
)


def _explicit_intent(question: str) -> str | None:
    if is_record_command(question):
        return "record"
    if is_forget_command(question):
        return "forget"
    if _REVIEW_RE.search(question):
        return "review"
    if _UTILITY_RE.search(question):
        return "utility"
    return None


def is_record_command(text: str) -> bool:
    """Only explicit positive storage commands enter the write pipeline."""
    text = text.strip()
    if _REMINDER_RE.search(text):
        return True
    if _RECORD_NEGATION_RE.search(text):
        return False
    return bool(_RECORD_RE.search(text))


def is_forget_command(text: str) -> bool:
    """Distinguish deleting an existing memory from 'do not remember this'."""
    text = text.strip()
    if _REMINDER_RE.search(text) or _RECORD_NEGATION_RE.search(text):
        return False
    return bool(_FORGET_RE.search(text))


def extract_record_content(text: str) -> str:
    """Remove only the write command while preserving the user's facts."""
    if not is_record_command(text):
        return ""
    content = _RECORD_PREFIX_RE.sub("", text.strip(), count=1)
    if content == text.strip() and _REMINDER_RE.search(text):
        content = re.sub(
            r"^\s*(?:请)?(?:帮我|替我)?\s*(?:不要|别)忘(?:记|了|掉)",
            "",
            text,
            count=1,
        )
    return content.strip(" ：:，。,.")


def _semantic_profile_query(question: str) -> bool:
    frame = analyze_semantics(question, is_query=True)
    personal_profile = bool(
        re.search(
            r"(?:我|我的|本人).*(?:什么|哪种|哪个|多少|几次|多久|什么时候)",
            question,
        )
    )
    return personal_profile and bool(frame.facets or frame.entities)


def _is_general_catalog_query(question: str) -> bool:
    """识别不依赖个人历史的分类、枚举类知识问题。"""
    if re.search(
        r"(?:我的|本人|按我|根据我|结合我|适合我)|"
        r"(?:记忆库|记录里|日记里|历书里|你记得|还记得)",
        question,
    ):
        return False
    return bool(_GENERAL_CATALOG_RE.search(question))


def _strong_scope(question: str) -> str | None:
    """只处理边界明确的问题，含糊表达仍交给语义模型。"""
    if is_temporal_activity_query(question):
        return "memory"
    if _PERSONAL_CONTEXT_RE.search(question):
        return "memory"
    if _PERSONAL_ADVICE_RE.search(question):
        return "memory"
    if _is_general_catalog_query(question):
        return "general"
    local_frame = analyze_semantics(question, is_query=True)
    if (
        local_frame.personal is True
        and local_frame.entities
        and local_frame.scope in ("current", "history", "evolution")
    ):
        return "memory"
    if (
        local_frame.personal is True
        and local_frame.entities
        and re.search(r"一般|通常|平时|习惯|偏好", question)
    ):
        return "memory"
    if _GENERAL_STRONG_RE.search(question) or _ADVICE_RE.search(question):
        return "general"
    if any(pattern.search(question) for pattern in _STRONG_MEMORY_RULES):
        return "memory"
    if _semantic_profile_query(question):
        return "memory"
    return None


def _rule_scope(question: str) -> str:
    if is_temporal_activity_query(question):
        return "memory"
    if _PERSONAL_CONTEXT_RE.search(question):
        return "memory"
    if _PERSONAL_ADVICE_RE.search(question):
        return "memory"
    if _is_general_catalog_query(question):
        return "general"
    if _ADVICE_RE.search(question):
        return "general"
    if any(pattern.search(question) for pattern in _MEMORY_RULES):
        return "memory"
    if _semantic_profile_query(question):
        return "memory"
    return "general"


def _has_probe_evidence(memory_probe) -> bool:
    if memory_probe is None:
        return False
    try:
        return bool(memory_probe() if callable(memory_probe) else memory_probe)
    except Exception:
        return False


def _memory_answer_mode(question: str, semantic_scope: str = "") -> str:
    """Facts are rendered deterministically; synthesis/advice uses retrieved evidence plus LLM."""
    if semantic_scope == "advice" or _MEMORY_REASONING_RE.search(question):
        return "retrieval_llm"
    return "retrieval_only"


def _route_result(
    intent: str,
    method: str,
    reason: str,
    question: str,
    *,
    semantic_scope: str = "",
    answer_mode: str = "",
    **extra,
) -> dict:
    if not answer_mode:
        if intent == "memory":
            answer_mode = _memory_answer_mode(question, semantic_scope)
        elif intent == "review":
            answer_mode = "retrieval_llm"
        elif intent == "general":
            answer_mode = "general_llm"
        else:
            answer_mode = "local_action"
    return {
        "intent": intent,
        "method": method,
        "reason": reason,
        "answer_mode": answer_mode,
        "requires_llm": answer_mode in ("retrieval_llm", "general_llm"),
        **({"semantic_scope": semantic_scope} if semantic_scope else {}),
        **extra,
    }


def classify_query(
    question: str,
    previous_scope: str | None = None,
    previous_answer_mode: str | None = None,
    memory_probe=None,
) -> dict:
    """返回 intent/method/reason/answer_mode。

    intent 决定是否读取个人记忆；answer_mode 决定读取后是否需要 LLM：
    retrieval_only=事实回查，retrieval_llm=检索后分析/建议，general_llm=普通问答。
    """
    question = question.strip()
    explicit = _explicit_intent(question)
    if explicit:
        return _route_result(
            explicit,
            "explicit",
            "明确指令或本地工具问题",
            question,
        )

    if previous_scope in ("memory", "general") and question.startswith(FOLLOW_UP_MARKERS):
        inherited_mode = ""
        if previous_answer_mode in ("retrieval_only", "retrieval_llm", "general_llm"):
            inherited_mode = previous_answer_mode
        if previous_scope == "memory" and _MEMORY_REASONING_RE.search(question):
            inherited_mode = "retrieval_llm"
        return _route_result(
            previous_scope,
            "followup",
            "明确省略追问，继承上一轮范围",
            question,
            answer_mode=inherited_mode,
        )

    if not looks_like_question(question):
        return _route_result(
            "general",
            "conversation",
            "当前表达不要求检索个人历史",
            question,
        )

    strong_scope = _strong_scope(question)
    if strong_scope:
        local_frame = analyze_query(question, use_llm=False)
        semantic_scope = (
            "history"
            if is_temporal_activity_query(question)
            else local_frame.scope
        )
        return _route_result(
            strong_scope,
            "rule_fast",
            "边界明确，使用本地快速分流",
            question,
            semantic_scope=semantic_scope,
        )

    if llm.is_enabled():
        frame = analyze_query(question, use_llm=True)
        local_frame = analyze_query(question, use_llm=False)
        if (
            frame.personal is False
            and local_frame.personal is None
            and _has_probe_evidence(memory_probe)
        ):
            return _route_result(
                "memory",
                "evidence_probe",
                "问法可作通用理解，但记忆库中存在与对象和属性直接匹配的个人事实",
                question,
                semantic_scope=frame.scope,
                normalized_query=frame.normalized_query,
            )
        if frame.personal is not None:
            intent = "memory" if frame.personal else "general"
            return _route_result(
                intent,
                "llm",
                (
                    "问题依赖用户自己的既定事实、习惯、配置或经历"
                    if frame.personal
                    else "问题可在不读取个人记忆的情况下回答"
                ),
                question,
                semantic_scope=frame.scope,
                normalized_query=frame.normalized_query,
            )
        if _has_probe_evidence(memory_probe):
            return _route_result(
                "memory",
                "evidence_probe",
                "表达存在歧义，但个人记忆库中有直接匹配的既定事实",
                question,
                semantic_scope=frame.scope,
                normalized_query=frame.normalized_query,
            )

    rule_intent = _rule_scope(question)
    local_frame = analyze_query(question, use_llm=False)
    if local_frame.personal is True:
        rule_intent = "memory"
    elif local_frame.personal is None and _has_probe_evidence(memory_probe):
        rule_intent = "memory"
    reason = (
        "问题依赖用户自己的既定事实或经历"
        if rule_intent == "memory"
        else "问题不依赖个人历史"
    )
    return _route_result(
        rule_intent,
        "offline_fallback",
        reason,
        question,
        semantic_scope=local_frame.scope,
    )
