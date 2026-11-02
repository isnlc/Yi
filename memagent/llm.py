# LLM 客户端与任务决策接口：OpenAI 兼容接口，纯标准库实现
# 通过环境变量启用：
#   MEMAGENT_BASE_URL  例如 https://dashscope.aliyuncs.com/compatible-mode/v1 或 https://api.deepseek.com/v1
#   MEMAGENT_API_KEY   对应密钥
#   MEMAGENT_MODEL     模型名，默认 deepseek-chat
#   MEMAGENT_LLM       设为 0 可强制关闭 LLM（即使配置了密钥），用于离线复现
# 未配置时 is_enabled() 返回 False，上层自动回退到规则模式，保证离线可跑。
import json
import os
import re
import urllib.request
from pathlib import Path


def _load_dotenv() -> None:
    """加载项目根目录的 .env（键值对，# 开头为注释）。不覆盖已存在的环境变量。"""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def is_enabled() -> bool:
    """LLM 为显式开启（MEMAGENT_LLM=1）+ 密钥齐全，默认离线保证实验可复现。
    交互式入口（agent.py）会自动置 MEMAGENT_LLM=1。"""
    if os.environ.get("MEMAGENT_LLM", "0") != "1":
        return False
    return bool(os.environ.get("MEMAGENT_BASE_URL") and os.environ.get("MEMAGENT_API_KEY"))




def vision_is_enabled() -> bool:
    """Enable vision only when a vision-capable endpoint is configured explicitly."""
    if not is_enabled():
        return False
    return bool(
        os.environ.get("MEMAGENT_VISION_MODEL")
        and os.environ.get("MEMAGENT_VISION_BASE_URL")
        and os.environ.get("MEMAGENT_VISION_API_KEY")
    )


def vision_chat(
    prompt: str,
    data_url: str | list[str],
    temperature: float = 0.2,
    system_prompt: str | None = None,
) -> str:
    """Ask an OpenAI-compatible vision model about one or more images."""
    base = os.environ["MEMAGENT_VISION_BASE_URL"].rstrip("/")
    api_key = os.environ["MEMAGENT_VISION_API_KEY"]
    model = os.environ["MEMAGENT_VISION_MODEL"]
    image_urls = [data_url] if isinstance(data_url, str) else data_url
    content = [
        {"type": "image_url", "image_url": {"url": url}}
        for url in image_urls
    ]
    content.append({"type": "text", "text": prompt})
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]

def chat(messages: list, temperature: float = 0.2) -> str:
    base = os.environ["MEMAGENT_BASE_URL"].rstrip("/")
    model = os.environ.get("MEMAGENT_MODEL", "deepseek-chat")
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["MEMAGENT_API_KEY"],
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def chat_json(messages: list, temperature: float = 0.0):
    """调用 LLM 并解析 JSON 输出（容忍代码围栏与前后缀文字）。失败抛异常，由上层回退规则。"""
    text = chat(messages, temperature).strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    if not text.startswith(("[", "{")):
        m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
        if not m:
            raise ValueError("LLM output is not JSON: " + text[:120])
        text = m.group(1)
    return json.loads(text)


# ==================== 任务决策接口 ====================
# 以下函数对应冲突治理、证据检索和回答审核中的语义判断环节。
# 全部为可选增强：LLM 不可用时上层走规则兜底，接口不变。

RECONCILER_SYSTEM = (
    "你负责记忆系统的冲突治理。下面给出同一主题下两两配对的记忆条目"
    "（older 较早，newer 较晚）。判断 newer 是否构成对 older 所述事项的"
    "推翻、改口或替换（时间表变了、方案换了、状态反转都算）。"
    '只输出 JSON 数组：[{"pair_id": "...", "conflict": true/false, "reason": "一句话理由"}]，'
    "不要输出其他内容。"
)

AUDITOR_SYSTEM = (
    "你负责记忆问答系统的回答审核。根据问题和候选证据，判断："
    "1) 证据是否足以回答问题；2) 是否存在明显的过时信息（若问题在问现状，"
    "而证据里有更晚的相反说法，则较早的过时）。"
    '只输出 JSON：{"sufficient": true/false, "stale": ["过时证据的日期"], '
    '"action": "answer" 或 "refine" 或 "refuse", "reason": "一句话理由"}。'
    "answer=证据足够；refine=需要换关键词再查一轮；refuse=记录里没有相关内容，应拒答。"
)

PLANNER_SYSTEM = (
    "你负责记忆问答系统的证据检索。根据问题选择首轮检索路由："
    "time+lex=问具体时间点/历史事件；lex+graph=问关系/习惯/整体总结；"
    "lex+time=其余一般问题（默认）。"
    '只输出 JSON：{"route": "...", "reason": "一句话理由"}。'
)

REFINER_SYSTEM = (
    "你负责记忆问答系统的证据检索。上一轮检索证据不足，请根据问题和已查到的"
    "零散证据，给出一个更可能命中的新查询短语（换同义词/换表述/换角度）。"
    '只输出 JSON：{"new_query": "...", "reason": "一句话理由"}。'
)


def judge_conflicts(pairs: list[dict]) -> list[dict]:
    """冲突治理：批量判断记忆对是否构成冲突。"""
    if not pairs:
        return []
    listing = "\n".join(
        f'{p["pair_id"]}: older=({p["older"]["date"]}) {p["older"]["detail"]} | '
        f'newer=({p["newer"]["date"]}) {p["newer"]["detail"]}'
        for p in pairs
    )
    return chat_json([
        {"role": "system", "content": RECONCILER_SYSTEM},
        {"role": "user", "content": listing},
    ])


def audit_evidence(question: str, evidence_lines: list[str], conflict_note: str = "") -> dict:
    """回答审核：证据充分性与过时性校验。"""
    ctx = "\n".join(evidence_lines) or "（无证据）"
    extra = f"\n已知冲突：{conflict_note}" if conflict_note else ""
    return chat_json([
        {"role": "system", "content": AUDITOR_SYSTEM},
        {"role": "user", "content": f"问题：{question}\n候选证据：\n{ctx}{extra}"},
    ])


def plan_route(question: str) -> dict:
    """证据检索：首轮路由决策。"""
    return chat_json([
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": question},
    ])


def refine_query(question: str, prev_query: str, evidence_lines: list[str]) -> dict:
    """证据检索：证据不足时改写查询。"""
    ctx = "\n".join(evidence_lines[:5]) or "（空）"
    return chat_json([
        {"role": "system", "content": REFINER_SYSTEM},
        {"role": "user", "content": f"问题：{question}\n上一轮查询：{prev_query}\n已查到：\n{ctx}"},
    ])
