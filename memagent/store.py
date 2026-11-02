# 数据层 + 记忆存储：原始时间线记录、结构化记忆条目、冲突、结论，统一持久化为 JSON
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import config


@dataclass
class Entry:
    """一条原始记录（日记/笔记/日程导出），带日期，按时间轴存放。"""
    id: str
    date: str
    text: str
    source: str = "diary"
    kind: str = "record"
    parent_id: str = ""


@dataclass
class MemoryItem:
    """从原始记录中抽取的一条结构化记忆。"""
    id: str
    date: str
    topic: str          # 主题：健身/饮食/作息/课程/运动/社交/生活
    action: str         # 动作类型：开始/放弃/更改/完成/约定/偏好/记录
    detail: str         # 记忆内容（原句或摘要）
    importance: float = 0.5
    frequency: int = 1
    status: str = "active"   # active / superseded / compressed / discarded
    evidence: str = ""       # 对应原始记录 id
    memory_type: str = "long_term"
    confidence: float = 1.0
    parent_id: str = ""
    preference_key: str = ""
    entities: list[str] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)
    semantic_values: list[str] = field(default_factory=list)


@dataclass
class Conflict:
    """同一主题上先后矛盾的两条记忆；默认"最新为准"解决。"""
    id: str
    topic: str
    action: str
    older_id: str
    newer_id: str
    resolved: bool = False
    winner_id: str = ""


@dataclass
class Conclusion:
    """由反复出现的经历提炼出的结论（记忆的"压缩"形态）。"""
    id: str
    topic: str
    text: str
    support: list = field(default_factory=list)   # 支撑该结论的 memory id
    updated: str = ""


class MemoryStore:
    def __init__(self, path: Optional[str] = None):
        self.path = str(path or config.STORE_FILE)
        self.entries: list[Entry] = []
        self.memories: list[MemoryItem] = []
        self.conflicts: list[Conflict] = []
        self.conclusions: list[Conclusion] = []
        self.decisions: list[dict] = []   # 遗忘决策审计日志
        self._save_lock = threading.RLock()
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if not os.path.exists(self.path):
            return
        raw = None
        for attempt in range(50):
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                break
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                if attempt == 49:
                    raise
                time.sleep(0.01)
        if raw is None:
            return
        self.entries = [Entry(**e) for e in raw.get("entries", [])]
        self.memories = [MemoryItem(**m) for m in raw.get("memories", [])]
        self.conflicts = [Conflict(**c) for c in raw.get("conflicts", [])]
        self.conclusions = [Conclusion(**c) for c in raw.get("conclusions", [])]
        self.decisions = raw.get("decisions", [])

    def save(self):
        data = {
            "entries": [asdict(e) for e in self.entries],
            "memories": [asdict(m) for m in self.memories],
            "conflicts": [asdict(c) for c in self.conflicts],
            "conclusions": [asdict(c) for c in self.conclusions],
            "decisions": self.decisions,
        }
        target = os.path.abspath(self.path)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        temporary = ""
        with self._save_lock:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=parent,
                    prefix=os.path.basename(target) + ".",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary = stream.name
                    json.dump(data, stream, ensure_ascii=False, indent=1)
                    stream.flush()
                    os.fsync(stream.fileno())
                for attempt in range(50):
                    try:
                        os.replace(temporary, target)
                        temporary = ""
                        break
                    except PermissionError:
                        if attempt == 49:
                            raise
                        time.sleep(0.01)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)

    # ---------- 写入 ----------
    def add_entry(
        self,
        date: str,
        text: str,
        source: str = "diary",
        kind: str = "record",
        parent_id: str = "",
    ) -> Entry:
        e = Entry(
            id="e" + uuid.uuid4().hex[:8],
            date=date,
            text=text,
            source=source,
            kind=kind,
            parent_id=parent_id,
        )
        self.entries.append(e)
        return e

    def add_memory(self, **kw) -> MemoryItem:
        m = MemoryItem(id="m" + uuid.uuid4().hex[:8], **kw)
        self.memories.append(m)
        return m

    # ---------- 查询 ----------
    def active_memories(self) -> list[MemoryItem]:
        return [m for m in self.memories if m.status == "active"]

    def archive_memories(self) -> list[MemoryItem]:
        """档案层：被更新取代/压缩归档/主动遗忘的历史记忆。不参与日常检索，
        但时间回溯类问题可以查到——遗忘不等于销毁证据链。"""
        return [m for m in self.memories if m.status in ("superseded", "compressed", "discarded")]

    def memory_by_id(self, mid: str) -> Optional[MemoryItem]:
        for m in self.memories:
            if m.id == mid:
                return m
        return None

    def conclusion_for(self, topic: str) -> Optional[Conclusion]:
        best = None
        for c in self.conclusions:
            if c.topic == topic and (best is None or c.updated > best.updated):
                best = c
        return best

    def conflicts_in(self, memory_ids: set) -> list[Conflict]:
        return [c for c in self.conflicts
                if c.older_id in memory_ids or c.newer_id in memory_ids]

    def stats(self) -> dict:
        by_status = {}
        for m in self.memories:
            by_status[m.status] = by_status.get(m.status, 0) + 1
        return {
            "entries": len(self.entries),
            "memories_total": len(self.memories),
            "memories_by_status": by_status,
            "conflicts": len(self.conflicts),
            "conflicts_resolved": sum(1 for c in self.conflicts if c.resolved),
            "conclusions": len(self.conclusions),
            "forget_decisions": len(self.decisions),
        }
