# 单元测试：覆盖记忆层（抽取/冲突/提炼/遗忘）、检索层（路由/同义词/档案层）、评测判分
# 运行：python -m unittest discover tests -v
import os
import sys
import tempfile
import unittest
from datetime import date as _d
from pathlib import Path

os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memagent import config
from memagent.memory import (
    _grounded_llm_items,
    extract,
    detect_conflicts,
    consolidate,
    retention_score,
    apply_retention,
)
from memagent.retrieve import Retriever, tokenize, expand_query
from memagent.store import MemoryStore
from memagent.agents import MemoryAgent
from memagent.evaluate import score_question


def _tmp_store() -> MemoryStore:
    """临时库：每个用例独立的空库，不污染正式数据。"""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    return MemoryStore(path=path)


class TestMemoryLayer(unittest.TestCase):
    """记忆层：抽取 -> 冲突 -> 提炼 -> 遗忘"""

    def test_extract_topic_action(self):
        store = _tmp_store()
        items = extract(store, "今天下决心开始跑步计划，每周三次。", "2026-03-05", "e1")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].topic, "健身")
        self.assertEqual(items[0].action, "开始")

    def test_extract_unmatched_defaults_to_life(self):
        store = _tmp_store()
        items = extract(store, "今天天气不错。", "2026-03-06", "e1")
        self.assertEqual(items[0].topic, "生活")
        self.assertEqual(items[0].action, "记录")

    def test_ingest_topic_override_preserves_confirmed_interaction_topic(self):
        store = _tmp_store()
        agent = MemoryAgent(store)
        _entry, items, _conflicts = agent.ingest(
            "2026-11-02",
            "我爱蒜香",
            memory_type="long_term",
            action_override="偏好",
            topic_override="饮食",
            preference_key="蒜香",
        )

        self.assertEqual(items[0].topic, "饮食")
        self.assertEqual(items[0].action, "偏好")

    def test_model_extraction_must_be_supported_by_source(self):
        source = "我每周跑步三次，准备坚持到期末。"
        supported = _grounded_llm_items(
            '[{"topic":"健身","action":"开始","detail":"我每周跑步三次，准备坚持到期末"}]',
            source,
        )
        unsupported = _grounded_llm_items(
            '[{"topic":"健身","action":"开始","detail":"用户计划每天跑步十公里并准备参加马拉松"}]',
            source,
        )
        self.assertEqual(supported[0]["detail"].rstrip("。"), source.rstrip("。"))
        self.assertIsNone(unsupported)

    def test_conflict_latest_wins(self):
        store = _tmp_store()
        old = extract(store, "跑步计划正式放弃。", "2026-05-20", "e1")[0]
        new = extract(store, "开学。重新开始跑步计划。", "2026-09-01", "e2")[0]
        conflicts = detect_conflicts(store, [new])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].older_id, old.id)
        self.assertEqual(conflicts[0].winner_id, new.id)
        self.assertEqual(old.status, "superseded")   # 旧的被取代，进档案层

    def test_consolidation_folds_records(self):
        store = _tmp_store()
        for i, d in enumerate(["2026-04-10", "2026-04-24", "2026-05-18"]):
            extract(store, "周三羽毛球，打得很爽。", d, f"e{i}")
        made = consolidate(store)
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0].topic, "运动")
        folded = [m for m in store.memories if m.status == "compressed"]
        self.assertEqual(len(folded), 3)   # 流水账折叠进结论，让位活跃层

    def test_consolidation_keeps_conversational_events_active(self):
        store = _tmp_store()
        for day, detail in (
            ("2026-10-01", "今天跑步三公里。"),
            ("2026-10-02", "今天跑步五公里。"),
            ("2026-10-03", "今天完成跑步训练。"),
        ):
            store.add_memory(
                date=day,
                topic="健身",
                action="记录",
                detail=detail,
                memory_type="long_term",
            )
        event = store.add_memory(
            date="2026-11-02",
            topic="健身",
            action="记录",
            detail="我今天要去跑步。",
            memory_type="event",
        )

        consolidate(store)

        self.assertEqual(event.status, "active")

    def test_retention_protects_key_events(self):
        store = _tmp_store()
        key = extract(store, "重新开始跑步计划，目标五公里。", "2026-09-01", "e1")[0]
        decisions = apply_retention(store, config.NOW)
        self.assertEqual(key.status, "active")          # 关键事件受保护
        self.assertEqual(decisions, [])

    def test_retention_discards_old_noise(self):
        store = _tmp_store()
        noise = extract(store, "学校银杏黄了一半。", "2026-09-28", "e1")[0]
        # 把"现在"推到很远的将来，噪音分数必然跌破阈值
        decisions = apply_retention(store, "2027-09-28")
        self.assertEqual(noise.status, "discarded")
        self.assertEqual(len(decisions), 1)


class TestRetrieval(unittest.TestCase):
    """检索层：路由、同义词扩展、档案层"""

    def setUp(self):
        self.store = _tmp_store()
        self.retriever = Retriever(self.store)
        agent = MemoryAgent(self.store, with_reconciler=True)
        for d, t in [("2026-05-20", "跑步计划正式放弃，先摆烂一段时间。"),
                     ("2026-09-01", "开学。重新开始跑步计划，这次目标五公里。"),
                     ("2026-04-10", "周三羽毛球，打了两小时。"),
                     ("2026-04-12", "又熬夜了，刷手机到两点。")]:
            agent.ingest(d, t)
        agent.maintain()

    def test_route(self):
        self.assertEqual(self.retriever.route("我什么时候开始跑步的"), "time+lex")
        self.assertEqual(self.retriever.route("总结一下我的作息习惯怎么样"), "lex+graph")
        self.assertEqual(self.retriever.route("我现在还在跑步吗"), "lex+time")

    def test_synonym_expansion(self):
        toks = expand_query(tokenize("我有哪些兴趣爱好？"))
        self.assertIn("跑步", toks)
        self.assertIn("羽毛球", toks)
        # 扩展后"爱好"类泛化查询能命中具体记忆
        hits = self.retriever.search("我有哪些兴趣爱好？")
        self.assertTrue(hits)

    def test_time_query_reaches_archive(self):
        """被取代的'放弃跑步'已进档案层，时间回溯问题仍能查到。"""
        out = MemoryAgent(self.store).answer("我什么时候放弃过跑步？")
        self.assertIn("05-20", out["answer"])

    def test_two_level_fallback(self):
        """活跃层查不到的问题，兜底翻档案层。"""
        out = MemoryAgent(self.store).answer("我是不是已经放弃了跑步计划？")
        self.assertNotIn("还没有相关的记录", out["answer"])


class TestScoring(unittest.TestCase):
    """评测判分：矛盾题从严"""

    def test_contradiction_strict_stale_penalty(self):
        q = {"type": "contradiction", "keywords": ["轮流"], "stale_keywords": ["每周末"]}
        self.assertEqual(score_question(q, "现在是大扫除轮流进行，此前是每周末。"), 0.5)
        self.assertEqual(score_question(q, "现在是大扫除轮流进行。"), 1.0)
        self.assertEqual(score_question(q, "大扫除每周末。"), 0.0)

    def test_normal_scoring(self):
        q = {"type": "recent", "keywords": ["浅烘", "手冲"]}
        self.assertEqual(score_question(q, "浅烘和手冲都不错"), 1.0)
        self.assertEqual(score_question(q, "只有浅烘"), 0.5)


if __name__ == "__main__":
    unittest.main()
