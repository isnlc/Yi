import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["MEMAGENT_LLM"] = "0"

from memagent.agents import MemoryAgent
from memagent.interaction import classify_interaction
from memagent.router import (
    classify_query,
    extract_record_content,
    is_forget_command,
    is_record_command,
)
from memagent.retrieve import calendar_spans, is_temporal_activity_query
from memagent.memory_policy import implicit_memory_candidate, sensitive_reason
from memagent.store import Conflict, MemoryStore
from tools.query_tool import query_memory
from web.app import (
    ChatRequest,
    SESSION,
    _context_memory_relevance,
    _handle_forget,
    _handle_preference_question,
    _is_general_query,
    _normalize_preference_topics,
    _review_window,
    chat_api,
)


class TestUnifiedQueryRouter(unittest.TestCase):
    def test_explicit_commands_use_positive_intent_not_keyword_fragments(self):
        cases = [
            ("不要记住这些附件，只分析", False, False),
            ("别记录这个，只回答", False, False),
            ("不要忘记明天开会", True, False),
            ("别忘了明天开会", True, False),
            ("帮我记录项目周五答辩", True, False),
            ("帮我删掉跑步计划", False, True),
            ("帮我忘记跑步计划", False, True),
            ("从记忆库移除跑步计划", False, True),
            ("你能记住多少内容", False, False),
        ]
        for question, record, forget in cases:
            with self.subTest(question=question):
                self.assertEqual(is_record_command(question), record)
                self.assertEqual(is_forget_command(question), forget)

        self.assertEqual(extract_record_content("帮我记住，项目周五答辩"), "项目周五答辩")
        self.assertEqual(extract_record_content("帮我记录项目周五答辩"), "项目周五答辩")
        self.assertEqual(extract_record_content("不要忘记明天开会"), "明天开会")

    def test_llm_semantic_router_is_primary_for_complete_questions(self):
        cases = {
            "这和我原来的安排一致吗": ("memory", "llm"),
            "按我的实际情况看，这个调整合适吗": ("memory", "rule_fast"),
            "这种做法合理吗": ("general", "llm"),
            "这个概念怎么理解": ("general", "llm"),
        }
        with patch("memagent.llm.is_enabled", return_value=True), \
                patch("memagent.llm.chat_json") as chat_json:
            for question, (intent, method) in cases.items():
                chat_json.return_value = {
                    "intent": intent,
                    "reason": "测试语义判定",
                    "confidence": 0.98,
                }
                result = classify_query(question)
                self.assertEqual(result["intent"], intent, question)
                self.assertEqual(result["method"], method, question)

    def test_llm_failure_uses_offline_fallback(self):
        with patch("memagent.llm.is_enabled", return_value=True), \
                patch("memagent.llm.chat_json", side_effect=TimeoutError):
            self.assertEqual(classify_query("我今天吃了什么")["intent"], "memory")
            self.assertEqual(classify_query("如何制定跑步计划")["intent"], "general")
            self.assertEqual(
                classify_query("根据我的记录，我该怎么改善睡眠")["intent"],
                "memory",
            )
    def test_routes_multiple_query_types_by_dependency(self):
        memory_queries = [
            "我现在喝什么咖啡？",
            "我国庆做了什么？",
            "我十一月做了什么？",
            "我2025年12月做了什么？",
            "我的课程项目最后是什么？",
            "我最近在忙什么？",
            "我的兴趣爱好有哪些？",
            "我今天吃了什么？",
            "我最近一次跑步是什么时候？",
            "根据我的记录，我该怎么改善睡眠？",
        ]
        general_queries = [
            "用一句话讲讲什么是黑洞",
            "我该怎么改善睡眠？",
            "推荐两本适合睡前读的书",
            "上周天气怎么样？",
            "帮我写一句春天的话",
            "麻辣烫有哪些口味？",
            "咖啡有哪些烘焙程度？",
            "路由器分为哪几类？",
        ]
        with patch("memagent.llm.is_enabled", return_value=False):
            for question in memory_queries:
                self.assertEqual(classify_query(question)["intent"], "memory", question)
            for question in general_queries:
                self.assertEqual(classify_query(question)["intent"], "general", question)
            self.assertEqual(classify_query("帮我查查今天星期几了")["intent"], "utility")
            self.assertEqual(classify_query("帮我记一下周五交报告")["intent"], "record")
            self.assertEqual(classify_query("帮我忘掉跑步计划")["intent"], "forget")
            self.assertEqual(classify_query("我上周做了什么")["intent"], "review")

    def test_short_followup_inherits_previous_scope(self):
        with patch("memagent.llm.is_enabled", return_value=False):
            self.assertEqual(
                classify_query("那第二天呢？", previous_scope="memory")["intent"],
                "memory",
            )
            self.assertEqual(
                classify_query("再简单一点", previous_scope="general")["intent"],
                "general",
            )
            self.assertEqual(
                classify_query("我现在喝什么咖啡？", previous_scope="general")["intent"],
                "memory",
            )
            self.assertEqual(
                classify_query("用一句话讲讲什么是黑洞", previous_scope="memory")["intent"],
                "general",
            )

    def test_catalog_questions_are_general_even_when_memory_matches(self):
        questions = [
            "麻辣烫有哪些口味？",
            "咖啡有哪些烘焙程度？",
            "相机镜头分为哪几类？",
            "PDF 有哪些常见用途？",
        ]
        with patch("memagent.llm.is_enabled", return_value=False):
            for question in questions:
                with self.subTest(question=question):
                    result = classify_query(question, memory_probe=lambda: True)
                    self.assertEqual(result["intent"], "general")
                    self.assertEqual(result["method"], "rule_fast")

    def test_personalized_advice_uses_memory_but_plain_advice_does_not(self):
        with patch("memagent.llm.is_enabled", return_value=False):
            personalized = classify_query("按我的口味推荐麻辣烫")
            plain = classify_query("推荐一种麻辣烫口味")
            preference = classify_query("我喜欢什么麻辣烫口味？")

        self.assertEqual(personalized["intent"], "memory")
        self.assertEqual(personalized["answer_mode"], "retrieval_llm")
        self.assertTrue(personalized["requires_llm"])
        self.assertEqual(plain["intent"], "general")
        self.assertEqual(plain["answer_mode"], "general_llm")
        self.assertEqual(preference["intent"], "memory")
        self.assertEqual(preference["answer_mode"], "retrieval_only")
        self.assertFalse(preference["requires_llm"])

    def test_broad_personalized_advice_retrieves_same_topic_memories(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = MemoryStore(str(Path(temporary.name) / "store.json"))
        running = store.add_memory(
            date="2026-09-01",
            topic="健身",
            action="开始",
            detail="重新开始跑步计划，目标五公里。",
        )
        store.add_memory(
            date="2026-10-01",
            topic="饮食",
            action="偏好",
            detail="最近更喜欢浅烘咖啡。",
        )

        with patch("memagent.llm.is_enabled", return_value=False):
            result = query_memory(
                MemoryAgent(store),
                "按我的实际情况，推荐一种适合的运动安排",
                answer_mode="retrieval_llm",
            )

        self.assertTrue(any(running.detail in line for line in result["evidence"]))
        self.assertFalse(any("咖啡" in line for line in result["evidence"]))

    def test_memory_answer_mode_separates_lookup_from_reasoning(self):
        lookup_questions = [
            "我现在喝什么咖啡？",
            "我最近一次跑步是什么时候？",
            "我的兴趣爱好有哪些？",
            "我的课程项目最后是什么？",
        ]
        reasoning_questions = [
            "根据我的记录分析作息问题",
            "比较我的跑步和游泳安排哪个更合理",
            "总结我最近的学习状态",
            "按我的口味推荐麻辣烫",
        ]
        with patch("memagent.llm.is_enabled", return_value=False):
            for question in lookup_questions:
                with self.subTest(question=question):
                    result = classify_query(question, memory_probe=lambda: True)
                    self.assertEqual(result["answer_mode"], "retrieval_only")
                    self.assertFalse(result["requires_llm"])
            for question in reasoning_questions:
                with self.subTest(question=question):
                    result = classify_query(question, memory_probe=lambda: True)
                    self.assertIn(result["intent"], ("memory", "review"))
                    self.assertEqual(result["answer_mode"], "retrieval_llm")
                    self.assertTrue(result["requires_llm"])

    def test_omitted_possessive_current_lookup_skips_online_router(self):
        with patch("memagent.llm.is_enabled", return_value=True), \
                patch("memagent.llm.chat_json") as chat_json:
            result = classify_query("现在的定投是多少")

        chat_json.assert_not_called()
        self.assertEqual(result["intent"], "memory")
        self.assertEqual(result["method"], "rule_fast")
        self.assertEqual(result["answer_mode"], "retrieval_only")

    def test_omitted_possessive_evolution_uses_matching_memory_evidence(self):
        with patch("memagent.llm.is_enabled", return_value=False):
            result = classify_query(
                "陶艺课的安排是怎么变化的？",
                memory_probe=lambda: True,
            )
            general = classify_query(
                "汽车市场价格是怎么变化的？",
                memory_probe=lambda: False,
            )

        self.assertEqual(result["intent"], "memory")
        self.assertEqual(result["semantic_scope"], "evolution")
        self.assertEqual(result["answer_mode"], "retrieval_only")
        self.assertEqual(general["intent"], "general")

class TestWebIntentRouting(unittest.TestCase):
    def setUp(self):
        SESSION["history"] = []

    def test_general_question_skips_memory_pipeline_offline(self):
        self.assertTrue(_is_general_query("用一句话讲讲什么是黑洞"))
        self.assertFalse(_is_general_query("我现在喝什么咖啡？"))
        with patch("web.app.get_pipeline") as get_pipeline, \
                patch("web.app.is_enabled", return_value=False), \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="用一句话讲讲什么是黑洞"))
        get_pipeline.assert_not_called()
        self.assertEqual(result["type"], "general")
        self.assertEqual(result["scope"], "general")
        self.assertEqual(result["evidence"], [])
        self.assertNotIn("离线", result["answer"])
        self.assertNotIn("模型", result["answer"])

    def test_transient_personal_state_is_conversation_not_memory_lookup(self):
        with patch("web.app.get_pipeline") as get_pipeline, \
                patch("web.app.is_enabled", return_value=False), \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="我好困啊"))

        get_pipeline.assert_not_called()
        self.assertEqual(result["scope"], "general")
        self.assertEqual(result["type"], "general")
        self.assertIn("休息", result["answer"])
        self.assertNotIn("记忆库", result["answer"])
        self.assertNotIn("离线", result["answer"])

    def test_unpunctuated_questions_never_become_memory_writes(self):
        questions = [
            "我每月定投多少钱",
            "我的鱼缸多久换一次水",
            "我一周跑几次",
            "我的课程项目是什么",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(classify_interaction(question).kind, "none")
                self.assertEqual(implicit_memory_candidate(question), (False, ""))

    def test_date_utility_answers_without_memory_pipeline(self):
        with patch("web.app.get_pipeline") as get_pipeline, \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="帮我查查今天星期几了"))
        get_pipeline.assert_not_called()
        self.assertEqual(result["type"], "utility")
        self.assertEqual(result["scope"], "general")
        self.assertIn("星期", result["answer"])
        self.assertEqual(result["evidence"], [])

    def test_clear_long_term_plan_is_recorded_without_confirmation(self):
        pipeline = SimpleNamespace(store=object())
        with patch("web.app.record_interaction", return_value={
            "effects": ["已更新长期计划"], "episode": None, "memory_ids": ["m1"]
        }) as record, \
                patch("web.app.get_pipeline", return_value=pipeline), \
                patch("web.app.append_event"), \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="我以后每天跑步，计划坚持三个月"))

        record.assert_called_once()
        self.assertEqual(result["type"], "long_term_plan")
        self.assertEqual(result["memory_effects"], ["已更新长期计划"])

    def test_repeated_decision_keeps_reply_short_and_uses_status_tags(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = MemoryStore(str(Path(temporary.name) / "store.json"))
        first_episode = store.add_memory(
            date="2026-10-20",
            topic="饮食",
            action="记录",
            detail="我今天想吃麻辣烫。",
            memory_type="event",
        )
        store.add_memory(
            date="2026-10-20",
            topic="饮食",
            action="选择",
            detail="关于麻辣烫，本次选择微辣。",
            memory_type="decision",
            parent_id=first_episode.id,
            preference_key="微辣",
        )
        current_episode = store.add_memory(
            date="2026-11-02",
            topic="饮食",
            action="记录",
            detail="我今天又想吃麻辣烫。",
            memory_type="event",
        )
        def ingest(_records):
            store.add_memory(
                date="2026-11-02",
                topic="饮食",
                action="选择",
                detail="关于麻辣烫，本次选择微辣。",
                memory_type="decision",
                parent_id=current_episode.id,
                preference_key="微辣",
            )
            return {
                "memory_ids": ["m-decision"],
                "entry_ids": ["e-decision"],
            }

        pipeline = SimpleNamespace(store=store, ingest=ingest)
        SESSION["history"] = [{
            "q": "我今天又想吃麻辣烫",
            "a": "记下了。",
            "scope": "memory",
            "episode": {
                "memory_id": current_episode.id,
                "topic": "饮食",
                "subject": "麻辣烫",
                "date": "2026-11-02",
            },
        }]

        with patch("web.app.get_pipeline", return_value=pipeline), \
                patch("web.app._add_memory_candidate", return_value={"id": "p-test"}), \
                patch("web.app.append_event"):
            result = chat_api(ChatRequest(question="那就微辣吧"))

        self.assertEqual(result["answer"], "好，这次就定微辣。")
        self.assertIn("已记录本次选择", result["memory_effects"])
        self.assertIn("偏好待确认", result["memory_effects"])
        self.assertNotIn("确认前", result["answer"])
        self.assertNotIn("确认", result["answer"])

    def test_near_term_personal_plan_is_recorded_without_memory_lookup(self):
        pipeline = SimpleNamespace(store=object())
        with patch("web.app.record_interaction", return_value={
            "effects": ["已记入今天"], "episode": {
                "memory_id": "m2", "topic": "健身", "subject": "跑步", "date": "2026-11-02"
            }, "memory_ids": ["m2"]
        }) as record, \
                patch("web.app.get_pipeline", return_value=pipeline), \
                patch("web.app.append_event"), \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="我今天要去跑步"))

        record.assert_called_once()
        self.assertEqual(result["type"], "event")
        self.assertEqual(result["memory_effects"], ["已记入今天"])
        self.assertIn("今天要去跑步", result["answer"])
        self.assertNotIn("记忆库中没有", result["answer"])

    def test_uncertain_preference_still_uses_confirmation_queue(self):
        with patch("web.app._add_memory_candidate", return_value={
            "id": "p125", "detail": "我最近可能更喜欢清淡一点"
        }) as add_candidate, \
                patch("web.app.get_pipeline") as get_pipeline, \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="我最近可能更喜欢清淡一点"))

        add_candidate.assert_called_once()
        get_pipeline.assert_not_called()
        self.assertEqual(result["type"], "memory_candidate")

    def test_clear_love_preference_is_written_not_sent_to_general_chat(self):
        pipeline = SimpleNamespace(store=object())
        with patch("web.app.record_interaction", return_value={
            "effects": ["已记录明确偏好"],
            "episode": None,
            "memory_ids": ["m-pref"],
        }) as record, \
                patch("web.app.get_pipeline", return_value=pipeline), \
                patch("web.app.append_event"), \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="我爱蒜香"))

        record.assert_called_once()
        self.assertEqual(result["type"], "stable_preference")
        self.assertEqual(result["memory_effects"], ["已记录明确偏好"])

    def test_sensitive_content_is_blocked_from_explicit_memory_write(self):
        self.assertEqual(sensitive_reason("记住 API_KEY: abcdef123456"), "credential")
        with patch("web.app.get_pipeline") as get_pipeline, \
                patch("memagent.llm.is_enabled", return_value=False):
            result = chat_api(ChatRequest(question="帮我记住 API_KEY: abcdef123456"))

        get_pipeline.assert_not_called()
        self.assertTrue(result.get("write_blocked"))

    def test_low_value_and_general_text_do_not_create_candidates(self):
        self.assertEqual(implicit_memory_candidate("你好"), (False, ""))
        self.assertEqual(implicit_memory_candidate("巴黎是法国的首都。"), (False, ""))
        self.assertEqual(implicit_memory_candidate("我的 API_KEY: abcdef123456"), (False, ""))
        self.assertEqual(
            implicit_memory_candidate("我女儿以后每周去游泳"),
            (False, ""),
        )
        self.assertEqual(
            sensitive_reason("我正在服用某种处方药"),
            "health_information",
        )
        self.assertEqual(
            implicit_memory_candidate("我今天要去跑步")[0],
            True,
        )
        self.assertEqual(
            implicit_memory_candidate("我今天要去跑步吗？"),
            (False, ""),
        )
        self.assertIsNone(sensitive_reason("截图时间 2026-11-02 021856"))
        self.assertEqual(sensitive_reason("银行卡号 4111 1111 1111 1111"), "payment_card")


class TestMemoryOperations(unittest.TestCase):
    def setUp(self):
        SESSION["history"] = []

    def _store(self):
        tmp = tempfile.TemporaryDirectory()
        store = MemoryStore(str(Path(tmp.name) / "store.json"))
        self.addCleanup(tmp.cleanup)
        return store

    def test_today_activity_query_uses_date_and_topic(self):
        store = self._store()
        today = date.today().isoformat()
        food = store.add_memory(date=today, topic="饮食", action="记录",
                                detail="午饭吃了牛肉面。")
        course = store.add_memory(date=today, topic="课程", action="记录",
                                  detail="下午整理了课程实验。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = query_memory(
                MemoryAgent(store, reference_date=today),
                "我今天吃了什么",
            )

        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn(food.detail, result["answer"])
        self.assertNotIn(course.detail, result["answer"])

    def test_today_activity_query_without_record_returns_no_evidence(self):
        store = self._store()
        with patch("memagent.llm.is_enabled", return_value=False):
            result = query_memory(
                MemoryAgent(store, reference_date=date.today()),
                "我今天吃了什么",
            )

        self.assertEqual(result["evidence"], [])
        self.assertNotIn("缺少可检索的主题", result["answer"])
        self.assertIn("没有找到", result["answer"])
    def test_forget_only_affects_directly_related_memories(self):
        store = self._store()
        running_a = store.add_memory(date="2026-09-01", topic="健身", action="开始",
                                     detail="重新开始跑步计划，目标五公里。")
        running_b = store.add_memory(date="2026-09-24", topic="健身", action="完成",
                                     detail="跑步第二周，已经能跑完五公里。")
        milk_tea = store.add_memory(date="2026-10-03", topic="饮食", action="偏好",
                                   detail="放弃戒奶茶的计划，重新喝奶茶。")
        pipeline = SimpleNamespace(store=store, agent=MemoryAgent(store))

        with patch("web.app.get_pipeline", return_value=pipeline), \
                patch("web.app.append_event"):
            result = _handle_forget("帮我忘掉跑步计划")

        self.assertEqual(result["count"], 2)
        self.assertEqual(running_a.status, "discarded")
        self.assertEqual(running_b.status, "discarded")
        self.assertEqual(milk_tea.status, "active")
        self.assertIn("重新开始跑步计划", result["answer"])
        self.assertIn("跑步第二周", result["answer"])
        self.assertIn("可在记忆库中恢复", result["answer"])
        self.assertNotIn("奶茶", result["answer"])

    def test_forget_uses_recent_episode_even_after_legacy_compression(self):
        store = self._store()
        running = store.add_memory(
            date="2026-11-02",
            topic="健身",
            action="记录",
            detail="我今天要去跑步。",
            status="compressed",
            memory_type="event",
        )
        milk_tea = store.add_memory(
            date="2026-11-02",
            topic="饮食",
            action="偏好",
            detail="我喜欢少糖奶茶。",
        )
        SESSION["history"] = [{
            "q": "我今天要去跑步",
            "a": "记下了。",
            "scope": "memory",
            "episode": {
                "memory_id": running.id,
                "topic": "健身",
                "subject": "跑步",
                "date": "2026-11-02",
            },
        }]
        pipeline = SimpleNamespace(store=store, agent=MemoryAgent(store))

        with patch("web.app.get_pipeline", return_value=pipeline), \
                patch("web.app.append_event"):
            result = _handle_forget("帮我忘掉跑步计划")

        self.assertEqual(result["count"], 1)
        self.assertEqual(running.status, "discarded")
        self.assertEqual(milk_tea.status, "active")
        self.assertIn("我今天要去跑步", result["answer"])
        self.assertNotIn("直接相关的 1 条", result["answer"])

    def test_calendar_query_only_returns_in_window_activities(self):
        store = self._store()
        trip = store.add_memory(date="2026-10-02", topic="旅行", action="完成",
                                detail="完成了泉州之行，参观开元寺。")
        coffee = store.add_memory(date="2026-10-08", topic="饮食", action="偏好",
                                  detail="试过深烘后，还是更喜欢浅烘。")
        meta = store.add_memory(date="2026-10-05", topic="生活", action="记录",
                                detail="用户询问上个月定投金额改成了多少。")
        outside = store.add_memory(date="2026-09-20", topic="生活", action="更改",
                                   detail="恢复定投并改为每月300元。", status="superseded")
        store.conflicts.append(Conflict(
            id="c1", topic="生活", action="记录",
            older_id=outside.id, newer_id=meta.id,
            resolved=True, winner_id=meta.id,
        ))

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer("我国庆做了什么？")

        self.assertEqual(len(result["evidence"]), 2)
        self.assertTrue(all("2026-10-0" in line for line in result["evidence"]))
        self.assertIn(trip.detail, result["answer"])
        self.assertIn(coffee.detail, result["answer"])
        self.assertNotIn(meta.detail, result["answer"])
        self.assertNotIn(outside.detail, result["answer"])
        self.assertNotIn("前后不同", result["answer"])

    def test_calendar_windows_are_not_bound_to_october_2026(self):
        reference = date(2027, 3, 18)
        self.assertEqual(
            calendar_spans("我二月做了什么", reference),
            [("2027-02-01", "2027-02-28")],
        )
        self.assertEqual(
            calendar_spans("回顾一下上个月", reference),
            [("2027-02-01", "2027-02-28")],
        )
        self.assertEqual(
            calendar_spans("回顾最近10天", reference),
            [("2027-03-09", "2027-03-18")],
        )
        self.assertEqual(
            calendar_spans("2025年12月做了什么", reference),
            [("2025-12-01", "2025-12-31")],
        )
        self.assertEqual(
            calendar_spans("我3月5日做了什么", reference),
            [("2027-03-05", "2027-03-05")],
        )
        self.assertEqual(
            calendar_spans("我4月1日做了什么", reference),
            [("2026-04-01", "2026-04-01")],
        )

    def test_broad_recent_activity_queries_use_a_default_seven_day_window(self):
        reference = date(2026, 10, 15)
        broad_queries = [
            "我最近做了什么？",
            "最近都干了什么？",
            "我最近在忙什么？",
            "最近发生了什么？",
            "这几天我做过什么？",
            "回顾一下最近的记录",
            "我近期有什么活动？",
            "最近有什么值得注意的事？",
            "近来都忙啥？",
            "最近一阵子有啥事情？",
        ]
        for query in broad_queries:
            with self.subTest(query=query):
                self.assertTrue(is_temporal_activity_query(query))
                self.assertEqual(
                    calendar_spans(query, reference),
                    [("2026-10-09", "2026-10-15")],
                )
                route = classify_query(query)
                if query.startswith("回顾"):
                    self.assertEqual(route["intent"], "review")
                else:
                    self.assertEqual(route["intent"], "memory")
                    self.assertEqual(route["semantic_scope"], "history")

        self.assertEqual(
            calendar_spans("回顾近三天的记录", reference),
            [("2026-10-13", "2026-10-15")],
        )

    def test_recent_activity_query_keeps_multiple_window_records(self):
        store = self._store()
        included = [
            store.add_memory(
                date="2026-10-09",
                topic="课程",
                action="完成",
                detail="完成了课程项目的数据整理。",
            ),
            store.add_memory(
                date="2026-10-12",
                topic="运动",
                action="记录",
                detail="晚上跑步五公里。",
            ),
            store.add_memory(
                date="2026-10-14",
                topic="饮食",
                action="记录",
                detail="和室友去吃了麻辣烫。",
            ),
        ]
        outside = store.add_memory(
            date="2026-09-20",
            topic="旅行",
            action="完成",
            detail="完成了苏州旅行。",
        )

        result = query_memory(
            MemoryAgent(store, reference_date="2026-10-15"),
            "我最近做了什么？",
            answer_mode="retrieval_only",
        )

        self.assertEqual(len(result["evidence"]), len(included))
        for memory in included:
            self.assertIn(memory.detail, result["answer"])
        self.assertNotIn(outside.detail, result["answer"])
        self.assertNotIn("缺少可检索的主题", result["answer"])

    def test_recent_activity_falls_back_to_latest_record_window_when_today_is_empty(self):
        store = self._store()
        included = [
            store.add_memory(
                date="2026-10-12",
                topic="运动",
                action="记录",
                detail="晚上跑步五公里。",
            ),
            store.add_memory(
                date="2026-10-15",
                topic="课程",
                action="完成",
                detail="完成了课程展示材料。",
            ),
        ]
        outside = store.add_memory(
            date="2026-10-01",
            topic="旅行",
            action="完成",
            detail="整理了国庆旅行行李。",
        )

        result = query_memory(
            MemoryAgent(store, reference_date="2026-11-02"),
            "我最近做了什么？",
            answer_mode="retrieval_only",
        )

        self.assertEqual(len(result["evidence"]), len(included))
        for memory in included:
            self.assertIn(memory.detail, result["answer"])
        self.assertNotIn(outside.detail, result["answer"])
        self.assertIn("当前最近七天没有相关记录", result["answer"])
        self.assertIn("2026-10-09 至 2026-10-15", result["answer"])

    def test_recent_time_boundaries_preserve_current_latest_and_topic_queries(self):
        reference = date(2026, 10, 15)
        current_and_latest = [
            "我最近一次跑步是什么时候？",
            "我现在还在坚持跑步吗？",
            "我最近喝什么咖啡？",
            "我近来喝什么咖啡？",
        ]
        for query in current_and_latest:
            with self.subTest(query=query):
                self.assertFalse(is_temporal_activity_query(query))
                self.assertEqual(calendar_spans(query, reference), [])

        store = self._store()
        coffee = store.add_memory(
            date="2026-10-14",
            topic="饮食",
            action="记录",
            detail="尝试了一杯浅烘咖啡。",
        )
        unrelated = store.add_memory(
            date="2026-10-13",
            topic="饮食",
            action="记录",
            detail="和室友去吃了麻辣烫。",
        )
        old_coffee = store.add_memory(
            date="2026-09-20",
            topic="饮食",
            action="记录",
            detail="喝了一杯深烘咖啡。",
        )

        result = query_memory(
            MemoryAgent(store, reference_date=reference),
            "最近有哪些咖啡记录？",
            answer_mode="retrieval_only",
        )

        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn(coffee.detail, result["answer"])
        self.assertNotIn(unrelated.detail, result["answer"])
        self.assertNotIn(old_coffee.detail, result["answer"])

    def test_named_period_plan_query_does_not_filter_by_record_date(self):
        reference = date(2026, 11, 2)
        self.assertEqual(
            calendar_spans("国庆我们最后决定去哪儿玩？", reference),
            [],
        )
        self.assertEqual(
            calendar_spans("国庆期间我做了什么？", reference),
            [("2026-10-01", "2026-10-08")],
        )

    def test_review_window_reuses_the_shared_calendar_parser(self):
        reference = date(2027, 3, 18)
        self.assertEqual(
            _review_window("回顾一下二月", reference),
            (date(2027, 2, 1), date(2027, 2, 28)),
        )
        self.assertEqual(
            _review_window("回顾一下本月", reference),
            (date(2027, 3, 1), date(2027, 3, 18)),
        )


class TestContextualPreferenceRelevance(unittest.TestCase):
    def _store(self):
        tmp = tempfile.TemporaryDirectory()
        store = MemoryStore(str(Path(tmp.name) / "store.json"))
        self.addCleanup(tmp.cleanup)
        return store

    @staticmethod
    def _plan():
        return SimpleNamespace(
            subject="麻辣烫",
            topic="饮食",
            query_text="我今天想吃麻辣烫，你说我吃什么口味",
            record_text="我今天想吃麻辣烫",
        )

    def test_object_specific_coffee_preference_does_not_leak_to_malatang(self):
        store = self._store()
        coffee = store.add_memory(
            date="2026-10-08",
            topic="饮食",
            action="偏好",
            detail="我一直喜欢浅烘咖啡。",
            memory_type="long_term",
            preference_key="浅烘咖啡",
        )

        self.assertEqual(
            _context_memory_relevance(store, coffee, self._plan()),
            0,
        )

    def test_object_specific_milk_tea_sweetness_does_not_leak_to_malatang(self):
        store = self._store()
        milk_tea = store.add_memory(
            date="2026-10-09",
            topic="饮食",
            action="偏好",
            detail="我平时喜欢少糖奶茶。",
            memory_type="long_term",
            preference_key="少糖奶茶",
        )

        self.assertEqual(
            _context_memory_relevance(store, milk_tea, self._plan()),
            0,
        )

    def test_general_spice_constraint_can_inform_malatang(self):
        store = self._store()
        spice = store.add_memory(
            date="2026-10-10",
            topic="饮食",
            action="偏好",
            detail="我不能吃辣。",
            memory_type="long_term",
            preference_key="不能吃辣",
        )

        self.assertGreater(
            _context_memory_relevance(store, spice, self._plan()),
            0,
        )

    def test_missing_subject_does_not_fall_back_to_all_food_preferences(self):
        store = self._store()
        coffee = store.add_memory(
            date="2026-10-08",
            topic="饮食",
            action="偏好",
            detail="我一直喜欢浅烘咖啡。",
            memory_type="long_term",
            preference_key="浅烘咖啡",
        )
        plan = SimpleNamespace(
            subject="",
            topic="饮食",
            query_text="我今天想吃点东西，你说选什么",
            record_text="我今天想吃点东西",
        )

        self.assertEqual(
            _context_memory_relevance(store, coffee, plan),
            0,
        )

    def test_same_object_decision_is_the_strongest_evidence(self):
        store = self._store()
        episode = store.add_memory(
            date="2026-10-11",
            topic="饮食",
            action="记录",
            detail="我今天想吃麻辣烫。",
            memory_type="event",
        )
        decision = store.add_memory(
            date="2026-10-11",
            topic="饮食",
            action="选择",
            detail="关于麻辣烫，本次选择微辣。",
            memory_type="decision",
            parent_id=episode.id,
            preference_key="微辣",
        )

        self.assertEqual(
            _context_memory_relevance(store, decision, self._plan()),
            100,
        )


class TestPreferenceQueryScoping(unittest.TestCase):
    def _store(self):
        tmp = tempfile.TemporaryDirectory()
        store = MemoryStore(str(Path(tmp.name) / "store.json"))
        self.addCleanup(tmp.cleanup)
        return store

    def _pipeline(self):
        store = self._store()
        store.add_memory(
            date="2026-11-02",
            topic="生活",
            action="偏好",
            detail="我爱蒜香。",
            memory_type="long_term",
            preference_key="蒜香",
        )
        store.add_memory(
            date="2026-06-05",
            topic="饮食",
            action="偏好",
            detail="浅烘咖啡豆到货，发现自己更喜欢浅烘的风味。",
            memory_type="long_term",
        )
        store.add_memory(
            date="2026-10-08",
            topic="饮食",
            action="偏好",
            detail="试了深烘豆子后，还是觉得浅烘更合口味。",
            memory_type="long_term",
        )
        return SimpleNamespace(store=store)

    def test_general_flavor_query_excludes_object_specific_coffee_preferences(self):
        with patch("web.app.get_pipeline", return_value=self._pipeline()):
            result = _handle_preference_question("饮食", "我喜欢什么口味？")

        self.assertIn("蒜香", result["answer"])
        self.assertNotIn("浅烘", result["answer"])
        self.assertEqual(len(result["evidence"]), 1)

    def test_object_query_returns_only_that_object_and_deduplicates_values(self):
        with patch("web.app.get_pipeline", return_value=self._pipeline()):
            result = _handle_preference_question("饮食", "我喜欢什么咖啡？")

        self.assertIn("浅烘", result["answer"])
        self.assertNotIn("蒜香", result["answer"])
        self.assertEqual(len(result["evidence"]), 1)

    def test_broad_diet_query_can_include_both_scopes(self):
        with patch("web.app.get_pipeline", return_value=self._pipeline()):
            result = _handle_preference_question("饮食", "我有什么饮食偏好？")

        self.assertIn("蒜香", result["answer"])
        self.assertIn("浅烘", result["answer"])
        self.assertEqual(len(result["evidence"]), 2)

    def test_legacy_diet_preference_topic_is_normalized(self):
        store = self._store()
        preference = store.add_memory(
            date="2026-11-02",
            topic="生活",
            action="偏好",
            detail="我爱蒜香。",
            memory_type="long_term",
            preference_key="蒜香",
        )

        _normalize_preference_topics(store)

        self.assertEqual(preference.topic, "饮食")


class TestCurrentStateAnswers(unittest.TestCase):
    def _store(self):
        tmp = tempfile.TemporaryDirectory()
        store = MemoryStore(str(Path(tmp.name) / "store.json"))
        self.addCleanup(tmp.cleanup)
        return store

    def test_fast_online_mode_skips_planner_and_auditor_calls(self):
        store = self._store()
        store.add_memory(date="2026-10-08", topic="饮食", action="偏好",
                         detail="试了深烘豆子后，还是觉得浅烘更合口味。")
        agent = MemoryAgent(store)

        with patch.dict("os.environ", {"MEMAGENT_DEEP_REASONING": "0"}), \
                patch("memagent.llm.is_enabled", return_value=True), \
                patch("memagent.llm.plan_route") as plan_route, \
                patch("memagent.llm.audit_evidence") as audit_evidence, \
                patch("memagent.llm.chat", return_value="当前偏好浅烘。"):
            result = agent.answer("我现在喝什么咖啡？")

        plan_route.assert_not_called()
        audit_evidence.assert_not_called()
        self.assertIn("浅烘", result["answer"])
        self.assertEqual(result["trace"][0]["method"], "rule")

    def test_retrieval_only_never_calls_llm(self):
        store = self._store()
        store.add_memory(
            date="2026-10-08",
            topic="饮食",
            action="偏好",
            detail="我目前偏好浅烘咖啡。",
        )
        agent = MemoryAgent(store)

        with patch("memagent.llm.is_enabled", return_value=True), \
                patch("memagent.llm.chat") as chat, \
                patch("memagent.llm.chat_json") as chat_json:
            result = agent.answer(
                "我现在喝什么咖啡？",
                answer_mode="retrieval_only",
            )

        chat.assert_not_called()
        chat_json.assert_not_called()
        self.assertFalse(result["llm_used"])
        self.assertEqual(result["answer_mode"], "retrieval_only")
        self.assertIn("浅烘", result["answer"])

    def test_current_amount_lookup_returns_latest_active_version(self):
        store = self._store()
        store.add_memory(
            date="2026-04-01",
            topic="理财",
            action="开始",
            detail="今天开始基金定投，每月500元。",
            status="superseded",
        )
        store.add_memory(
            date="2026-09-20",
            topic="理财",
            action="更改",
            detail="恢复定投，改成每月300元。",
            status="superseded",
        )
        store.add_memory(
            date="2026-11-03",
            topic="理财",
            action="更改",
            detail="定投每月改成500元。",
        )

        with patch("memagent.llm.is_enabled", return_value=False):
            agent = MemoryAgent(store)
            for question in (
                "我的定投现在是多少？",
                "现在的定投是多少？",
                "我每月定投多少钱？",
            ):
                with self.subTest(question=question):
                    result = query_memory(
                        agent,
                        question,
                        answer_mode="retrieval_only",
                    )
                    self.assertIn("500元", result["answer"])
                    self.assertNotIn("300元", result["answer"])

    def test_current_property_lookup_considers_later_stop_state(self):
        store = self._store()
        store.add_memory(
            date="2026-04-01",
            topic="理财",
            action="开始",
            detail="今天开始基金定投，每月500元。",
            status="superseded",
        )
        store.add_memory(
            date="2026-08-10",
            topic="理财",
            action="放弃",
            detail="我停止了基金定投，先观望一阵。",
        )

        with patch("memagent.llm.is_enabled", return_value=False):
            result = query_memory(
                MemoryAgent(store),
                "我的定投现在是多少？",
                answer_mode="retrieval_only",
            )

        self.assertIn("停止", result["answer"])
        self.assertNotIn("每月500元", result["answer"])

    def test_retrieval_llm_receives_memory_evidence(self):
        store = self._store()
        store.add_memory(
            date="2026-10-08",
            topic="饮食",
            action="偏好",
            detail="我不能吃太辣。",
        )
        agent = MemoryAgent(store)

        with patch("memagent.llm.is_enabled", return_value=True), \
                patch("memagent.llm.chat", return_value="建议选择微辣。") as chat:
            result = agent.answer(
                "按我的口味推荐麻辣烫",
                answer_mode="retrieval_llm",
            )

        chat.assert_called_once()
        prompt = chat.call_args.args[0]
        self.assertIn("不能吃太辣", prompt[-1]["content"])
        self.assertTrue(result["llm_used"])
        self.assertEqual(result["answer_mode"], "retrieval_llm")
        self.assertEqual(result["answer"], "建议选择微辣。")
    def test_latest_event_selects_matching_record_not_later_preference(self):
        store = self._store()
        takeaway = store.add_memory(date="2026-09-19", topic="饮食", action="记录",
                                    detail="周末和室友一起点外卖，边吃边聊天。",
                                    status="compressed")
        coffee = store.add_memory(date="2026-10-08", topic="饮食", action="偏好",
                                  detail="试了深烘豆子后，还是觉得浅烘更合口味。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer("我最近一次吃的外卖是")

        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn(takeaway.detail, result["answer"])
        self.assertNotIn(coffee.detail, result["answer"])
    def test_coffee_answer_uses_latest_explicit_preference_only(self):
        store = self._store()
        store.add_memory(date="2026-10-13", topic="饮食", action="记录",
                         detail="查了不同咖啡豆产区的风味资料。")
        store.add_memory(date="2026-10-12", topic="饮食", action="记录",
                         detail="订了一袋新的咖啡豆，准备周末试冲。")
        store.add_memory(date="2026-10-08", topic="饮食", action="偏好",
                         detail="试了深烘豆子后，还是觉得浅烘更合口味。")
        store.add_memory(date="2026-10-07", topic="饮食", action="偏好",
                         detail="最近更喜欢少糖奶茶。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer("我现在喝什么咖啡？")

        self.assertIn("浅烘", result["answer"])
        self.assertNotIn("奶茶", result["answer"])
        self.assertNotIn("前后不同", result["answer"])
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("浅烘", result["evidence"][0])

    def test_true_conflict_keeps_latest_state_and_chain(self):
        store = self._store()
        agent = MemoryAgent(store)
        agent.ingest("2026-05-20", "跑步计划正式放弃。")
        agent.ingest("2026-09-01", "重新开始跑步计划，目标五公里。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = agent.answer("我现在还在跑步吗？")

        self.assertIn("重新开始", result["answer"])
        self.assertIn("状态有过更新", result["answer"])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertTrue(any("正式放弃" in line for line in result["evidence"]))
        self.assertTrue(any("重新开始" in line for line in result["evidence"]))


if __name__ == "__main__":
    unittest.main()

