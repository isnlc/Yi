import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memagent.agents import MemoryAgent
from memagent.router import classify_query
from memagent.semantics import (
    analyze_query,
    analyze_semantics,
    enrich_memory_semantics,
    frame_for_memory,
    semantic_match_score,
)
from memagent.store import MemoryStore
from web.app import (
    _context_memory_relevance,
    _handle_forget,
    _handle_preference_question,
)
import web.app as web_app


class SharedSemanticTestCase(unittest.TestCase):
    def make_store(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return MemoryStore(str(Path(temporary.name) / "store.json"))


class TestSemanticMatching(SharedSemanticTestCase):
    def test_equipment_terms_are_not_misclassified_as_locations(self):
        query = analyze_query("我写钢笔习惯用什么笔尖和墨水？", use_llm=False)
        memory = analyze_semantics(
            "我写钢笔时喜欢 EF 尖，常用蓝黑墨水。",
            topic_hint="生活",
        )

        self.assertNotIn("地点", query.facets)
        self.assertGreater(semantic_match_score(query, memory), 0)

    def test_location_facet_still_requires_real_spatial_context(self):
        query = analyze_query("我上周去了上海吗？", use_llm=False)

        self.assertIn("地点", query.facets)

    def test_equipment_type_and_configuration_facets_are_compatible(self):
        query = analyze_query(
            "现在打字用的那把键盘，手感是什么类型的？",
            use_llm=False,
        )
        memory = analyze_semantics(
            "机械键盘已经换成静音红轴。",
            topic_hint="生活",
        )

        self.assertIn("类型", query.facets)
        self.assertIn("配置", memory.facets)
        self.assertGreater(semantic_match_score(query, memory), 0)

    def test_money_questions_and_records_share_an_amount_facet(self):
        query = analyze_query("我每月定投多少钱？", use_llm=False)
        memory = analyze_semantics("定投每月改成500元。", topic_hint="理财")

        self.assertIn("金额", query.facets)
        self.assertIn("金额", memory.facets)
        self.assertGreater(semantic_match_score(query, memory), 0)

    def test_natural_interval_question_is_a_frequency_facet(self):
        query = analyze_query("我的鱼缸多久换一次水？", use_llm=False)
        memory = analyze_semantics("鱼缸每两周换一次水。")

        self.assertIn("频率", query.facets)
        self.assertIn("频率", memory.facets)
        self.assertGreater(semantic_match_score(query, memory), 0)

    def test_global_attribute_can_inform_a_concrete_object(self):
        query = analyze_semantics("汽车选什么颜色", is_query=True)
        preference = analyze_semantics("我喜欢蓝色")
        self.assertGreater(semantic_match_score(query, preference), 0)

    def test_object_specific_attribute_does_not_leak_to_another_object(self):
        query = analyze_semantics("衣服选什么颜色", is_query=True)
        preference = analyze_semantics("我的汽车喜欢黑色")
        self.assertEqual(semantic_match_score(query, preference), 0)

    def test_legacy_memory_is_enriched_without_rewriting_content(self):
        store = self.make_store()
        memory = store.add_memory(
            date="2026-11-02",
            topic="健身",
            action="记录",
            detail="我每周跑步三次。",
        )
        self.assertTrue(enrich_memory_semantics(store))
        self.assertIn("跑步", memory.entities)
        self.assertIn("频率", memory.facets)
        self.assertEqual(memory.detail, "我每周跑步三次。")


class TestCrossDomainRetrieval(SharedSemanticTestCase):
    def test_specific_object_query_never_uses_unrelated_lexical_fallback(self):
        store = self.make_store()
        store.add_memory(
            date="2026-06-11",
            topic="生活",
            action="记录",
            detail="最近多喝了水，含糖饮料少了一半。",
        )

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer(
                "我的鱼缸多久换一次水？",
                answer_mode="retrieval_only",
            )

        self.assertEqual(result["evidence"], [])
        self.assertNotIn("含糖饮料", result["answer"])

    def test_frequency_lookup_uses_the_matching_concrete_object(self):
        store = self.make_store()
        store.add_memory(
            date="2026-06-11",
            topic="生活",
            action="约定",
            detail="鱼缸每两周换一次水。",
        )
        store.add_memory(
            date="2026-06-12",
            topic="健身",
            action="约定",
            detail="我每周跑步三次。",
        )
        enrich_memory_semantics(store)

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer(
                "我的鱼缸多久换一次水？",
                answer_mode="retrieval_only",
            )

        self.assertIn("每两周", result["answer"])
        self.assertNotIn("跑步", result["answer"])

    def test_explicit_change_count_is_an_evolution_query(self):
        frame = analyze_semantics("我的咖啡口味经历了几次变化？", is_query=True)
        self.assertEqual(frame.scope, "evolution")

    def test_current_plan_uses_latest_state_change_not_latest_completion(self):
        store = self.make_store()
        agent = MemoryAgent(store)
        agent.ingest("2026-05-20", "跑步计划正式放弃。")
        agent.ingest("2026-09-01", "重新开始跑步计划，目标五公里。")
        agent.ingest("2026-10-14", "今天跑步按计划完成，膝盖没有不适。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = agent.answer(
                "我是不是已经放弃了跑步计划？",
                answer_mode="retrieval_only",
            )

        self.assertIn("重新开始跑步计划", result["answer"])
        self.assertNotIn("按计划完成", result["answer"])

    def test_current_arrangement_keeps_latest_semantic_candidate(self):
        store = self.make_store()
        agent = MemoryAgent(store)
        agent.ingest("2026-03-10", "和室友约定，每周日一起大扫除。")
        agent.ingest("2026-06-03", "大扫除改成本月我自己做。")
        agent.ingest("2026-09-05", "和室友约定改成轮流大扫除，一人一周。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = agent.answer(
                "我是不是还每周日自己大扫除？",
                answer_mode="retrieval_only",
            )

        self.assertIn("轮流大扫除", result["answer"])
        self.assertNotIn("本月我自己做", result["answer"])

    def test_evolution_query_keeps_method_and_preference_changes(self):
        store = self.make_store()
        agent = MemoryAgent(store)
        agent.ingest("2026-04-22", "咖啡喝完了，决定换成手冲，买了滤杯和磨豆机。")
        agent.ingest("2026-06-05", "浅烘的豆子到了，发现更喜欢浅烘的风味。")
        agent.ingest("2026-10-08", "深烘豆子试了试，还是浅烘合口味。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = agent.answer(
                "我的咖啡口味经历了几次变化？",
                answer_mode="retrieval_only",
            )

        self.assertIn("手冲", result["answer"])
        self.assertIn("浅烘", result["answer"])

    def test_current_configuration_recognizes_axis_type(self):
        store = self.make_store()
        old = store.add_memory(
            date="2026-06-01",
            topic="生活",
            action="更改",
            detail="机械键盘原来使用青轴。",
            status="superseded",
        )
        current = store.add_memory(
            date="2026-09-01",
            topic="生活",
            action="更改",
            detail="机械键盘已经换成静音红轴。",
        )
        enrich_memory_semantics(store)

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer(
                "机械键盘现在用的是什么轴？",
                answer_mode="retrieval_only",
            )

        self.assertIn("静音红轴", result["answer"])
        self.assertNotIn("青轴", result["answer"])
        self.assertIn("配置", current.facets)
        self.assertIn("配置", old.facets)

    def test_initial_state_query_excludes_later_replacement(self):
        store = self.make_store()
        store.add_memory(
            date="2026-04-02",
            topic="课程",
            action="更改",
            detail="课程项目最初选的是校园导航。",
            status="superseded",
        )
        store.add_memory(
            date="2026-05-12",
            topic="课程",
            action="更改",
            detail="课程项目后来改成了长期记忆智能体。",
        )
        enrich_memory_semantics(store)

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer(
                "课程项目一开始选的方向是什么？",
                answer_mode="retrieval_only",
            )

        self.assertIn("校园导航", result["answer"])
        self.assertNotIn("长期记忆智能体", result["answer"])

    def test_confirmation_value_is_a_hypothesis_not_a_required_match(self):
        store = self.make_store()
        store.add_memory(
            date="2026-10-01",
            topic="饮食",
            action="偏好",
            detail="我不能吃太辣，最多接受微辣。",
        )
        store.add_memory(
            date="2026-10-29",
            topic="饮食",
            action="偏好",
            detail="我不吃香菜。",
        )
        enrich_memory_semantics(store)

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer(
                "我吃火锅不是一直点特辣吗？",
                answer_mode="retrieval_only",
            )

        self.assertIn("微辣", result["answer"])
        self.assertNotIn("香菜", result["answer"])

    def test_omitted_entity_uses_matching_topic_attribute_and_value(self):
        store = self.make_store()
        store.add_memory(
            date="2026-10-05",
            topic="运动",
            action="约定",
            detail="我通常在周日上午去游泳。",
        )
        store.add_memory(
            date="2026-10-11",
            topic="生活",
            action="偏好",
            detail="我一直喜欢蓝色。",
        )
        store.add_memory(
            date="2026-10-27",
            topic="社交",
            action="约定",
            detail="我和室友约好每周日下午大扫除。",
        )
        enrich_memory_semantics(store)

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer(
                "周日我通常有什么运动安排？",
                answer_mode="retrieval_only",
            )

        self.assertIn("游泳", result["answer"])
        self.assertNotIn("蓝色", result["answer"])
        self.assertNotIn("大扫除", result["answer"])

    def test_frequency_query_uses_running_memory_not_coffee_memory(self):
        store = self.make_store()
        agent = MemoryAgent(store)
        agent.ingest("2026-10-01", "我每周跑步三次。")
        agent.ingest("2026-10-02", "我一直喜欢浅烘咖啡。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = agent.answer("我的跑步频率是什么？")

        self.assertIn("每周跑步三次", result["answer"])
        self.assertNotIn("浅烘咖啡", result["answer"])

    def test_attribute_query_reaches_matching_archived_memory(self):
        store = self.make_store()
        store.add_memory(
            date="2026-03-05",
            topic="运动",
            action="约定",
            detail="我每周跑步三次。",
            status="compressed",
        )
        store.add_memory(
            date="2026-05-20",
            topic="健身",
            action="放弃",
            detail="跑步计划正式放弃。",
        )

        with patch("memagent.llm.is_enabled", return_value=False):
            result = MemoryAgent(store).answer("我的跑步频率是什么？")

        self.assertIn("每周跑步三次", result["answer"])
        self.assertNotIn("正式放弃", result["answer"])

    def test_reading_time_query_normalizes_read_and_reading(self):
        store = self.make_store()
        agent = MemoryAgent(store)
        agent.ingest("2026-10-03", "我习惯晚上读书。")
        agent.ingest("2026-10-04", "我每周跑步三次。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = agent.answer("我通常什么时间阅读？")

        self.assertIn("晚上读书", result["answer"])
        self.assertNotIn("跑步", result["answer"])

    def test_project_query_uses_project_entity_instead_of_topic_only(self):
        store = self.make_store()
        agent = MemoryAgent(store)
        agent.ingest("2026-10-05", "课程项目改成记忆智能体。")
        agent.ingest("2026-10-06", "课程论文准备分析推荐系统。")

        with patch("memagent.llm.is_enabled", return_value=False):
            result = agent.answer("我的课程项目是什么？")

        self.assertIn("记忆智能体", result["answer"])
        self.assertNotIn("推荐系统", result["answer"])


class TestGenericPreferenceAndAdvice(SharedSemanticTestCase):
    def test_preference_question_supports_non_food_attributes(self):
        store = self.make_store()
        store.add_memory(
            date="2026-11-02",
            topic="生活",
            action="偏好",
            detail="我喜欢蓝色。",
            memory_type="long_term",
            preference_key="蓝色",
        )
        with patch("web.app.get_pipeline", return_value=SimpleNamespace(store=store)):
            result = _handle_preference_question("生活", "我喜欢什么颜色？")

        self.assertIn("蓝色", result["answer"])

    def test_advice_uses_global_attribute_but_rejects_foreign_object(self):
        store = self.make_store()
        global_colour = store.add_memory(
            date="2026-11-01",
            topic="生活",
            action="偏好",
            detail="我喜欢蓝色。",
            memory_type="long_term",
        )
        car_colour = store.add_memory(
            date="2026-11-02",
            topic="生活",
            action="偏好",
            detail="我的汽车喜欢黑色。",
            memory_type="long_term",
        )
        plan = SimpleNamespace(
            subject="衣服",
            topic="生活",
            query_text="我想买衣服，你说选什么颜色",
            record_text="我想买衣服",
        )

        self.assertGreater(_context_memory_relevance(store, global_colour, plan), 0)
        self.assertEqual(_context_memory_relevance(store, car_colour, plan), 0)


class TestSemanticRoutingAndForgetting(SharedSemanticTestCase):
    def test_offline_router_recognizes_personal_attribute_queries(self):
        with patch("memagent.router.llm.is_enabled", return_value=False):
            frequency = classify_query("我的跑步频率是什么？")
            reading_time = classify_query("我通常什么时间阅读？")

        self.assertEqual(frequency["intent"], "memory")
        self.assertEqual(reading_time["intent"], "memory")

    def test_clear_personal_attribute_query_skips_online_router_call(self):
        with patch("memagent.router.llm.is_enabled", return_value=True), \
                patch("memagent.router.llm.chat_json") as chat_json:
            result = classify_query("我的跑步频率是什么？")

        self.assertEqual(result["intent"], "memory")
        self.assertEqual(result["method"], "rule_fast")
        chat_json.assert_not_called()

    def test_general_knowledge_stays_general_even_when_memory_exists(self):
        with patch("memagent.router.llm.is_enabled", return_value=False):
            result = classify_query(
                "浅烘和深烘咖啡豆在酸度上有什么差别？",
                memory_probe=lambda: True,
            )

        self.assertEqual(result["intent"], "general")

    def test_ambiguous_fixed_configuration_uses_matching_memory_probe(self):
        with patch("memagent.router.llm.is_enabled", return_value=False):
            result = classify_query(
                "鱼缸多久换一次水？",
                memory_probe=lambda: True,
            )

        self.assertEqual(result["intent"], "memory")

    def test_personal_schedule_is_not_claimed_by_clock_utility(self):
        with patch("memagent.router.llm.is_enabled", return_value=False):
            result = classify_query("组会现在安排在周几晚上？")

        self.assertNotEqual(result["intent"], "utility")

    def test_semantic_forget_does_not_remove_another_domain(self):
        store = self.make_store()
        running = store.add_memory(
            date="2026-10-01",
            topic="健身",
            action="约定",
            detail="我每周跑步三次。",
        )
        coffee = store.add_memory(
            date="2026-10-02",
            topic="饮食",
            action="偏好",
            detail="我喜欢浅烘咖啡。",
        )
        pipeline = SimpleNamespace(store=store)

        with patch("web.app.get_pipeline", return_value=pipeline):
            result = _handle_forget("帮我忘掉跑步计划")

        self.assertEqual(result["type"], "forget")
        self.assertEqual(running.status, "discarded")
        self.assertEqual(coffee.status, "active")
        self.assertIn("我每周跑步三次", result["answer"])

    def test_excluded_entity_is_removed_from_weekend_evidence(self):
        store = self.make_store()
        running = store.add_memory(
            date="2026-10-01",
            topic="健身",
            action="约定",
            detail="我每周六晚上夜跑五公里。",
        )
        swimming = store.add_memory(
            date="2026-10-02",
            topic="运动",
            action="约定",
            detail="我通常在周日上午去游泳。",
        )
        enrich_memory_semantics(store)

        with patch("memagent.llm.is_enabled", return_value=False):
            hits = MemoryAgent(store).retriever.search(
                "周末除了夜跑，我还会去做什么运动？"
            )

        hit_ids = {memory.id for memory, _score in hits}
        self.assertIn(swimming.id, hit_ids)
        self.assertNotIn(running.id, hit_ids)

    def test_multi_object_query_keeps_each_objects_fact(self):
        store = self.make_store()
        train = store.add_memory(
            date="2026-10-01",
            topic="旅行",
            action="偏好",
            detail="坐高铁时我喜欢靠窗座位。",
        )
        plane = store.add_memory(
            date="2026-10-02",
            topic="旅行",
            action="偏好",
            detail="坐飞机时我喜欢靠过道座位。",
        )
        enrich_memory_semantics(store)

        with patch("memagent.llm.is_enabled", return_value=False):
            hits = MemoryAgent(store).retriever.search(
                "坐火车和坐飞机时，我分别喜欢坐哪边？"
            )

        hit_ids = {memory.id for memory, _score in hits}
        self.assertIn(train.id, hit_ids)
        self.assertIn(plane.id, hit_ids)


class TestConcurrentInitialization(SharedSemanticTestCase):
    def test_pipeline_is_constructed_once_for_parallel_first_requests(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store_path = Path(temporary.name) / "store.json"
        store_path.write_text("{}", encoding="utf-8")
        store = MemoryStore(str(store_path))
        fake_pipeline = SimpleNamespace(
            store=store,
            agent=SimpleNamespace(
                reference_date="",
                retriever=SimpleNamespace(reference_date=""),
            ),
        )

        def build(*_args, **_kwargs):
            time.sleep(0.03)
            return fake_pipeline

        with patch.object(web_app, "STORE_PATH", store_path), \
                patch.object(web_app, "_pipeline", None), \
                patch.object(web_app, "MemoryPipeline", side_effect=build) as constructor:
            with ThreadPoolExecutor(max_workers=8) as pool:
                pipelines = list(pool.map(lambda _index: web_app.get_pipeline(), range(8)))

        self.assertEqual(constructor.call_count, 1)
        self.assertTrue(all(pipeline is fake_pipeline for pipeline in pipelines))

    def test_store_file_stays_valid_during_concurrent_saves(self):
        store = self.make_store()
        store.add_memory(
            date="2026-11-02",
            topic="生活",
            action="记录",
            detail="并发保存测试。",
        )
        store.save()

        def save_many():
            for _index in range(30):
                store.save()

        def read_many():
            for _index in range(60):
                loaded = MemoryStore(store.path)
                self.assertEqual(len(loaded.memories), 1)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(save_many),
                pool.submit(save_many),
                pool.submit(read_many),
                pool.submit(read_many),
            ]
            for future in futures:
                future.result()
        with open(store.path, encoding="utf-8") as stream:
            json.load(stream)


if __name__ == "__main__":
    unittest.main()
