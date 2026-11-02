import unittest
from unittest.mock import patch

from memagent.interaction import classify_interaction
from memagent.interaction import classify_interaction_online


class TestInteractionPolicy(unittest.TestCase):
    def test_clear_daily_event_is_a_recordable_fact(self):
        plan = classify_interaction("我今天要去跑步")
        self.assertEqual(plan.kind, "event")
        self.assertEqual(plan.topic, "健身")
        self.assertEqual(plan.subject, "跑步")
        self.assertEqual(plan.record_text, "我今天要去跑步")

    def test_event_and_advice_are_split(self):
        plan = classify_interaction("我今天想吃麻辣烫，你说我吃什么口味")
        self.assertEqual(plan.kind, "mixed_request")
        self.assertTrue(plan.needs_answer)
        self.assertEqual(plan.record_text, "我今天想吃麻辣烫")
        self.assertEqual(plan.query_text, "我今天想吃麻辣烫，你说我吃什么口味")
        self.assertEqual(plan.topic, "饮食")
        self.assertEqual(plan.subject, "麻辣烫")

    def test_decision_inherits_the_previous_episode(self):
        previous = {
            "memory_id": "m-event",
            "topic": "饮食",
            "subject": "麻辣烫",
            "date": "2026-11-02",
        }
        plan = classify_interaction("那就微辣吧", previous)
        self.assertEqual(plan.kind, "decision")
        self.assertEqual(plan.parent_id, "m-event")
        self.assertEqual(plan.preference_key, "微辣")
        self.assertIn("麻辣烫", plan.record_text)

    def test_intervening_empty_episode_does_not_break_followup(self):
        previous = {
            "memory_id": "m-event",
            "topic": "饮食",
            "subject": "麻辣烫",
        }
        self.assertEqual(
            classify_interaction("普通聊天", previous).kind,
            "none",
        )

    def test_explicit_stable_preference_is_distinct_from_a_daily_event(self):
        plan = classify_interaction("我一直喜欢浅烘咖啡")
        self.assertEqual(plan.kind, "stable_preference")
        self.assertEqual(plan.preference_key, "浅烘咖啡")
        self.assertEqual(classify_interaction("我今天想吃麻辣烫").kind, "event")

    def test_love_expression_is_a_stable_preference(self):
        plan = classify_interaction("我爱蒜香")
        self.assertEqual(plan.kind, "stable_preference")
        self.assertEqual(plan.topic, "饮食")
        self.assertEqual(plan.preference_key, "蒜香")

    def test_uncertain_preference_is_not_a_confirmed_preference(self):
        plan = classify_interaction("我最近可能更喜欢清淡一点")
        self.assertEqual(plan.kind, "inferred_preference_candidate")
        self.assertNotEqual(plan.kind, "stable_preference")

    def test_general_questions_do_not_enter_interaction_policy(self):
        for text in (
            "麻辣烫有哪些口味",
            "我喜欢什么口味？",
            "我今天吃了什么？",
            "用一句话解释黑洞",
        ):
            with self.subTest(text=text):
                self.assertIn(
                    classify_interaction(text).kind,
                    {"none", "preference_query"},
                )

    def test_broad_diet_preference_question_uses_preference_query(self):
        plan = classify_interaction("我有什么饮食偏好？")
        self.assertEqual(plan.kind, "preference_query")
        self.assertEqual(plan.topic, "饮食")

    def test_online_classifier_must_quote_a_contiguous_user_fact(self):
        with patch("memagent.interaction.llm.is_enabled", return_value=True), \
                patch("memagent.interaction.llm.chat_json", return_value={
                    "kind": "event",
                    "fact_clause": "我上周去看了电影",
                    "topic": "生活",
                    "subject": "看电影",
                    "preference_key": "",
                    "confidence": 0.94,
                    "reason": "明确的本人时间事件",
                }):
            plan = classify_interaction_online("我上周去看了电影，感觉很放松")
        self.assertEqual(plan.kind, "event")
        self.assertEqual(plan.record_text, "我上周去看了电影")
        self.assertEqual(plan.method, "llm")

    def test_online_classifier_rejects_model_invented_fact(self):
        with patch("memagent.interaction.llm.is_enabled", return_value=True), \
                patch("memagent.interaction.llm.chat_json", return_value={
                    "kind": "event",
                    "fact_clause": "我上周去了上海",
                    "topic": "旅行",
                    "subject": "上海",
                    "confidence": 0.99,
                }):
            plan = classify_interaction_online("我上周出门办事")
        self.assertEqual(plan.kind, "none")


if __name__ == "__main__":
    unittest.main()
