import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["MEMAGENT_LLM"] = "0"

from pipeline import MemoryPipeline
from tools.data_tool import load_records


class TestCoursePipeline(unittest.TestCase):
    def test_pipeline_writes_auditable_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = MemoryPipeline(root / "store.json", root / "run_log.jsonl")
            records = load_records(Path(__file__).parents[1] / "data" / "sample_diary.jsonl")
            report = pipeline.ingest(records)
            result = pipeline.ask("我最近在做什么课程项目？")

            self.assertEqual(report["records"], 100)
            self.assertTrue(result["evidence"])
            events = [
                json.loads(line)
                for line in (root / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event["event"] for event in events][:1], ["ingest_start"])
            self.assertIn("maintenance_complete", [event["event"] for event in events])
            self.assertEqual([event["event"] for event in events][-1], "query")
            # 任务决策留痕：冲突判断必须进日志，且方法可追溯
            decisions = [e for e in events if e["event"] == "reconciler_decision"]
            self.assertTrue(decisions, "冲突治理决策应写入日志")
            for d in decisions:
                self.assertIn(d.get("method"), ("rule_ingest", "rule_sweep", "llm"))
                self.assertIn("conflict", d)
                self.assertIn("reason", d)

    def test_empty_question_is_handled_as_an_abnormal_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = MemoryPipeline(Path(tmp) / "store.json", Path(tmp) / "run_log.jsonl")
            result = pipeline.ask("   ")
            self.assertEqual(result["evidence"], [])
            self.assertIn("问题为空", result["answer"])


if __name__ == "__main__":
    unittest.main()
