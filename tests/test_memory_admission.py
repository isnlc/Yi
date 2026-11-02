import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["MEMAGENT_LLM"] = "0"

from web.app import (
    _add_memory_candidate,
    _store_parsed_document,
    accept_memory_candidate,
    delete_document,
    dismiss_memory_candidate,
)


class TestMemoryAdmission(unittest.TestCase):
    def test_document_archive_does_not_create_personal_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_dir = root / "documents"
            catalog = document_dir / "catalog.json"
            parsed = SimpleNamespace(
                filename="arbitrary.docx",
                kind="Word",
                sections=[("正文", "第三方案例中的任意内容")],
                char_count=12,
                warning="",
            )
            with patch("web.app.DOCUMENT_DIR", document_dir), \
                    patch("web.app.DOCUMENT_CATALOG_PATH", catalog), \
                    patch("web.app.append_event"), \
                    patch("web.app.get_pipeline") as get_pipeline:
                result = _store_parsed_document(parsed, b"original-file", "2026-11-02")

            get_pipeline.assert_not_called()
            self.assertTrue(result["ok"])
            stored = json.loads(catalog.read_text(encoding="utf-8"))
            self.assertEqual(stored[0]["trust"], "untrusted_attachment")
            self.assertEqual(stored[0]["sections"][0]["text"], "第三方案例中的任意内容")

    def test_candidate_requires_user_acceptance_for_long_term_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            pending_path = Path(temporary) / "pending.json"
            log_path = Path(temporary) / "events.jsonl"
            pipeline = SimpleNamespace(ingest=MagicMock())
            with patch("web.app.PENDING_MEMORIES_PATH", pending_path), \
                    patch("web.app.LOG_PATH", log_path), \
                    patch("web.app.get_pipeline", return_value=pipeline):
                candidate = _add_memory_candidate(
                    "我以后每周跑步三次",
                    day="2026-11-02",
                    source="用户对话（待确认）",
                    reason="长期计划",
                )
                self.assertEqual(json.loads(pending_path.read_text(encoding="utf-8"))[0]["id"], candidate["id"])
                pipeline.ingest.assert_not_called()

                result = accept_memory_candidate(candidate["id"])

            self.assertTrue(result["ok"])
            pipeline.ingest.assert_called_once()
            self.assertEqual(json.loads(pending_path.read_text(encoding="utf-8")), [])

    def test_candidate_rejection_removes_it_without_memory_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            pending_path = Path(temporary) / "pending.json"
            log_path = Path(temporary) / "events.jsonl"
            with patch("web.app.PENDING_MEMORIES_PATH", pending_path), \
                    patch("web.app.LOG_PATH", log_path), \
                    patch("web.app.append_event"):
                candidate = _add_memory_candidate(
                    "我以后每周跑步三次",
                    day="2026-11-02",
                    source="用户对话（待确认）",
                    reason="长期计划",
                )
                result = dismiss_memory_candidate(candidate["id"])

            self.assertTrue(result["ok"])
            self.assertEqual(json.loads(pending_path.read_text(encoding="utf-8")), [])

    def test_document_delete_removes_original_and_catalog_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document_dir = root / "documents"
            document_dir.mkdir()
            original = document_dir / "stored.docx"
            original.write_bytes(b"original-file")
            catalog = document_dir / "catalog.json"
            catalog.write_text(json.dumps([{
                "id": "d123",
                "stored_file": "stored.docx",
            }]), encoding="utf-8")
            with patch("web.app.DOCUMENT_DIR", document_dir), \
                    patch("web.app.DOCUMENT_CATALOG_PATH", catalog), \
                    patch("web.app.LOG_PATH", root / "events.jsonl"), \
                    patch("web.app.append_event"):
                result = delete_document("d123")

            self.assertTrue(result["ok"])
            self.assertFalse(original.exists())
            self.assertEqual(json.loads(catalog.read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
