import asyncio
import base64
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import UploadFile
from starlette.datastructures import Headers

os.environ["MEMAGENT_LLM"] = "0"

from tools.document_tool import DocumentParseError
from web.app import (
    PhotoRequest,
    _attachment_context,
    _attachment_fallback_answer,
    _attachment_memory_note,
    _decode_image_data_url,
    _store_attachment_batch,
    add_photo,
    handle_attachments,
    handle_image,
    _attachment_memory_note,
)


class TestImageRouting(unittest.TestCase):
    def _request(self, note: str) -> PhotoRequest:
        return PhotoRequest(
            date="2026-11-02",
            note=note,
            dataUrl="data:image/png;base64,AA==",
        )

    def test_normal_image_question_uses_vision_without_writing_memory(self):
        with patch("web.app.vision_is_enabled", return_value=True), \
                patch("web.app.vision_chat", return_value="这是一个项目界面。"), \
                patch("web.app.add_photo") as add_photo:
            result = handle_image(self._request("这是什么界面？"))

        add_photo.assert_not_called()
        self.assertEqual(result["mode"], "analysis")
        self.assertTrue(result["visual_understanding"])
        self.assertIn("项目界面", result["answer"])

    def test_explicit_memory_command_uses_photo_memory_pipeline(self):
        with patch(
            "web.app.add_photo",
            return_value={"ok": True, "message": "已保存。"},
        ) as save_photo, patch("web.app.get_pipeline") as get_pipeline, \
                patch("web.app.vision_chat") as vision_chat:
            result = handle_image(self._request("帮我记住这张图"))

        save_photo.assert_called_once()
        get_pipeline.assert_not_called()
        vision_chat.assert_not_called()
        self.assertEqual(result["mode"], "saved")

    def test_explicit_image_command_does_not_store_command_fragment(self):
        with patch("web.app.add_photo", return_value={"ok": True}) as add_photo_call:
            result = handle_image(self._request("帮我记住这张图"))

        self.assertTrue(result["ok"])
        stored_request = add_photo_call.call_args.args[0]
        self.assertEqual(stored_request.note, "")

    def test_offline_image_question_falls_back_to_ocr(self):
        with patch("web.app.vision_is_enabled", return_value=False), \
                patch("web.app._ocr_data_url", return_value=("课程项目答辩", "")), \
                patch("web.app.is_enabled", return_value=False):
            result = handle_image(self._request("图里写了什么？"))

        self.assertEqual(result["mode"], "analysis")
        self.assertFalse(result["visual_understanding"])
        self.assertIn("课程项目答辩", result["answer"])


class TestCombinedAttachments(unittest.TestCase):
    def _upload(self, name: str, data: bytes, content_type: str) -> UploadFile:
        return UploadFile(
            io.BytesIO(data),
            filename=name,
            size=len(data),
            headers=Headers({"content-type": content_type}),
        )

    def test_attachment_storage_instructions_are_not_saved_as_user_facts(self):
        self.assertEqual(_attachment_memory_note("帮我记住这些附件"), "")
        self.assertEqual(_attachment_memory_note("帮我记住这些附件的核心结论"), "")
        self.assertEqual(_attachment_memory_note("帮我记住这张图"), "")
        self.assertEqual(
            _attachment_memory_note("帮我记住我的答辩时间是11月5日"),
            "我的答辩时间是11月5日",
        )

    def test_multiple_documents_and_text_are_analyzed_together(self):
        files = [
            self._upload("one.txt", "第一份：架构设计。".encode(), "text/plain"),
            self._upload("two.txt", "第二份：实验结果。".encode(), "text/plain"),
        ]
        with patch("web.app.is_enabled", return_value=True), \
                patch("web.app.chat", return_value="两份文件分别介绍架构和实验。"):
            result = asyncio.run(handle_attachments(
                files=files,
                prompt="综合这两个文件",
                day="2026-11-02",
            ))

        self.assertEqual(result["mode"], "analysis")
        self.assertEqual(len(result["files"]), 2)
        self.assertIn("架构和实验", result["answer"])

    def test_mixed_attachments_use_memory_pipeline_only_on_explicit_command(self):
        files = [
            self._upload("screen.png", b"image", "image/png"),
            self._upload("note.txt", "项目说明".encode(), "text/plain"),
        ]
        pipeline = SimpleNamespace(ingest=MagicMock())
        with patch("web.app._save_photo", return_value={"ok": True, "ocr_text": ""}), \
                patch("web.app._store_parsed_document", return_value={"ok": True}) as store_doc, \
                patch("web.app.get_pipeline", return_value=pipeline):
            result = asyncio.run(handle_attachments(
                files=files,
                prompt="帮我记住这些附件",
                day="2026-11-02",
            ))

        store_doc.assert_called_once()
        pipeline.ingest.assert_not_called()
        self.assertEqual(result["mode"], "saved")
        self.assertEqual(len(result["files"]), 2)
        self.assertIn("没有写入个人记忆", result["answer"])

    def test_explicit_attachment_note_is_the_only_memory_payload(self):
        files = [self._upload("notes.txt", "第三方案例中的项目安排".encode(), "text/plain")]
        pipeline = SimpleNamespace(ingest=MagicMock())
        with patch("web.app._store_parsed_document", return_value={"ok": True}), \
                patch("web.app.get_pipeline", return_value=pipeline):
            result = asyncio.run(handle_attachments(
                files=files,
                prompt="帮我记住，我的答辩时间是11月5日",
                day="2026-11-02",
            ))

        self.assertTrue(result["ok"])
        pipeline.ingest.assert_called_once()
        written = pipeline.ingest.call_args.args[0]
        self.assertEqual(len(written), 1)
        self.assertIn("我的答辩时间是11月5日", written[0]["text"])
        self.assertNotIn("第三方案例", written[0]["text"])

    def test_negated_memory_prompt_analyzes_without_writing(self):
        files = [self._upload("note.txt", "只用于分析".encode(), "text/plain")]
        with patch("web.app._save_photo") as save_photo, \
                patch("web.app._store_parsed_document") as store_doc, \
                patch("web.app.is_enabled", return_value=False):
            result = asyncio.run(handle_attachments(
                files=files,
                prompt="不要记住这些附件，只分析",
                day="2026-11-02",
            ))

        save_photo.assert_not_called()
        store_doc.assert_not_called()
        self.assertEqual(result["mode"], "analysis")

    def test_all_files_validate_before_any_memory_write(self):
        files = [
            self._upload("screen.png", b"image", "image/png"),
            self._upload("bad.exe", b"not supported", "application/octet-stream"),
        ]
        with patch("web.app._save_photo") as save_photo:
            result = asyncio.run(handle_attachments(
                files=files,
                prompt="帮我记住这些附件",
                day="2026-11-02",
            ))

        save_photo.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("不支持", result["message"])

    def test_document_context_allocates_space_to_every_file(self):
        parsed = [
            SimpleNamespace(
                filename=f"file-{index}.txt",
                sections=[("全文", marker * 3000)],
                char_count=3000,
            )
            for index, marker in enumerate(("甲", "乙", "丙"), 1)
        ]
        context = _attachment_context(parsed, limit=3000)
        for index, marker in enumerate(("甲", "乙", "丙"), 1):
            self.assertIn(f"《file-{index}.txt》", context)
            self.assertIn(marker, context)

    def test_offline_fallback_reports_files_without_dumping_contents(self):
        answer = _attachment_fallback_answer([
            {
                "name": "slides.pptx",
                "kind": "PowerPoint",
                "sections": 18,
                "status": "已提取文字",
            },
        ])
        self.assertIn("《slides.pptx》", answer)
        self.assertIn("PowerPoint", answer)
        self.assertIn("18 个内容区", answer)
        self.assertIn("已提取文字", answer)
        self.assertNotIn("OUTLINE", answer)

    def test_offline_image_without_ocr_still_returns_attachment_status(self):
        files = [self._upload("photo.png", b"image", "image/png")]
        with patch("web.app.vision_is_enabled", return_value=False), \
                patch("web.app.parse_document", side_effect=DocumentParseError("没有文字")):
            result = asyncio.run(handle_attachments(
                files=files,
                prompt="请分析图片",
                day="2026-11-02",
            ))

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "analysis")
        self.assertFalse(result["visual_understanding"])
        self.assertIn("photo.png", result["answer"])
        self.assertIn("未识别出文字", result["answer"])

    def test_all_image_entry_points_share_the_same_size_limit(self):
        raw = b"1234"
        data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        request = PhotoRequest(date="2026-11-02", note="分析", dataUrl=data_url)

        with patch("web.app.MAX_IMAGE_BYTES", 3):
            with self.assertRaises(DocumentParseError):
                _decode_image_data_url(data_url)

            image_result = handle_image(request)
            photo_result = add_photo(request)
            attachment_result = asyncio.run(handle_attachments(
                files=[self._upload("large.png", raw, "image/png")],
                prompt="分析",
                day="2026-11-02",
            ))

        self.assertFalse(image_result["ok"])
        self.assertFalse(photo_result["ok"])
        self.assertFalse(attachment_result["ok"])

    def test_photo_ocr_is_not_written_as_personal_memory(self):
        with patch("web.app._decode_image_data_url", return_value=(b"img", "image/png", ".png")), \
                patch("web.app.PHOTO_DIR") as photo_dir, \
                patch("web.app._load_photos", return_value=[]), \
                patch("web.app._save_photos"), \
                patch("web.app.ocr_path", return_value=[("图片", "页面中的第三方信息")]), \
                patch("web.app.get_pipeline") as get_pipeline, \
                patch("web.app.append_event"):
            photo_dir.mkdir = MagicMock()
            photo_dir.__truediv__ = MagicMock(return_value=SimpleNamespace(write_bytes=MagicMock()))
            result = add_photo(PhotoRequest(
                date="2026-11-02",
                note="",
                dataUrl="data:image/png;base64,AA==",
            ))

        self.assertTrue(result["ok"])
        get_pipeline.assert_not_called()

    def test_sensitive_document_stays_local_when_online_is_available(self):
        files = [self._upload("private.txt", "API_KEY: abcdef123456".encode(), "text/plain")]
        with patch("web.app.is_enabled", return_value=True), \
                patch("web.app.chat") as chat_call:
            result = asyncio.run(handle_attachments(
                files=files,
                prompt="概括这个文件",
                day="2026-11-02",
            ))

        self.assertTrue(result["ok"])
        self.assertTrue(result["privacy_local_only"])
        self.assertIn("未发送到模型接口", result["answer"])
        chat_call.assert_not_called()

    def test_mixed_batch_rolls_back_previous_items_on_storage_failure(self):
        loaded = [
            {"filename": "one.txt", "is_image": False, "parsed": object(), "data": b"one"},
            {"filename": "two.txt", "is_image": False, "parsed": object(), "data": b"two"},
        ]
        first = {"ok": True, "document_id": "d1", "storage_kind": "document"}
        with patch("web.app._store_parsed_document", side_effect=[first, RuntimeError("disk full")]), \
                patch("web.app._rollback_saved_attachments") as rollback:
            results, error = _store_attachment_batch(loaded, "2026-11-02")

        self.assertEqual(results, [])
        self.assertIn("disk full", error)
        rollback.assert_called_once_with([first])


if __name__ == "__main__":
    unittest.main()
