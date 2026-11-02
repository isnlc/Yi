import re
import unittest
from pathlib import Path


HTML = (
    Path(__file__).resolve().parents[1] / "web" / "index.html"
).read_text(encoding="utf-8")


class TestAttachmentFrontendContract(unittest.TestCase):
    def test_file_inputs_allow_multiple_selection(self):
        self.assertRegex(HTML, r'id="imgFile"[^>]*\bmultiple\b')
        self.assertRegex(HTML, r'id="docFile"[^>]*\bmultiple\b')

    def test_file_lists_are_copied_before_input_is_cleared(self):
        for input_id in ("imgFile", "docFile"):
            match = re.search(
                rf'\$\("{input_id}"\)\.onchange\s*=\s*async e => \{{(?P<body>.*?)\n\}};',
                HTML,
                re.S,
            )
            self.assertIsNotNone(match, input_id)
            body = match.group("body")
            copy_index = body.index("[...e.target.files]")
            clear_index = body.index('e.target.value = ""')
            self.assertLess(copy_index, clear_index, input_id)

    def test_combined_attachments_use_the_unified_endpoint(self):
        self.assertIn('fetch("/api/attachments"', HTML)
        self.assertIn('form.append("prompt", effectiveQuestion)', HTML)
        self.assertIn('form.append("files", item.file, item.file.name)', HTML)

    def test_file_cards_show_the_real_extension(self):
        self.assertIn('const extension = (item.file.name.split(".").pop()', HTML)
        self.assertIn('<span class="file-mark">${esc(extension)}</span>', HTML)

    def test_attachment_only_message_adds_default_analysis_prompt(self):
        self.assertIn(
            'const effectiveQuestion = q || "请分析这些附件。";',
            HTML,
        )
        self.assertIn(
            'const userMessage = addMsg("user", effectiveQuestion, attachments.length ? "media" : "");',
            HTML,
        )
        self.assertIn('form.append("prompt", effectiveQuestion)', HTML)

    def test_date_only_values_use_local_calendar_date(self):
        self.assertIn("const localDateIso = (value = new Date()) =>", HTML)
        self.assertIn("const today = () => localDateIso();", HTML)
        self.assertIn("const todayS = localDateIso();", HTML)
        self.assertNotIn("new Date().toISOString().slice(0, 10)", HTML)

    def test_document_memories_use_shared_collapsible_cards(self):
        self.assertIn("function parseDocumentMemory(value)", HTML)
        self.assertIn("【文件《", HTML)
        self.assertIn("function renderMemoryValue(value, compact = false)", HTML)
        self.assertIn('class="doc-memory ${compact ? "compact" : ""}"', HTML)
        self.assertIn("${renderMemoryValue(m.detail, true)}", HTML)
        self.assertIn("${renderMemoryValue(primaryDetail)}", HTML)
        self.assertIn("${renderMemoryValue(t)}", HTML)

    def test_memory_detail_prefers_full_document_source(self):
        self.assertIn(
            "const primaryDetail = d.source && parseDocumentMemory(d.source.text)",
            HTML,
        )
        self.assertIn("? d.source.text", HTML)

    def test_memory_ui_separates_candidates_and_documents_from_personal_memory(self):
        self.assertIn('data-memtab="memory"', HTML)
        self.assertIn('data-memtab="candidates"', HTML)
        self.assertIn('data-memtab="documents"', HTML)
        self.assertIn('fetch("/api/memory-candidates")', HTML)
        self.assertIn('fetch("/api/documents")', HTML)
        self.assertIn('method:"DELETE"', HTML)

    def test_memory_views_are_height_constrained_and_scrollable(self):
        self.assertIn(
            "#memoryView { flex: 1; display: flex; flex-direction: column; "
            "min-height: 0; overflow: hidden; }",
            HTML,
        )
        self.assertIn(
            ".mem-split { flex: 1; display: flex; min-height: 0; overflow: hidden; }",
            HTML,
        )
        self.assertRegex(HTML, r"\.mem-list \{[^}]*overflow-y: auto;")
        self.assertRegex(HTML, r"\.mem-detail \{[^}]*overflow-y: auto;")
        self.assertIn(
            ".memory-secondary { flex: 1; min-height: 0; overflow: auto;",
            HTML,
        )

    def test_chat_exposes_interaction_policy_scenarios_and_memory_effects(self):
        self.assertIn('id="chipsQuick"', HTML)
        self.assertIn('<span class="chip-lb">常用</span>', HTML)
        self.assertIn('{ label: "最近记录", query: "我最近做了什么？" }', HTML)
        self.assertIn('{ label: "当前计划", query: "我现在有什么计划？" }', HTML)
        self.assertIn('{ label: "我的偏好", query: "我有哪些长期偏好？" }', HTML)
        self.assertIn('{ label: "一周回顾", query: "回顾最近一周" }', HTML)
        self.assertIn('{ label: "记录事项", draft: "帮我记一下：" }', HTML)
        self.assertIn('$("q").value = item.draft', HTML)
        self.assertIn("memory_effects: d.memory_effects || []", HTML)
        self.assertIn('class="memory-effects"', HTML)


if __name__ == "__main__":
    unittest.main()
