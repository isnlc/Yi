import io
import unittest
from unittest.mock import patch

from tools.document_tool import DocumentParseError, parse_document


class TestDocumentParser(unittest.TestCase):
    def test_plain_text_and_json(self):
        text = parse_document("notes.txt", "课程项目采用三路检索。".encode("utf-8"))
        self.assertEqual(text.kind, "文本")
        self.assertIn("三路检索", text.sections[0][1])

        data = parse_document("data.json", b'{"accuracy": 0.979}')
        self.assertIn("accuracy", data.sections[0][1])

    def test_pdf(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 72 720 Td (Memory agent PDF content) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        buf = io.BytesIO()
        writer.write(buf)

        parsed = parse_document("report.pdf", buf.getvalue())
        self.assertEqual(parsed.kind, "PDF")
        self.assertIn("Memory agent PDF content", parsed.sections[0][1])

    def test_docx(self):
        from docx import Document
        doc = Document()
        doc.add_heading("Project Architecture", level=1)
        doc.add_paragraph("Recorder, Reconciler, Retriever and Auditor.")
        table = doc.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Accuracy"
        table.cell(0, 1).text = "97.9%"
        buf = io.BytesIO()
        doc.save(buf)

        parsed = parse_document("report.docx", buf.getvalue())
        self.assertEqual(parsed.kind, "Word")
        self.assertIn("Project Architecture", parsed.sections[0][1])
        self.assertIn("97.9%", parsed.sections[0][1])

    def test_xlsx(self):
        from openpyxl import Workbook
        book = Workbook()
        sheet = book.active
        sheet.title = "Results"
        sheet.append(["Method", "Accuracy"])
        sheet.append(["Memory Agent", 0.979])
        buf = io.BytesIO()
        book.save(buf)
        book.close()

        parsed = parse_document("results.xlsx", buf.getvalue())
        self.assertEqual(parsed.kind, "Excel")
        self.assertEqual(parsed.sections[0][0], "工作表：Results")
        self.assertIn("Memory Agent", parsed.sections[0][1])

    def test_pptx(self):
        from pptx import Presentation
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "System Architecture"
        slide.placeholders[1].text = "Input, memory management, retrieval and answer"
        buf = io.BytesIO()
        presentation.save(buf)

        parsed = parse_document("demo.pptx", buf.getvalue())
        self.assertEqual(parsed.kind, "PowerPoint")
        self.assertEqual(parsed.sections[0][0], "第 1 页")
        self.assertIn("System Architecture", parsed.sections[0][1])

    @patch("tools.document_tool._ocr_bytes")
    def test_scanned_pdf_uses_ocr_fallback(self, mocked_ocr):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        mocked_ocr.return_value = [("第 1 页", "OCR PDF TEST 2026")]

        parsed = parse_document("scan.pdf", buf.getvalue())
        self.assertEqual(parsed.kind, "扫描 PDF（OCR）")
        self.assertIn("OCR PDF TEST 2026", parsed.sections[0][1])
        mocked_ocr.assert_called_once()

    @patch("tools.document_tool._ocr_bytes")
    def test_image_uses_ocr(self, mocked_ocr):
        mocked_ocr.return_value = [("图片", "课程项目答辩")]
        parsed = parse_document("board.png", b"fake-image")
        self.assertEqual(parsed.kind, "图片（OCR）")
        self.assertIn("课程项目答辩", parsed.sections[0][1])

    def test_legacy_office_formats_are_explicitly_rejected(self):
        with self.assertRaisesRegex(DocumentParseError, "docx"):
            parse_document("old.doc", b"legacy")
        with self.assertRaisesRegex(DocumentParseError, "xlsx"):
            parse_document("old.xls", b"legacy")
        with self.assertRaisesRegex(DocumentParseError, "pptx"):
            parse_document("old.ppt", b"legacy")


if __name__ == "__main__":
    unittest.main()