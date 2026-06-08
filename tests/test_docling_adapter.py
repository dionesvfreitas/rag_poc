import tempfile
import unittest
from pathlib import Path

from parser_core.infrastructure.docling_adapter import docling_document_to_parsed_document


class DictExportDocument:
    pages = [1]

    def export_to_dict(self):
        return {
            "texts": [
                {
                    "text": "1 DO OBJETO",
                    "label": "section_header",
                    "page_no": 1,
                }
            ]
        }


class MarkdownExportDocument:
    pages = [1]

    def export_to_markdown(self):
        return "# 1 DO OBJETO\n\nTexto do objeto."


class EmptyDocument:
    pages = [1]


def temp_pdf_path():
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "doc.pdf"
    path.write_bytes(b"pdf")
    return tmp, path


class DoclingAdapterTests(unittest.TestCase):
    def test_uses_dict_export_fallback_when_native_has_no_blocks(self):
        tmp, path = temp_pdf_path()
        self.addCleanup(tmp.cleanup)

        document = docling_document_to_parsed_document(DictExportDocument(), path)

        self.assertEqual(document.metadata["extraction_strategy"], "docling_dict_export")
        self.assertEqual(len(document.blocks), 1)
        self.assertEqual(document.blocks[0].text, "1 DO OBJETO")
        self.assertTrue(document.metadata["warnings"])

    def test_uses_markdown_fallback_when_dict_export_has_no_blocks(self):
        tmp, path = temp_pdf_path()
        self.addCleanup(tmp.cleanup)

        document = docling_document_to_parsed_document(MarkdownExportDocument(), path)

        self.assertEqual(document.metadata["extraction_strategy"], "docling_markdown_export")
        self.assertEqual([block.text for block in document.blocks], ["1 DO OBJETO", "Texto do objeto."])

    def test_records_clear_warning_when_all_strategies_fail(self):
        tmp, path = temp_pdf_path()
        self.addCleanup(tmp.cleanup)

        document = docling_document_to_parsed_document(EmptyDocument(), path)

        self.assertEqual(document.metadata["extraction_strategy"], "none")
        self.assertEqual(document.blocks, [])
        self.assertIn("produced no blocks", document.metadata["warnings"][-1])


if __name__ == "__main__":
    unittest.main()
