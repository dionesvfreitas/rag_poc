import unittest
from pathlib import Path

from parse_pdf import ParserConfig, extract_section_records, normalize_engine


class Label:
    def __init__(self, value):
        self.value = value


class Provenance:
    def __init__(self, page_no, bbox=None):
        self.page_no = page_no
        self.bbox = bbox


class BBox:
    def __init__(self, top=0.05, bottom=0.10):
        self.t = top
        self.b = bottom


class Item:
    def __init__(self, text, label="text", page_no=1, level=0, markdown=None, bbox=None):
        self.text = text
        self.label = Label(label)
        self.prov = [Provenance(page_no, bbox)]
        self._markdown = markdown
        if markdown is not None:
            self.text = ""

    def export_to_markdown(self, doc=None):
        return self._markdown


class Document:
    def __init__(self, items, page_total=1):
        self.pages = list(range(page_total))
        self._items = items

    def iterate_items(self):
        for item in self._items:
            yield item, getattr(item, "level", 0)


def config(**overrides):
    values = {
        "engine": "cpu",
        "remove_repeated_headers_footers": True,
        "header_footer_min_repetition_ratio": 0.30,
        "header_footer_max_text_length": 120,
        "max_chunk_chars": 3000,
        "min_chunk_chars": 1,
        "merge_small_chunks": False,
        "preserve_tables_as_chunks": True,
    }
    values.update(overrides)
    return ParserConfig(**values)


class ParserTests(unittest.TestCase):
    def test_default_engine_is_cpu(self):
        self.assertEqual(ParserConfig.from_env().engine, "cpu")
        self.assertFalse(ParserConfig.from_env().do_ocr)
        self.assertEqual(normalize_engine("cpu"), "cpu")

    def test_header_footer_detection_is_generic_and_conservative(self):
        items = []
        for page_no in range(1, 5):
            items.append(Item("Repeated short banner", page_no=page_no, bbox=BBox()))
            items.append(Item(f"Unique body content page {page_no}", page_no=page_no))

        chunks = extract_section_records(
            Document(items, page_total=4),
            config=config(),
            document_id="synthetic",
        )

        content = "\n".join(chunk["page_content"] for chunk in chunks)
        self.assertNotIn("Repeated short banner", content)
        self.assertIn("Unique body content page 1", content)
        self.assertIn("Unique body content page 4", content)

    def test_unique_short_text_is_not_removed(self):
        chunks = extract_section_records(
            Document([Item("Unique notice", page_no=1), Item("Body", page_no=2)], 2),
            config=config(),
            document_id="synthetic",
        )

        content = "\n".join(chunk["page_content"] for chunk in chunks)
        self.assertIn("Unique notice", content)

    def test_numbered_sections_build_section_path_and_clause_number(self):
        chunks = extract_section_records(
            Document(
                [
                    Item("7 DA ANALISE DAS PROPOSTAS", label="section_header", page_no=1),
                    Item("7.2 O julgamento das propostas compreende etapas", page_no=1),
                    Item("7.2.1.3.1 A reuniao sera registrada em ata", page_no=2),
                ],
                2,
            ),
            config=config(),
            document_id="synthetic",
        )

        clause_chunk = next(
            chunk for chunk in chunks if chunk["clause_number"] == "7.2.1.3.1"
        )
        self.assertEqual(clause_chunk["section_path"][-1], "7 DA ANALISE DAS PROPOSTAS")
        self.assertIn("7 DA ANALISE DAS PROPOSTAS", clause_chunk["section_path"])
        self.assertEqual(
            clause_chunk["metadata"]["clause_path"],
            ["7", "7.2", "7.2.1", "7.2.1.3", "7.2.1.3.1"],
        )

    def test_short_numbered_clause_is_not_title(self):
        chunks = extract_section_records(
            Document(
                [
                    Item("5 DA PARTICIPACAO", label="section_header", page_no=1),
                    Item("5.2.2 Que esteja suspensa pela entidade;", page_no=1),
                ],
                1,
            ),
            config=config(),
            document_id="synthetic",
        )

        clause_chunk = next(chunk for chunk in chunks if chunk["clause_number"] == "5.2.2")
        self.assertEqual(clause_chunk["content_type"], "list")
        self.assertEqual(clause_chunk["section_title"], "5 DA PARTICIPACAO")

    def test_macro_section_is_title(self):
        chunks = extract_section_records(
            Document([Item("4 CRONOGRAMA", label="section_header", page_no=1)], 1),
            config=config(),
            document_id="synthetic",
        )

        self.assertEqual(chunks[0]["content_type"], "title")
        self.assertEqual(chunks[0]["section_title"], "4 CRONOGRAMA")
        self.assertEqual(chunks[0]["metadata"]["clause_confidence"], 0.0)
        self.assertEqual(chunks[0]["metadata"]["section_confidence"], 1.0)

    def test_intermediate_introductory_item_sets_subsection_title(self):
        chunks = extract_section_records(
            Document(
                [
                    Item("5 DA PARTICIPACAO", label="section_header", page_no=1),
                    Item(
                        "5.2 Nao e admitida nesta licitacao a participacao de empresa(s):",
                        label="section_header",
                        page_no=1,
                    ),
                    Item("5.2.1 Em recuperacao judicial;", page_no=1),
                ],
                1,
            ),
            config=config(),
            document_id="synthetic",
        )

        subsection = next(chunk for chunk in chunks if chunk["clause_number"] == "5.2")
        leaf = next(chunk for chunk in chunks if chunk["clause_number"] == "5.2.1")
        self.assertEqual(subsection["section_title"], "5 DA PARTICIPACAO")
        self.assertEqual(
            subsection["subsection_title"],
            "5.2 Nao e admitida nesta licitacao a participacao de empresa(s):",
        )
        self.assertEqual(leaf["section_title"], "5 DA PARTICIPACAO")
        self.assertEqual(
            leaf["subsection_title"],
            "5.2 Nao e admitida nesta licitacao a participacao de empresa(s):",
        )
        self.assertEqual(leaf["section_path"], ["5 DA PARTICIPACAO"])
        self.assertEqual(subsection["metadata"]["subsection_confidence"], 1.0)
        self.assertLess(subsection["metadata"]["clause_confidence"], 1.0)

    def test_numbered_siblings_do_not_pollute_section_path(self):
        chunks = extract_section_records(
            Document(
                [
                    Item("6 DA PROPOSTA", label="section_header", page_no=1),
                    Item("6.2.1 Primeiro item comum;", page_no=1),
                    Item("6.2.2 Segundo item comum;", page_no=1),
                    Item("6.3 Terceiro item comum;", page_no=1),
                    Item("6.4 Quarto item comum;", page_no=1),
                ],
                1,
            ),
            config=config(),
            document_id="synthetic",
        )

        by_clause = {chunk["clause_number"]: chunk for chunk in chunks if chunk["clause_number"]}
        self.assertEqual(by_clause["6.2.2"]["section_path"], ["6 DA PROPOSTA"])
        self.assertEqual(by_clause["6.3"]["section_path"], ["6 DA PROPOSTA"])
        self.assertEqual(by_clause["6.4"]["section_path"], ["6 DA PROPOSTA"])
        self.assertEqual(by_clause["6.2.2"]["metadata"]["clause_path"], ["6", "6.2", "6.2.2"])
        self.assertEqual(by_clause["6.3"]["metadata"]["clause_path"], ["6", "6.3"])

    def test_deep_numbered_siblings_keep_clause_path_siblings(self):
        chunks = extract_section_records(
            Document(
                [
                    Item("5 DA PARTICIPACAO", label="section_header", page_no=1),
                    Item("5.2.11.2 A quem tenha relacao com outra pessoa;", page_no=1),
                    Item("5.2.11.3 Cujo proprietario tenha restricao;", page_no=1),
                ],
                1,
            ),
            config=config(),
            document_id="synthetic",
        )

        chunk = next(chunk for chunk in chunks if chunk["clause_number"] == "5.2.11.3")
        self.assertEqual(chunk["section_path"], ["5 DA PARTICIPACAO"])
        self.assertEqual(
            chunk["metadata"]["clause_path"],
            ["5", "5.2", "5.2.11", "5.2.11.3"],
        )

    def test_common_numeric_values_do_not_become_clause_number(self):
        samples = [
            "4004 0104 Capitais e regioes metropolitanas",
            "12/06/2026 data de recebimento",
            "R$ 1.600.000,00 valor estimado",
            "5% de desconto",
            "00.360.305/5649-03 cadastro",
        ]
        chunks = extract_section_records(
            Document([Item(sample, page_no=index + 1) for index, sample in enumerate(samples)], 5),
            config=config(),
            document_id="synthetic",
        )

        self.assertTrue(all(chunk["clause_number"] is None for chunk in chunks))

    def test_duplicate_whitespace_is_normalized_in_plain_text(self):
        chunks = extract_section_records(
            Document([Item("A  proposta   sera\n enviada", page_no=1)], 1),
            config=config(),
            document_id="synthetic",
        )

        self.assertEqual(chunks[0]["page_content"], "A proposta sera enviada")

    def test_tables_are_preserved_as_table_chunks_with_metadata(self):
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        chunks = extract_section_records(
            Document([Item(table, label="table", markdown=table, page_no=1)], 1),
            config=config(),
            document_id="synthetic",
        )

        self.assertEqual(chunks[0]["content_type"], "table")
        self.assertEqual(chunks[0]["markdown"], table)
        self.assertEqual(chunks[0]["page_content"], table)
        self.assertGreaterEqual(chunks[0]["metadata"]["table_estimated_rows"], 3)
        self.assertEqual(chunks[0]["metadata"]["table_estimated_columns"], 2)
        self.assertIn(chunks[0]["metadata"]["table_quality"], {"high", "medium", "low"})
        self.assertIn("table_syntax_quality", chunks[0]["metadata"])
        self.assertIn("table_semantic_quality", chunks[0]["metadata"])

    def test_misaligned_table_has_lower_quality_metadata(self):
        table = "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n|   |   |"
        chunks = extract_section_records(
            Document([Item(table, label="table", markdown=table, page_no=1)], 1),
            config=config(),
            document_id="synthetic",
        )

        self.assertIn(chunks[0]["metadata"]["table_quality"], {"medium", "low"})
        self.assertTrue(chunks[0]["metadata"]["table_quality_reasons"])

    def test_semantically_suspicious_table_does_not_get_max_quality(self):
        table = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| This first cell contains a very long displaced looking value | x | y |\n"
            "| Another very long value that dwarfs its neighbors | m | n |"
        )
        chunks = extract_section_records(
            Document([Item(table, label="table", markdown=table, page_no=1)], 1),
            config=config(),
            document_id="synthetic",
        )

        self.assertIn(chunks[0]["metadata"]["table_semantic_quality"], {"medium", "low"})
        self.assertTrue(chunks[0]["metadata"]["table_quality_reasons"])

    def test_cross_page_chunk_gets_continuation_metadata(self):
        chunks = extract_section_records(
            Document(
                [
                    Item("1 SECTION", label="section_header", page_no=1),
                    Item("1.1 This clause starts on one page", page_no=1),
                    Item("continues on the next page", page_no=2),
                ],
                2,
            ),
            config=config(),
            document_id="synthetic",
        )

        chunk = next(chunk for chunk in chunks if chunk["clause_number"] == "1.1")
        self.assertTrue(chunk["metadata"]["cross_page"])
        self.assertEqual(
            chunk["metadata"]["cross_page_reason"],
            "page_start_different_from_page_end",
        )

    def test_mid_sentence_flags_are_set(self):
        chunks = extract_section_records(
            Document([Item("continues without final punctuation", page_no=1)], 1),
            config=config(),
            document_id="synthetic",
        )

        self.assertTrue(chunks[0]["metadata"]["starts_mid_sentence"])
        self.assertTrue(chunks[0]["metadata"]["ends_mid_sentence"])
        self.assertGreater(chunks[0]["metadata"]["continuation_confidence"], 0.0)

    def test_large_chunks_split_and_keep_parent_chunk_id(self):
        long_text = "\n\n".join([f"Paragraph {index} " + ("x" * 40) for index in range(8)])
        chunks = extract_section_records(
            Document([Item("1 SECTION", label="section_header", page_no=1), Item(long_text, page_no=1)], 1),
            config=config(max_chunk_chars=120),
            document_id="synthetic",
        )

        split_chunks = [chunk for chunk in chunks if chunk["metadata"].get("split_total")]
        self.assertGreater(len(split_chunks), 1)
        self.assertTrue(all(chunk["parent_chunk_id"] for chunk in split_chunks))

    def test_page_fields_are_filled(self):
        chunks = extract_section_records(
            Document([Item("Body", page_no=2)], 3),
            config=config(),
            document_id="synthetic",
        )

        self.assertEqual(chunks[0]["page_no"], 2)
        self.assertEqual(chunks[0]["page_start"], 2)
        self.assertEqual(chunks[0]["page_end"], 2)
        self.assertEqual(chunks[0]["page_total"], 3)

    def test_no_document_specific_processing_terms_in_parser_source(self):
        source = Path("parse_pdf.py").read_text(encoding="utf-8").casefold()
        forbidden_terms = [
            "caixa",
            "sinapi",
            "portal de licitações",
            "portal de licitacoes",
        ]
        self.assertFalse(any(term in source for term in forbidden_terms))


if __name__ == "__main__":
    unittest.main()
