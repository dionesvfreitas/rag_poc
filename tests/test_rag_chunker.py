import unittest

from rag_chunker import RagChunkerConfig, build_hierarchical_chunks


def parsed_chunk(
    chunk_id,
    page_content,
    section_title="1 DO OBJETO",
    section_path=None,
    clause_number=None,
    clause_path=None,
    content_type="list",
    subsection_title=None,
    markdown=None,
    metadata=None,
):
    return {
        "document_id": "doc.pdf",
        "chunk_id": chunk_id,
        "parent_chunk_id": "parser-parent",
        "section_title": section_title,
        "subsection_title": subsection_title,
        "section_path": section_path or [section_title],
        "clause_number": clause_number,
        "page_no": 1,
        "page_start": 1,
        "page_end": 1,
        "page_total": 2,
        "content_type": content_type,
        "page_content": page_content,
        "markdown": markdown,
        "metadata": metadata or {"clause_path": clause_path or []},
    }


class RagChunkerTests(unittest.TestCase):
    def test_generates_parent_and_child_chunks(self):
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk("s1", "1 DO OBJETO", clause_number="1", content_type="title"),
                parsed_chunk("c1", "1.1 Seleção de pessoas.", clause_number="1.1", clause_path=["1", "1.1"]),
            ],
            config=RagChunkerConfig(include_section_context=False),
        )

        parent = next(chunk for chunk in chunks if chunk["chunk_type"] == "parent")
        child = next(chunk for chunk in chunks if chunk["chunk_type"] == "child")
        self.assertEqual(parent["section_title"], "1 DO OBJETO")
        self.assertEqual(child["parent_chunk_id"], parent["chunk_id"])
        self.assertIn(child["chunk_id"], parent["children"])
        self.assertEqual(child["clause_number"], "1.1")

    def test_large_child_generates_semantic_fragments(self):
        text = "\n\n".join(
            [
                "1.2 Primeiro paragrafo com conteudo suficiente para formar unidade semantica.",
                "Segundo paragrafo com mais conteudo e uma frase completa.",
                "Terceiro paragrafo com mais conteudo e outra frase completa.",
            ]
        )
        chunks = build_hierarchical_chunks(
            [parsed_chunk("c1", text, clause_number="1.2", clause_path=["1", "1.2"])],
            config=RagChunkerConfig(target_chunk_chars=80, max_chunk_chars=120, include_section_context=False),
        )

        child = next(chunk for chunk in chunks if chunk["chunk_type"] == "child")
        fragments = [chunk for chunk in chunks if chunk["chunk_type"] == "fragment"]
        self.assertGreater(len(fragments), 1)
        self.assertEqual(child["fragment_chunk_ids"], [fragment["chunk_id"] for fragment in fragments])
        self.assertTrue(all(fragment["source_child_chunk_id"] == child["chunk_id"] for fragment in fragments))
        self.assertTrue(all(len(fragment["content"]) <= 120 for fragment in fragments))

    def test_context_fields_are_preserved_in_fragments(self):
        text = " ".join([f"Sentence {index} with words." for index in range(20)])
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "c1",
                    text,
                    section_title="5 DA PARTICIPACAO",
                    section_path=["5 DA PARTICIPACAO"],
                    clause_number="5.2.1",
                    clause_path=["5", "5.2", "5.2.1"],
                    subsection_title="5.2 Nao e admitida:",
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=90, max_chunk_chars=120, include_section_context=False),
        )

        fragment = next(chunk for chunk in chunks if chunk["chunk_type"] == "fragment")
        self.assertEqual(fragment["section_path"], ["5 DA PARTICIPACAO"])
        self.assertEqual(fragment["clause_path"], ["5", "5.2", "5.2.1"])
        self.assertEqual(fragment["subsection_title"], "5.2 Nao e admitida:")

    def test_sibling_previous_and_next_links_are_generated(self):
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk("c1", "1.1 Primeira.", clause_number="1.1", clause_path=["1", "1.1"]),
                parsed_chunk("c2", "1.2 Segunda.", clause_number="1.2", clause_path=["1", "1.2"]),
                parsed_chunk("c3", "1.3 Terceira.", clause_number="1.3", clause_path=["1", "1.3"]),
            ],
            config=RagChunkerConfig(include_section_context=False),
        )

        children = [chunk for chunk in chunks if chunk["chunk_type"] == "child"]
        self.assertIsNone(children[0]["previous_chunk_id"])
        self.assertEqual(children[0]["next_chunk_id"], children[1]["chunk_id"])
        self.assertEqual(children[1]["previous_chunk_id"], children[0]["chunk_id"])
        self.assertEqual(children[1]["next_chunk_id"], children[2]["chunk_id"])
        self.assertIn(children[0]["chunk_id"], children[1]["sibling_chunk_ids"])
        self.assertIn(children[2]["chunk_id"], children[1]["sibling_chunk_ids"])

    def test_table_chunk_remains_integral(self):
        table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "t1",
                    table,
                    content_type="table",
                    markdown=table,
                    clause_number=None,
                    metadata={
                        "table_estimated_rows": 4,
                        "table_estimated_columns": 2,
                        "table_pages": [1],
                        "table_quality": "high",
                        "table_syntax_quality": "high",
                        "table_semantic_quality": "high",
                        "table_quality_reasons": [],
                    },
                )
            ],
            config=RagChunkerConfig(max_chunk_chars=500, include_section_context=False),
        )

        table_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
        self.assertEqual(table_chunk["markdown"], table)
        self.assertEqual(table_chunk["content"], table)
        self.assertEqual(table_chunk["table_structure"]["estimated_columns"], 2)

    def test_large_table_splits_only_by_rows(self):
        table = "| A | B |\n|---|---|\n" + "\n".join(
            f"| row {index} value | {index} |" for index in range(8)
        )
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "t1",
                    table,
                    content_type="table",
                    markdown=table,
                    metadata={"table_quality": "medium"},
                )
            ],
            config=RagChunkerConfig(max_chunk_chars=90, include_section_context=False),
        )

        table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
        self.assertGreater(len(table_chunks), 1)
        self.assertTrue(all(chunk["markdown"].startswith("| A | B |\n|---|---|") for chunk in table_chunks))
        self.assertTrue(all("table_fragment_index" in chunk for chunk in table_chunks))

    def test_section_context_chunk_lists_structure(self):
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk("c1", "5.1 Podem participar.", section_title="5 DA PARTICIPACAO", section_path=["5 DA PARTICIPACAO"], clause_number="5.1", clause_path=["5", "5.1"]),
                parsed_chunk("c2", "5.2 Nao e admitida:", section_title="5 DA PARTICIPACAO", section_path=["5 DA PARTICIPACAO"], clause_number="5.2", clause_path=["5", "5.2"], subsection_title="5.2 Nao e admitida:"),
                parsed_chunk("c3", "5.2.1 Em recuperacao.", section_title="5 DA PARTICIPACAO", section_path=["5 DA PARTICIPACAO"], clause_number="5.2.1", clause_path=["5", "5.2", "5.2.1"], subsection_title="5.2 Nao e admitida:"),
            ],
            config=RagChunkerConfig(include_section_context=True),
        )

        context = next(chunk for chunk in chunks if chunk["chunk_type"] == "section_context")
        self.assertIn("Seção: 5 DA PARTICIPACAO", context["content"])
        self.assertIn("- 5.1", context["content"])
        self.assertIn("- 5.2.1", context["content"])
        self.assertIn("5.2 Nao e admitida:", context["content"])

    def test_clause_order_preserves_parser_order(self):
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk("c1", "18.1 Primeira.", section_title="18 FINAIS", section_path=["18 FINAIS"], clause_number="18.1", clause_path=["18", "18.1"]),
                parsed_chunk("c2", "18.2 Segunda.", section_title="18 FINAIS", section_path=["18 FINAIS"], clause_number="18.2", clause_path=["18", "18.2"]),
                parsed_chunk("c3", "18.10 Decima.", section_title="18 FINAIS", section_path=["18 FINAIS"], clause_number="18.10", clause_path=["18", "18.10"]),
            ],
            config=RagChunkerConfig(include_section_context=True),
        )

        context = next(chunk for chunk in chunks if chunk["chunk_type"] == "section_context")
        child_clauses = [
            chunk["clause_number"] for chunk in chunks if chunk["chunk_type"] == "child"
        ]
        self.assertEqual(child_clauses, ["18.1", "18.2", "18.10"])
        self.assertLess(context["content"].index("- 18.2"), context["content"].index("- 18.10"))

    def test_hierarchy_can_be_reconstructed_from_metadata(self):
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk("c1", "1.1 Primeira.", clause_number="1.1", clause_path=["1", "1.1"]),
                parsed_chunk("c2", "1.2 Segunda.", clause_number="1.2", clause_path=["1", "1.2"]),
            ],
            config=RagChunkerConfig(include_section_context=False),
        )

        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        child = next(chunk for chunk in chunks if chunk.get("clause_number") == "1.2")
        parent = by_id[child["parent_chunk_id"]]
        siblings = [by_id[sibling_id] for sibling_id in child["sibling_chunk_ids"]]
        self.assertEqual(parent["chunk_type"], "parent")
        self.assertEqual(child["section_root_chunk_id"], parent["chunk_id"])
        self.assertEqual(parent["section_title"], child["section_title"])
        self.assertEqual(siblings[0]["clause_number"], "1.1")


if __name__ == "__main__":
    unittest.main()
