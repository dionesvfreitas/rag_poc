import unittest

from rag_chunker import RagChunkerConfig, build_hierarchical_chunks, split_semantic_ranges, trim_range


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
    source_block_ids=None,
    source_spans=None,
    related_assets=None,
    document_region=None,
    page_start=1,
    page_end=1,
    page_total=2,
):
    record = {
        "document_id": "doc.pdf",
        "chunk_id": chunk_id,
        "parent_chunk_id": "parser-parent",
        "section_title": section_title,
        "subsection_title": subsection_title,
        "section_path": section_path or [section_title],
        "clause_number": clause_number,
        "page_no": 1,
        "page_start": page_start,
        "page_end": page_end,
        "page_total": page_total,
        "content_type": content_type,
        "page_content": page_content,
        "markdown": markdown,
        "metadata": metadata or {"clause_path": clause_path or []},
    }
    if source_block_ids is not None:
        record["source_block_ids"] = list(source_block_ids)
    if source_spans is not None:
        record["source_spans"] = list(source_spans)
    if related_assets is not None:
        record["related_assets"] = list(related_assets)
    if document_region is not None:
        record["document_region"] = document_region
    return record


def single_block_span(block_id, text):
    return {
        "block_id": block_id,
        "block_start_char": 0,
        "block_end_char": len(text),
        "chunk_start_char": 0,
        "chunk_end_char": len(text),
        "separator_before": None,
        "separator_after": None,
        "split_index": 0,
        "start_char": 0,
        "end_char": len(text),
    }


def block_span(block_id, block_text, chunk_start, *, separator_before=None, page_no=None):
    span = {
        "block_id": block_id,
        "block_start_char": 0,
        "block_end_char": len(block_text),
        "chunk_start_char": chunk_start,
        "chunk_end_char": chunk_start + len(block_text),
        "separator_before": separator_before,
        "separator_after": None,
        "split_index": 0,
        "start_char": 0,
        "end_char": len(block_text),
    }
    if page_no is not None:
        span["page_no"] = page_no
    return span


def two_block_source_text(first, second):
    return f"{first}\n\n{second}"


def two_block_spans(first, second, *, first_page=None, second_page=None):
    return [
        block_span("b1", first, 0, page_no=first_page),
        block_span("b2", second, len(first) + 2, separator_before="\n\n", page_no=second_page),
    ]


def rebuild_from_spans(blocks_by_id, spans):
    parts = []
    for span in sorted(spans, key=lambda item: item["chunk_start_char"]):
        if span.get("separator_before"):
            parts.append(span["separator_before"])
        source_text = blocks_by_id[span["block_id"]]
        parts.append(source_text[span["block_start_char"] : span["block_end_char"]])
        if span.get("separator_after"):
            parts.append(span["separator_after"])
    return "".join(parts)


def assert_valid_ranges(test_case, text, ranges):
    test_case.assertTrue(ranges)
    for start, end in ranges:
        test_case.assertGreaterEqual(start, 0)
        test_case.assertGreater(end, start)
        test_case.assertLessEqual(end, len(text))
        test_case.assertTrue(text[start:end].strip())


def linked_asset(asset_id="fig_001"):
    return {
        "asset_id": asset_id,
        "asset_uri": f"images/{asset_id}.png",
        "asset_type": "image",
        "link_strategy": "same_page_nearest_text",
        "link_reason": "figure shares same page and section with nearest text block",
        "link_evidence": {"decision": "linked", "reason": "same_page_nearest_text"},
        "metadata": {"target_block_id": "b1"},
    }


def targeted_asset(asset_id, target_block_id):
    asset = linked_asset(asset_id)
    asset["metadata"] = {"target_block_id": target_block_id}
    return asset


def evidence_targeted_asset(asset_id, target_block_id):
    asset = linked_asset(asset_id)
    asset["metadata"] = {}
    asset["link_evidence"] = {
        "decision": "linked",
        "reason": "same_page_nearest_text",
        "target_block_id": target_block_id,
    }
    return asset


def untargeted_asset(asset_id="fig_untargeted"):
    asset = linked_asset(asset_id)
    asset["metadata"] = {}
    asset["link_evidence"] = {"decision": "linked", "reason": "legacy_without_target"}
    return asset


def decorative_asset(asset_id="fig_999"):
    return {
        "asset_id": asset_id,
        "asset_uri": f"images/{asset_id}.png",
        "asset_type": "image",
        "link_strategy": "not_linked_decorative",
        "link_reason": "figure was marked decorative",
        "link_evidence": {"decision": "decorative", "reason": "decorative_front_matter_logo"},
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

    def test_child_chunk_preserves_source_spans_and_audited_assets(self):
        text = "Texto auditavel do objeto."
        asset = linked_asset()
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "c1",
                    text,
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", text)],
                    related_assets=[asset],
                )
            ],
            config=RagChunkerConfig(include_section_context=False),
        )

        child = next(chunk for chunk in chunks if chunk["chunk_type"] == "child")
        self.assertEqual(child["source_chunk_id"], "c1")
        self.assertEqual(child["source_block_ids"], ["b1"])
        self.assertEqual(child["source_spans"], [single_block_span("b1", text)])
        self.assertEqual(rebuild_from_spans({"b1": text}, child["source_spans"]), child["content"])
        self.assertEqual(child["related_assets"][0]["asset_id"], "fig_001")
        self.assertEqual(child["related_assets"][0]["link_strategy"], "same_page_nearest_text")
        self.assertIn("link_reason", child["related_assets"][0])
        self.assertIn("link_evidence", child["related_assets"][0])

    def test_fragment_chunk_adjusts_source_spans_and_reconstructs_exactly(self):
        text = "Primeira frase completa. Segunda frase completa. Terceira frase completa."
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "c1",
                    text,
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", text)],
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=24, max_chunk_chars=32, include_section_context=False),
        )

        fragments = [chunk for chunk in chunks if chunk["chunk_type"] == "fragment"]
        self.assertGreater(len(fragments), 1)
        second = fragments[1]
        self.assertEqual(second["source_block_ids"], ["b1"])
        self.assertGreater(second["source_spans"][0]["block_start_char"], 0)
        self.assertEqual(second["source_spans"][0]["chunk_start_char"], 0)
        self.assertLessEqual(second["source_spans"][0]["chunk_end_char"], len(second["content"]))
        self.assertEqual(rebuild_from_spans({"b1": text}, second["source_spans"]), second["content"])

    def test_word_boundary_fragments_preserve_spaces_tabs_newlines_and_spans(self):
        text = "Alpha  beta\tgamma\ndelta epsilon zeta eta theta iota kappa lambda mu"
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "c1",
                    text,
                    content_type="paragraph",
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", text)],
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=20, max_chunk_chars=25, include_section_context=False),
        )

        fragments = [chunk for chunk in chunks if chunk["chunk_type"] == "fragment"]
        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(fragment["content"].strip() for fragment in fragments))
        self.assertTrue(all(fragment["source_spans"] for fragment in fragments))
        self.assertTrue(all(fragment["source_block_ids"] for fragment in fragments))
        self.assertTrue(any("  " in fragment["content"] for fragment in fragments))
        self.assertTrue(any("\t" in fragment["content"] for fragment in fragments))
        self.assertTrue(any("\n" in fragment["content"] for fragment in fragments))
        for fragment in fragments:
            self.assertEqual(
                rebuild_from_spans({"b1": text}, fragment["source_spans"]),
                fragment["content"],
            )

    def test_second_paragraph_oversized_with_small_config_preserves_provenance(self):
        first = "Primeiro paragrafo curto com contexto inicial e texto valido."
        second_sentences = [
            (
                f"Sentenca {index} com varios termos relevantes,  espacos duplos, "
                f"tab\tinterno e conteudo suficiente para fragmentar corretamente."
            )
            for index in range(1, 8)
        ]
        text = f"{first}\n\n" + " ".join(second_sentences)
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "c1",
                    text,
                    content_type="paragraph",
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", text)],
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=240, max_chunk_chars=320, include_section_context=False),
        )

        fragments = [chunk for chunk in chunks if chunk["chunk_type"] == "fragment"]
        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(len(fragment["content"]) <= 320 for fragment in fragments))
        for fragment in fragments:
            self.assertTrue(fragment["source_spans"])
            self.assertTrue(fragment["source_block_ids"])
            self.assertEqual(
                rebuild_from_spans({"b1": text}, fragment["source_spans"]),
                fragment["content"],
            )

    def test_fragment_page_range_is_derived_from_spanned_block_page(self):
        first = "Bloco da pagina um com frase completa."
        second = "Bloco da pagina dois com frase completa. Outra frase da pagina dois completa."
        text = two_block_source_text(first, second)
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "cross",
                    text,
                    content_type="paragraph",
                    page_start=1,
                    page_end=2,
                    source_block_ids=["b1", "b2"],
                    source_spans=two_block_spans(first, second, first_page=1, second_page=2),
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=35, max_chunk_chars=45, include_section_context=False),
        )

        page_two_fragments = [
            chunk for chunk in chunks
            if chunk["chunk_type"] == "fragment" and chunk["source_block_ids"] == ["b2"]
        ]
        self.assertTrue(page_two_fragments)
        for fragment in page_two_fragments:
            self.assertEqual(fragment["page_start"], 2)
            self.assertEqual(fragment["page_end"], 2)
            self.assertEqual(fragment["metadata"]["page_range_strategy"], "derived_from_source_spans")
            self.assertEqual(fragment["metadata"]["source_block_ids"], fragment["source_block_ids"])
            self.assertEqual(fragment["metadata"]["parent_source_block_ids"], ["b1", "b2"])

    def test_fragment_page_range_can_span_pages_one_and_two(self):
        first = "Um curto."
        second = "Dois curto."
        third = "Terceiro bloco da pagina dois com frase completa. Quarta frase da pagina dois."
        text = f"{first}\n\n{second}\n\n{third}"
        spans = [
            block_span("b1", first, 0, page_no=1),
            block_span("b2", second, len(first) + 2, separator_before="\n\n", page_no=2),
            block_span(
                "b3",
                third,
                len(first) + len(second) + 4,
                separator_before="\n\n",
                page_no=2,
            ),
        ]
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "cross",
                    text,
                    content_type="paragraph",
                    page_start=1,
                    page_end=2,
                    source_block_ids=["b1", "b2", "b3"],
                    source_spans=spans,
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=20, max_chunk_chars=60, include_section_context=False),
        )

        fragment = next(
            chunk for chunk in chunks
            if chunk["chunk_type"] == "fragment" and chunk["source_block_ids"] == ["b1", "b2"]
        )
        self.assertEqual(fragment["page_start"], 1)
        self.assertEqual(fragment["page_end"], 2)
        self.assertEqual(fragment["metadata"]["page_range_strategy"], "derived_from_source_spans")
        self.assertEqual(fragment["metadata"]["source_block_ids"], fragment["source_block_ids"])

    def test_fragment_page_range_falls_back_when_spans_have_no_pages(self):
        first = "Primeiro bloco sem pagina."
        second = "Segundo bloco sem pagina. Outra frase para fragmentar."
        text = two_block_source_text(first, second)
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "fallback",
                    text,
                    content_type="paragraph",
                    page_start=3,
                    page_end=4,
                    source_block_ids=["b1", "b2"],
                    source_spans=two_block_spans(first, second),
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=25, max_chunk_chars=35, include_section_context=False),
        )

        fragments = [chunk for chunk in chunks if chunk["chunk_type"] == "fragment"]
        self.assertTrue(fragments)
        for fragment in fragments:
            self.assertEqual(fragment["page_start"], 3)
            self.assertEqual(fragment["page_end"], 4)
            self.assertEqual(fragment["metadata"]["page_range_strategy"], "inherited_from_source_chunk")
            self.assertEqual(fragment["metadata"]["source_block_ids"], fragment["source_block_ids"])

    def test_related_asset_with_metadata_target_only_propagates_to_matching_fragment(self):
        first = "Texto do primeiro bloco com frase completa."
        second = "Texto do segundo bloco com frase completa. Outra frase do segundo bloco."
        text = two_block_source_text(first, second)
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "asset-target",
                    text,
                    content_type="paragraph",
                    source_block_ids=["b1", "b2"],
                    source_spans=two_block_spans(first, second, first_page=1, second_page=1),
                    related_assets=[targeted_asset("fig_b2", "b2")],
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=35, max_chunk_chars=45, include_section_context=False),
        )

        b1_fragment = next(chunk for chunk in chunks if chunk["chunk_type"] == "fragment" and chunk["source_block_ids"] == ["b1"])
        b2_fragment = next(chunk for chunk in chunks if chunk["chunk_type"] == "fragment" and chunk["source_block_ids"] == ["b2"])
        self.assertEqual(b1_fragment["related_assets"], [])
        self.assertEqual([asset["asset_id"] for asset in b2_fragment["related_assets"]], ["fig_b2"])

    def test_related_asset_uses_link_evidence_target_block_fallback(self):
        first = "Texto do primeiro bloco com frase completa."
        second = "Texto do segundo bloco com frase completa. Outra frase do segundo bloco."
        text = two_block_source_text(first, second)
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "asset-evidence-target",
                    text,
                    content_type="paragraph",
                    source_block_ids=["b1", "b2"],
                    source_spans=two_block_spans(first, second, first_page=1, second_page=1),
                    related_assets=[evidence_targeted_asset("fig_b2", "b2")],
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=35, max_chunk_chars=45, include_section_context=False),
        )

        b2_fragment = next(chunk for chunk in chunks if chunk["chunk_type"] == "fragment" and chunk["source_block_ids"] == ["b2"])
        self.assertEqual([asset["asset_id"] for asset in b2_fragment["related_assets"]], ["fig_b2"])

    def test_untargeted_and_decorative_assets_do_not_propagate_to_partial_fragments(self):
        first = "Texto do primeiro bloco com frase completa."
        second = "Texto do segundo bloco com frase completa. Outra frase do segundo bloco."
        text = two_block_source_text(first, second)
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "asset-untargeted",
                    text,
                    content_type="paragraph",
                    source_block_ids=["b1", "b2"],
                    source_spans=two_block_spans(first, second, first_page=1, second_page=1),
                    related_assets=[untargeted_asset(), decorative_asset()],
                )
            ],
            config=RagChunkerConfig(target_chunk_chars=35, max_chunk_chars=45, include_section_context=False),
        )

        fragments = [chunk for chunk in chunks if chunk["chunk_type"] == "fragment"]
        self.assertTrue(fragments)
        for fragment in fragments:
            self.assertEqual(fragment["related_assets"], [])
            self.assertEqual(
                fragment["metadata"]["asset_propagation_skipped_reason"],
                "missing_target_block_id",
            )

    def test_semantic_ranges_use_absolute_offsets_for_non_initial_paragraphs(self):
        prefix = "Paragrafo inicial curto apenas para deslocar o offset."
        cases = {
            "list": "\n".join(
                [
                    "- Primeiro item com texto suficiente para passar do limite pequeno",
                    "- Segundo item com texto suficiente para virar outra unidade",
                    "- Terceiro item com texto suficiente para manter lista valida",
                ]
            ),
            "sentence": " ".join(
                [
                    "Primeira sentenca longa o bastante para fragmentar.",
                    "Segunda sentenca longa o bastante para fragmentar.",
                    "Terceira sentenca longa o bastante para fragmentar.",
                    "Quarta sentenca longa o bastante para fragmentar.",
                ]
            ),
            "word": " ".join(f"palavra{index}" for index in range(1, 35)),
        }
        for label, body in cases.items():
            with self.subTest(label=label):
                text = f"{prefix}\n\n{body}"
                second_paragraph_start = len(prefix) + 2
                ranges = split_semantic_ranges(text, target_chunk_chars=70, max_chunk_chars=90)

                assert_valid_ranges(self, text, ranges)
                self.assertTrue(any(start >= second_paragraph_start for start, _end in ranges))

    def test_trim_range_corrects_invalid_bounds_without_index_error(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(trim_range("  texto  ", -5, 99), (2, 7))
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(trim_range("texto", 4, 2), (2, 2))

    def test_table_spans_follow_markdown_when_markdown_differs_from_text(self):
        text = "Tabela extraida em texto simples"
        markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "t1",
                    text,
                    content_type="table",
                    markdown=markdown,
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", markdown)],
                    metadata={"table_quality": "high"},
                )
            ],
            config=RagChunkerConfig(max_chunk_chars=200, include_section_context=False),
        )

        table_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
        self.assertEqual(table_chunk["content"], markdown)
        self.assertEqual(table_chunk["source_block_ids"], ["b1"])
        self.assertTrue(table_chunk["source_spans"])
        self.assertEqual(
            rebuild_from_spans({"b1": markdown}, table_chunk["source_spans"]),
            table_chunk["content"],
        )

    def test_split_table_chunks_with_repeated_header_remain_reconstructable(self):
        markdown = "| A | B |\n|---|---|\n" + "\n".join(
            f"| row {index} value | {index} |" for index in range(8)
        )
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "t1",
                    "Tabela extraida em texto simples",
                    content_type="table",
                    markdown=markdown,
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", markdown)],
                    metadata={"table_quality": "medium"},
                )
            ],
            config=RagChunkerConfig(max_chunk_chars=90, include_section_context=False),
        )

        table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
        self.assertGreater(len(table_chunks), 1)
        for table_chunk in table_chunks:
            self.assertTrue(table_chunk["source_spans"])
            self.assertTrue(table_chunk["source_block_ids"])
            self.assertEqual(
                rebuild_from_spans({"b1": markdown}, table_chunk["source_spans"]),
                table_chunk["content"],
            )

    def test_front_matter_region_is_preserved_in_child_chunk(self):
        text = "CAPA DO DOCUMENTO"
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "fm1",
                    text,
                    section_title="front_matter",
                    section_path=["front_matter"],
                    metadata={"is_front_matter": True},
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", text)],
                )
            ],
            config=RagChunkerConfig(include_section_context=False),
        )

        child = next(chunk for chunk in chunks if chunk["chunk_type"] == "child")
        self.assertEqual(child["section_path"], ["front_matter"])
        self.assertEqual(child["section_title"], "front_matter")
        self.assertEqual(child["document_region"], "front_matter")
        self.assertTrue(child["metadata"]["is_front_matter"])
        self.assertEqual(rebuild_from_spans({"b1": text}, child["source_spans"]), child["content"])

    def test_decorative_assets_are_not_reintroduced_as_related_assets(self):
        text = "Texto real da secao."
        chunks = build_hierarchical_chunks(
            [
                parsed_chunk(
                    "c1",
                    text,
                    source_block_ids=["b1"],
                    source_spans=[single_block_span("b1", text)],
                    related_assets=[decorative_asset()],
                )
            ],
            config=RagChunkerConfig(include_section_context=False),
        )

        child = next(chunk for chunk in chunks if chunk["chunk_type"] == "child")
        self.assertEqual(child["related_assets"], [])


if __name__ == "__main__":
    unittest.main()
