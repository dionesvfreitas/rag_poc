import unittest

from parser_core.application.normalizer import normalize_blocks
from parser_core.application.pipeline import ParserPipelineConfig
from parser_core.application.sections import apply_sections, is_body_section_start, is_front_matter_like
from parser_core.domain.models import ParsedBlock


def block(text, sequence_no, label="text", metadata=None):
    block_metadata = {"label": label}
    if metadata:
        block_metadata.update(metadata)
    return ParsedBlock(
        block_id=f"b{sequence_no}",
        document_id="doc",
        page_no=1,
        page_total=1,
        sequence_no=sequence_no,
        content_type="unknown",
        text=text,
        metadata=block_metadata,
    )


def sectioned_texts(blocks):
    normalized, _ = normalize_blocks(blocks, ParserPipelineConfig())
    return apply_sections(normalized)


class SectionTests(unittest.TestCase):
    def test_numbered_section_and_clause_path_are_preserved(self):
        blocks, _ = normalize_blocks(
            [
                block("5 DA PARTICIPACAO", 1, "section_header"),
                block("5.2.1 Em recuperacao judicial;", 2),
            ],
            ParserPipelineConfig(),
        )

        sectioned = apply_sections(blocks)

        self.assertEqual(sectioned[1].section_path, ["5 DA PARTICIPACAO"])
        self.assertEqual(sectioned[1].metadata["clause_number"], "5.2.1")
        self.assertEqual(sectioned[1].metadata["clause_path"], ["5", "5.2", "5.2.1"])

    def test_legal_headings_are_sections(self):
        sectioned = sectioned_texts([block("CLÁUSULA PRIMEIRA DO OBJETO", 1)])

        self.assertEqual(sectioned[0].section_title, "CLÁUSULA PRIMEIRA DO OBJETO")

    def test_cover_before_first_numbered_section_is_front_matter(self):
        sectioned = sectioned_texts(
            [
                block("PODER EXECUTIVO", 1, "section_header"),
                block("SECRETARIA DE ADMINISTRACAO", 2, "section_header"),
                block("EDITAL 123/2026", 3, "section_header"),
                block("1 DO OBJETO", 4, "section_header"),
                block("Texto do corpo.", 5),
            ]
        )

        self.assertEqual(sectioned[0].section_path, ["front_matter"])
        self.assertEqual(sectioned[1].section_path, ["front_matter"])
        self.assertEqual(sectioned[2].section_path, ["front_matter"])
        self.assertEqual(sectioned[0].section_title, "front_matter")
        self.assertTrue(sectioned[2].metadata["is_front_matter"])
        self.assertEqual(sectioned[3].section_path, ["1 DO OBJETO"])
        self.assertEqual(sectioned[4].section_path, ["1 DO OBJETO"])

    def test_document_starting_directly_with_numbered_section_starts_body(self):
        sectioned = sectioned_texts(
            [
                block("1. INTRODUÇÃO", 1, "section_header"),
                block("Primeiro parágrafo.", 2),
            ]
        )

        self.assertEqual(sectioned[0].section_path, ["1. INTRODUÇÃO"])
        self.assertEqual(sectioned[1].section_path, ["1. INTRODUÇÃO"])

    def test_document_starting_with_hierarchical_number_starts_body(self):
        sectioned = sectioned_texts(
            [
                block("1.1 Escopo inicial do documento.", 1),
                block("Detalhamento.", 2),
            ]
        )

        self.assertEqual(sectioned[0].section_path, ["1.1 Escopo inicial do documento."])
        self.assertEqual(sectioned[1].section_path, ["1.1 Escopo inicial do documento."])

    def test_normative_artigo_starts_body(self):
        sectioned = sectioned_texts(
            [
                block("Art. 1º Esta norma estabelece critérios gerais.", 1),
                block("Parágrafo de execução.", 2),
            ]
        )

        self.assertEqual(sectioned[0].section_path, ["Art. 1º Esta norma estabelece critérios gerais."])
        self.assertEqual(sectioned[1].section_title, "Art. 1º Esta norma estabelece critérios gerais.")

    def test_annex_starts_body(self):
        sectioned = sectioned_texts(
            [
                block("ANEXO I", 1, "section_header"),
                block("Especificações técnicas.", 2),
            ]
        )

        self.assertEqual(sectioned[0].section_path, ["ANEXO I"])
        self.assertEqual(sectioned[1].section_path, ["ANEXO I"])

    def test_uppercase_cover_lines_do_not_start_sections(self):
        sectioned = sectioned_texts(
            [
                block("DOCUMENTO DE REFERÊNCIA", 1, "section_header"),
                block("UNIDADE ADMINISTRATIVA CENTRAL", 2, "section_header"),
                block("AVISO PUBLICADO EM 01/02/2026", 3, "section_header"),
                block("CAPÍTULO I DISPOSIÇÕES GERAIS", 4, "section_header"),
            ]
        )

        self.assertEqual(
            [item.section_path for item in sectioned[:3]],
            [["front_matter"], ["front_matter"], ["front_matter"]],
        )
        self.assertTrue(all(item.metadata["is_front_matter"] for item in sectioned[:3]))
        self.assertEqual(sectioned[3].section_path, ["CAPÍTULO I DISPOSIÇÕES GERAIS"])

    def test_summary_before_body_does_not_start_sections(self):
        sectioned = sectioned_texts(
            [
                block("SUMÁRIO", 1, "section_header"),
                block("1 INTRODUÇÃO ................ 3", 2),
                block("2 OBJETO ................ 4", 3),
                block("1 INTRODUÇÃO", 4, "section_header"),
                block("Conteúdo introdutório.", 5),
            ]
        )

        self.assertEqual(
            [item.section_path for item in sectioned[:3]],
            [["front_matter"], ["front_matter"], ["front_matter"]],
        )
        self.assertEqual(sectioned[3].section_path, ["1 INTRODUÇÃO"])
        self.assertEqual(sectioned[4].section_path, ["1 INTRODUÇÃO"])

    def test_visual_heading_without_formal_marker_stays_front_matter(self):
        sectioned = sectioned_texts(
            [
                block(
                    "EDITAL",
                    1,
                    "section_header",
                    metadata={
                        "is_bold": True,
                        "font_size": 16,
                        "body_font_size": 11,
                        "visual_role": "heading",
                    },
                ),
                block(
                    "MINISTÉRIO DA GESTÃO",
                    2,
                    "section_header",
                    metadata={
                        "is_bold": True,
                        "font_size": 16,
                        "body_font_size": 11,
                        "visual_role": "heading",
                    },
                ),
                block(
                    "CONCORRÊNCIA ELETRÔNICA",
                    3,
                    "section_header",
                    metadata={
                        "is_bold": True,
                        "font_size": 16,
                        "body_font_size": 11,
                        "visual_role": "heading",
                    },
                ),
                block(
                    "DO OBJETO",
                    4,
                    "section_header",
                    metadata={
                        "is_bold": True,
                        "font_size": 16,
                        "body_font_size": 11,
                        "visual_role": "heading",
                    },
                ),
                block("1 DO OBJETO", 5, "section_header"),
                block("Texto da seção.", 6),
            ]
        )

        self.assertEqual(
            [item.section_path for item in sectioned[:4]],
            [["front_matter"], ["front_matter"], ["front_matter"], ["front_matter"]],
        )
        self.assertTrue(all(item.metadata["is_front_matter"] for item in sectioned[:4]))
        self.assertEqual(sectioned[4].section_path, ["1 DO OBJETO"])
        self.assertEqual(sectioned[5].section_path, ["1 DO OBJETO"])

    def test_formal_markers_start_sections_without_visual_evidence(self):
        examples = [
            "1 DO OBJETO",
            "CLÁUSULA PRIMEIRA DO OBJETO",
            "Art. 1º Esta norma estabelece critérios gerais.",
        ]

        for index, text in enumerate(examples, start=1):
            with self.subTest(text=text):
                sectioned = sectioned_texts([block(text, index)])
                self.assertEqual(sectioned[0].section_path, [text])

    def test_front_matter_and_body_start_helpers_are_document_independent(self):
        cover = block("EDITAL 123/2026", 1, "section_header")
        body = block("Artigo 1º Das regras gerais", 2)

        self.assertTrue(is_front_matter_like(cover))
        self.assertFalse(is_body_section_start(cover))
        self.assertTrue(is_body_section_start(body))


if __name__ == "__main__":
    unittest.main()
