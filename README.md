# Parser PDF RAG-Ready com Docling

PoC evoluída para uma base incremental de parsing documental auditável. O foco é:

```text
PDF -> Parser -> Blocos -> Seções -> Chunks -> Artefatos auditáveis
```

Não há vector DB, embeddings, reranker, LLM, chat, API web ou banco de dados.

## Arquitetura

O código principal fica em `parser_core`:

- `domain`: contratos puros (`ParsedBlock`, `ParsedDocument`, `Chunk`, `ParserReport`) e IDs SHA256.
- `application`: normalização, reconstrução de seções, chunking e report.
- `infrastructure`: adapter Docling e exporters.
- `cli`: entrada de linha de comando.
- `registry`: registro de parsers por nome.

`parse_pdf.py` é apenas um wrapper compatível para a CLI. Imports de Docling devem ficar isolados em `parser_core/infrastructure`.

## Requisitos

O projeto usa Python `3.14.0`, conforme `.python-version`.

O repositório não está configurado com Poetry no momento: não há `pyproject.toml`
nem `poetry.lock`. A execução suportada hoje é via virtualenv local e
`requirements.txt`.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

A dependência principal segue fixada:

```text
docling==2.96.0
```

## Execução

Execute o parser:

```bash
.venv/bin/python parse_pdf.py
```

Por padrão, a CLI usa o primeiro `*.pdf` no diretório atual. Caminhos e saídas podem ser configurados:

```bash
INPUT_PDF=documento.pdf \
OUTPUT_JSONL=parsed_sections.jsonl \
BLOCKS_OUTPUT_JSONL=normalized_blocks.jsonl \
MARKDOWN_OUTPUT=normalized.md \
REPORT_OUTPUT=parser_report.json \
.venv/bin/python parse_pdf.py
```

Depois, gere a camada hierárquica para RAG:

```bash
.venv/bin/python rag_chunker.py
```

Exemplo completo usando o PDF local do repositório e saídas temporárias:

```bash
INPUT_PDF=LC_0071_2026_Edital_SINAPI.pdf \
OUTPUT_JSONL=/tmp/parsed_sections.jsonl \
BLOCKS_OUTPUT_JSONL=/tmp/normalized_blocks.jsonl \
MARKDOWN_OUTPUT=/tmp/normalized.md \
REPORT_OUTPUT=/tmp/parser_report.json \
.venv/bin/python parse_pdf.py

RAG_INPUT_JSONL=/tmp/parsed_sections.jsonl \
RAG_OUTPUT_JSONL=/tmp/rag_chunks.jsonl \
.venv/bin/python rag_chunker.py
```

## Configuração

Docling:

```bash
DOCLING_ENGINE=cpu
DOCLING_NUM_THREADS=8
DOCLING_DO_OCR=false
```

Engines aceitas: `auto`, `cpu`, `cuda`, `mps`, `xpu`.

Parser e chunking:

```bash
HEADER_FOOTER_MIN_REPETITION_RATIO=0.70
HEADER_FOOTER_MAX_TEXT_LENGTH=120
TARGET_CHUNK_CHARS=1200
MAX_CHUNK_CHARS=2400
MIN_CHUNK_CHARS=200
MERGE_SMALL_CHUNKS=true
PRESERVE_TABLES_AS_CHUNKS=true
```

Headers e footers repetidos são marcados por padrão em `metadata`; não são removidos automaticamente.

## Artefatos

A CLI gera:

- `normalized_blocks.jsonl`: uma linha por `ParsedBlock`.
- `parsed_sections.jsonl`: chunks finais, com campos novos e campos legados de compatibilidade.
- `normalized.md`: inspeção humana com `page_no`, `block_id` e `section_path`.
- `parser_report.json`: métricas de auditoria.

Campos principais de `Chunk`:

```json
{
  "chunk_id": "sha256",
  "document_id": "sha256",
  "chunk_no": 1,
  "text": "conteúdo",
  "page_start": 1,
  "page_end": 1,
  "section_title": "1 DO OBJETO",
  "section_path": ["1 DO OBJETO"],
  "source_block_ids": ["sha256"],
  "content_types": ["paragraph"],
  "metadata": {}
}
```

`document_id`, `block_id`, `chunk_id` e IDs do `rag_chunker.py` são determinísticos com SHA256.

## Seções e Tabelas

A reconstrução de seções reconhece padrões genéricos como `1`, `1.1`, `1.1.1`, `Art.`, `Artigo`, `CLÁUSULA`, `CAPÍTULO`, `ANEXO` e `APÊNDICE`.

Tabelas são preservadas como chunks próprios quando `PRESERVE_TABLES_AS_CHUNKS=true`. Metadados simples de qualidade e estrutura são registrados sem tentar corrigir semanticamente a tabela.

## Testes

```bash
.venv/bin/python -m unittest discover -s tests -v
```

A suíte cobre modelos, IDs, registry, normalização, reconstrução de seções, chunk builder, exporters, pipeline, golden schema e `rag_chunker.py`.
