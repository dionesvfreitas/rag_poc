# Parser PDF RAG-Ready com Docling

PoC evoluída para uma base incremental de parsing documental auditável. O foco é:

```text
PDF -> Parser -> Blocos -> Seções -> Chunks -> Mini-RAG auditável local -> Retrieval Evaluation
```

Não há vector DB externo, BM25, reranker, LLM, chat, API web, frontend ou banco
de dados externo. A camada Mini-RAG usa embeddings locais com
`sentence-transformers`, persistência JSON e avaliação de recuperação também em
JSON.

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

## Execução Operacional

### Objetivo do projeto

O projeto combina três peças auditáveis:

- Parser Auditável: converte PDF em blocos, seções, chunks e metadados de proveniência.
- Mini-RAG Auditável: gera chunks hierárquicos, embeddings e índice local em JSON.
- Retrieval Evaluation: mede se o `Retriever` encontra os chunks esperados no top-k.

### Fluxo

```text
PDF
↓
Parser
↓
Chunks
↓
Embeddings
↓
Index
↓
Retrieval
↓
Evaluation
```

### Como gerar artefatos

Execute o parser:

```bash
.venv/bin/python parse_pdf.py
```

Por padrão, a CLI usa o primeiro `*.pdf` no diretório atual. Também é possível
passar o PDF como argumento:

```bash
.venv/bin/python parse_pdf.py documento.pdf
```

Caminhos e saídas podem ser configurados:

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

Os principais artefatos gerados são `parsed_sections.jsonl`,
`normalized_blocks.jsonl`, `normalized.md`, `parser_report.json` e, após o
chunker, `rag_chunks.jsonl`.

### Como construir índice

Para construir o índice vetorial local:

```bash
.venv/bin/python build_index.py
```

Com caminhos explícitos:

```bash
.venv/bin/python build_index.py --input parsed_sections.jsonl --input-type parsed_sections --output index/document_index.json
```

Para o smoke reproduzível da avaliação versionada no repositório:

```bash
.venv/bin/python build_index.py --input tests/fixtures/retrieval_eval/parsed_sections.jsonl --input-type parsed_sections --output index/document_index.json
```

`build_index.py` usa `rag_chunks.jsonl` quando existir. Caso contrário, lê
`parsed_sections.jsonl` e reaproveita `rag_chunker.build_hierarchical_chunks`
em memória, sem recriar chunking.

### Como executar avaliação

Execute a avaliação de recuperação contra um índice local:

```bash
.venv/bin/python evaluate_retrieval.py --dataset tests/fixtures/retrieval_eval/dataset.json --index index/document_index.json --output reports/retrieval_eval_report.json --top-k 5
```

O comando grava um report JSON com `schema_version`, metadados de dataset e
índice, configuração, métricas globais, diagnósticos e casos avaliados.

### Como interpretar métricas

- `top1_hit`: fração de perguntas com pelo menos um chunk esperado na posição 1.
- `top3_hit`: fração de perguntas com pelo menos um chunk esperado até a posição 3.
- `top5_hit`: fração de perguntas com pelo menos um chunk esperado até a posição 5.
- `MRR`: média de `1 / rank` do primeiro chunk relevante; vale `0` quando não há hit.
- `recall@1`: média da fração de itens esperados recuperados no top 1.
- `recall@3`: média da fração de itens esperados recuperados até o top 3.
- `recall@5`: média da fração de itens esperados recuperados até o top 5.

Os diagnósticos `hits_by_section_path` e `hits_by_page` ajudam a localizar
acertos por seção e página esperadas. Eles não substituem o critério principal:
a relevância usa `source_chunk_id` quando disponível e `chunk_id` apenas como
fallback.

### Consulta local

Para consultar sem LLM:

```bash
.venv/bin/python query.py --query "Qual o objeto do edital?"
```

A saída mostra `score`, `chunk_id`, `source_chunk_id`, página, seção,
`source_block_ids` e trecho do chunk recuperado.

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

Mini-RAG:

```bash
EMBEDDING_MODEL=BAAI/bge-m3
TOP_K=10
SIMILARITY_THRESHOLD=0.0
INDEX_PATH=index/document_index.json
```

O primeiro uso do modelo pode baixar/cachear pesos localmente via
`sentence-transformers`.

Headers e footers repetidos são marcados por padrão em `metadata`; não são removidos automaticamente.

## Artefatos

A CLI gera:

- `normalized_blocks.jsonl`: uma linha por `ParsedBlock`.
- `parsed_sections.jsonl`: chunks finais, com campos novos e campos legados de compatibilidade.
- `normalized.md`: inspeção humana com `page_no`, `block_id` e `section_path`.
- `parser_report.json`: métricas de auditoria.

A camada Mini-RAG gera:

- `index/document_index.json`: índice local com `chunk_id`, `content`,
  `embedding` e `metadata` auditável.
- `reports/retrieval_eval_report.json`: report de avaliação de recuperação
  quando `evaluate_retrieval.py` é executado.

Assets de imagem são salvos localmente hoje e referenciados por `asset_uri`, por exemplo
`images/page_001_figure_001.png`. O domínio já possui o contrato inicial
`DocumentAsset`/`AssetType` para a Onda 4, mas a integração completa em
`ParsedDocument` e storage externo ficam para uma etapa futura. Enquanto isso,
chunks continuam usando referências leves em `related_assets`.

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

A suíte cobre modelos, IDs, registry, normalização, reconstrução de seções,
chunk builder, exporters, pipeline, golden schema e `rag_chunker.py`. Também
cobre embeddings injetáveis, persistência do índice local, busca por
similaridade, retriever, citações, explainability do Mini-RAG, avaliação de
recuperação, report JSON e o fluxo E2E `build_index.py` -> `evaluate_retrieval.py`.

## Limitações e Próximos Passos

- O índice é JSON local e busca por cosine similarity em memória; é adequado
  para validação ponta-a-ponta, não para grande escala.
- Não há resposta generativa: `query.py` retorna apenas os chunks relevantes.
- A recuperação atual é sem BM25, sem reranker, sem vector DB externo, sem LLM,
  sem API e sem frontend.
