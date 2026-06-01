# PoC Docling PDF Parser

PoC para validar o parser de PDF com `docling==2.96.0` e gerar chunks
semânticos adequados para um futuro pipeline de RAG.

O projeto não implementa vector store, embeddings, LLM ou API web. O foco é
extrair, normalizar e enriquecer o conteúdo do PDF de forma genérica.

## Requisitos

O projeto usa Python `3.14.0`, conforme `.python-version`, e uma virtualenv local
em `.venv`.

Instale as dependências:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Execução

Execute com a engine default, `cpu`:

```bash
.venv/bin/python parse_pdf.py
```

Por padrão, o script usa o primeiro arquivo `*.pdf` encontrado no diretório atual
e grava `parsed_sections.jsonl`. Também é possível configurar os caminhos:

```bash
INPUT_PDF=documento.pdf OUTPUT_JSONL=chunks.jsonl .venv/bin/python parse_pdf.py
```

## Configuração

`cpu` é o default para manter a execução previsível no macOS.

```bash
DOCLING_ENGINE=cpu .venv/bin/python parse_pdf.py
DOCLING_ENGINE=mps .venv/bin/python parse_pdf.py
DOCLING_NUM_THREADS=4 .venv/bin/python parse_pdf.py
DOCLING_DO_OCR=false .venv/bin/python parse_pdf.py
```

Engines aceitas:

- `auto`
- `cpu`
- `cuda`
- `mps`
- `xpu`

Configurações de limpeza e chunking:

```bash
REMOVE_REPEATED_HEADERS_FOOTERS=true
HEADER_FOOTER_MIN_REPETITION_RATIO=0.30
HEADER_FOOTER_MAX_TEXT_LENGTH=120
MAX_CHUNK_CHARS=3000
MIN_CHUNK_CHARS=300
MERGE_SMALL_CHUNKS=true
PRESERVE_TABLES_AS_CHUNKS=true
```

Artefatos de debug são opcionais:

```bash
DEBUG_ARTIFACTS_ENABLED=true
DEBUG_ARTIFACTS_DIR=.artifacts/parser
```

Quando habilitado, o parser salva blocos brutos, blocos removidos como candidatos
a cabeçalho/rodapé e chunks finais.

## Saída

O script gera uma linha JSON por chunk semântico. Campos principais:

```json
{
  "document_id": "documento.pdf",
  "chunk_id": "uuid",
  "parent_chunk_id": "uuid|null",
  "section_title": "string|null",
  "subsection_title": "string|null",
  "section_path": ["string"],
  "clause_number": "string|null",
  "page_no": 1,
  "page_start": 1,
  "page_end": 1,
  "page_total": 28,
  "content_type": "title|paragraph|list|table|mixed",
  "page_content": "texto",
  "markdown": "string|null",
  "metadata": {}
}
```

`page_no` foi mantido por compatibilidade. `page_start` e `page_end` representam
o intervalo real do chunk. Quando um chunk pertence a uma cláusula numerada, o
número aparece em `clause_number` e a trilha hierárquica fica em
`metadata.clause_path`.

## Limpeza Genérica

A remoção de cabeçalhos e rodapés é conservadora. O parser procura blocos curtos
que se repetem em várias páginas, usa posição quando o Docling fornece `bbox`, e
aplica um threshold mais alto quando só há recorrência textual.

Para reduzir falso positivo, a limpeza é ignorada em documentos com menos de três
páginas e exige ocorrência em pelo menos duas páginas. Blocos removidos podem ser
auditados nos artefatos de debug.

## Hierarquia e Chunking

A hierarquia é inferida por sinais estruturais genéricos:

- títulos detectados pelo Docling;
- seções macro numeradas, como `1 DO OBJETO` ou `4. CRONOGRAMA`;
- anexos;
- headings curtos em caixa alta.

Cláusulas comuns, como `5.2.2 ...`, não viram `section_title` por padrão. Elas
preservam `clause_number`, herdam a seção vigente e recebem metadados como
`clause_path`, `heading_confidence`, `section_confidence`,
`subsection_confidence` e `clause_confidence`.

Itens intermediários que introduzem filhos, como `5.2 ...:`, podem preencher
`subsection_title`. Assim, `section_title` continua representando a seção macro,
enquanto `subsection_title` representa agrupamentos internos.

Chunks que atravessam páginas ou parecem continuação recebem metadados como
`cross_page`, `cross_page_reason`, `starts_mid_sentence`, `ends_mid_sentence` e
`continuation_confidence`. Esses sinais são apenas auditáveis; o parser não junta
conteúdo agressivamente por causa deles.

O chunking usa unidades semânticas: título, parágrafo, item numerado, lista e
tabela. Chunks grandes são divididos respeitando `MAX_CHUNK_CHARS` e mantendo
`parent_chunk_id` para recomposição por seção.

## Tabelas

Tabelas são preservadas como chunks próprios quando `PRESERVE_TABLES_AS_CHUNKS`
está habilitado. O campo `markdown` mantém a representação retornada pelo
Docling, e `metadata` inclui estimativas de linhas, colunas, páginas envolvidas e
continuação entre páginas.

Também são gerados metadados conservadores de qualidade:

```json
{
  "table_syntax_quality": "high|medium|low",
  "table_semantic_quality": "high|medium|low|unknown",
  "table_quality": "high|medium|low",
  "table_quality_reasons": []
}
```

Esses sinais indicam problemas estruturais e suspeitas semânticas simples, como
contagem inconsistente de colunas, muitas células vazias, alta variação de
tamanho entre células, linhas muito longas ou possível continuação de página. O
parser não tenta reconstruir semanticamente tabelas.

## Testes

Os testes usam fixtures sintéticas e não dependem exclusivamente do PDF real:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Eles validam engine default, limpeza genérica, preservação de conteúdo único,
separação entre seções e cláusulas, falsos positivos numéricos, normalização de
whitespace, tabelas, split de chunks grandes e campos de página.

## Limitações

- A detecção de cabeçalhos/rodapés depende da qualidade dos blocos do Docling.
- Sem `bbox`, a limpeza usa apenas recorrência textual e fica mais conservadora.
- O parser não tenta corrigir semanticamente tabelas quebradas entre páginas.
- OCR não é habilitado por padrão. Use `DOCLING_DO_OCR=true` apenas quando for
  necessário para PDFs escaneados.
