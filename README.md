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

Depois do parsing, gere a camada hierárquica para RAG:

```bash
.venv/bin/python rag_chunker.py
```

Por padrão, o chunker lê `parsed_sections.jsonl` e grava `rag_chunks.jsonl`.
Também é possível configurar os caminhos:

```bash
RAG_INPUT_JSONL=parsed_sections.jsonl RAG_OUTPUT_JSONL=rag_chunks.jsonl .venv/bin/python rag_chunker.py
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

Configurações da camada hierárquica:

```bash
RAG_TARGET_CHUNK_CHARS=1200
RAG_MAX_CHUNK_CHARS=1800
RAG_INCLUDE_SECTION_CONTEXT=true
```

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

## Chunking Hierárquico Para RAG

`rag_chunker.py` consome a saída do parser sem alterar seu contrato. Ele cria uma
representação em três níveis:

- `parent`: seção ou agrupamento lógico amplo, como `1 DO OBJETO`.
- `child`: cláusula, item ou subitem derivado do parser.
- `fragment`: pedaço semântico de um `child` grande demais para embedding.

Também pode gerar `section_context`, um chunk estrutural por seção, sem resumo
por IA. Ele lista subseções, cláusulas e quantidade de tabelas para ajudar o
retriever a entender a forma da seção.

Campos principais da saída hierárquica:

```json
{
  "chunk_type": "parent|section_context|child|fragment|table",
  "chunk_id": "uuid",
  "parent_chunk_id": "uuid|null",
  "section_root_chunk_id": "uuid",
  "sibling_chunk_ids": ["uuid"],
  "previous_chunk_id": "uuid|null",
  "next_chunk_id": "uuid|null",
  "document_id": "documento.pdf",
  "section_title": "string|null",
  "subsection_title": "string|null",
  "section_path": ["string"],
  "clause_number": "string|null",
  "clause_path": ["string"],
  "content": "texto"
}
```

Exemplo de entrada do parser:

```json
{
  "document_id": "doc.pdf",
  "chunk_id": "c1",
  "section_title": "1 DO OBJETO",
  "section_path": ["1 DO OBJETO"],
  "clause_number": "1.1",
  "page_content": "1.1 Seleção de pessoas físicas e jurídicas..."
}
```

Exemplo de saída do chunker:

```json
{
  "chunk_type": "child",
  "parent_chunk_id": "uuid-da-secao",
  "section_root_chunk_id": "uuid-da-secao",
  "clause_number": "1.1",
  "clause_path": ["1", "1.1"],
  "sibling_chunk_ids": ["uuid-irmao"],
  "previous_chunk_id": null,
  "next_chunk_id": "uuid-proximo",
  "content": "1.1 Seleção de pessoas físicas e jurídicas..."
}
```

Fragments são criados apenas quando o `child` excede `RAG_MAX_CHUNK_CHARS`. A
fragmentação respeita, nesta ordem, parágrafos, itens de lista, frases e, por
fim, quebra em limite de palavra. Assim, o texto não é dividido cegamente por
caracteres. Cada fragment preserva `section_path`, `clause_path`,
`parent_chunk_id`, `source_child_chunk_id` e links de vizinhança.

Essa estrutura prepara parent-child retrieval: o embedding pode ser calculado
sobre `child`, `fragment`, `table` e `section_context`; após a recuperação, a
aplicação consegue expandir contexto pelo `parent_chunk_id`, reconstruir a seção
via `section_root_chunk_id`, navegar por irmãos com `sibling_chunk_ids` e montar
janelas ordenadas com `previous_chunk_id` e `next_chunk_id`.

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

Na camada hierárquica, tabelas viram `chunk_type: "table"` e carregam
`table_quality`, `markdown` e `table_structure`. Quando uma tabela excede
`RAG_MAX_CHUNK_CHARS`, ela pode ser fragmentada por linhas, repetindo o cabeçalho
Markdown em cada fragmento. Não há split de tabela por caracteres.

## Testes

Os testes usam fixtures sintéticas e não dependem exclusivamente do PDF real:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Eles validam engine default, limpeza genérica, preservação de conteúdo único,
separação entre seções e cláusulas, falsos positivos numéricos, normalização de
whitespace, tabelas, split de chunks grandes, campos de página e a camada
hierárquica de RAG.

## Limitações

- A detecção de cabeçalhos/rodapés depende da qualidade dos blocos do Docling.
- Sem `bbox`, a limpeza usa apenas recorrência textual e fica mais conservadora.
- O parser não tenta corrigir semanticamente tabelas quebradas entre páginas.
- OCR não é habilitado por padrão. Use `DOCLING_DO_OCR=true` apenas quando for
  necessário para PDFs escaneados.
