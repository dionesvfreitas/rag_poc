import re


HEADING_LABELS = {"title", "section_header"}
CLAUSE_RE = re.compile(
    r"^\s*(?P<number>\d{1,3}(?:\.\d{1,3})*)[.)]?\s+(?P<body>\S.+?)\s*$"
)
LEGAL_HEADING_RE = re.compile(
    r"^\s*(Art\.?|Artigo|CL[ÁA]USULA|CAP[ÍI]TULO|SE[ÇC][ÃA]O|T[ÍI]TULO|LIVRO|PARTE|ANEXO|AP[ÊE]NDICE)\b",
    re.I,
)
ANNEX_RE = re.compile(r"^\s*(ANEXO|ANNEX)\s+[IVXLCDM\d]+(?:[-\s][A-Z0-9]+)?\b", re.I)
FORMAL_DIVISION_RE = re.compile(
    r"^\s*(CAP[ÍI]TULO|SE[ÇC][ÃA]O|T[ÍI]TULO|LIVRO|PARTE)\s+"
    r"([IVXLCDM]+|\d+|[A-Z][A-ZÀ-Ú]{2,})\b",
    re.I,
)
ADMIN_METADATA_RE = re.compile(
    r"\b("
    r"AVISO|C[ÓO]DIGO|CONCORR[ÊE]NCIA|CONTRATO|CPF|CNPJ|DATA|DI[ÁA]RIO\s+OFICIAL|"
    r"DISPENSA|DOU|EDITAL|ENDERE[ÇC]O|E-?MAIL|HOMOLOGA[ÇC][ÃA]O|"
    r"INEXIGIBILIDADE|LICITA[ÇC][ÃA]O|LOCAL|MODALIDADE|PROCESSO|PROTOCOLO|"
    r"PUBLICA[ÇC][ÃA]O|PREG[ÃA]O|SUM[ÁA]RIO|[ÍI]NDICE"
    r")\b",
    re.I,
)
INSTITUTION_RE = re.compile(
    r"\b("
    r"ADMINISTRA[ÇC][ÃA]O|ASSESSORIA|AUTARQUIA|COORDENA[ÇC][ÃA]O|CONSELHO|"
    r"DEPARTAMENTO|DIRETORIA|ESTADO|FUNDA[ÇC][ÃA]O|GER[ÊE]NCIA|GOVERNO|"
    r"INSTITUTO|MINIST[ÉE]RIO|MUNIC[ÍI]PIO|PODER\s+(EXECUTIVO|JUDICI[ÁA]RIO|LEGISLATIVO)|"
    r"PREFEITURA|PROCURADORIA|SECRETARIA|SUPERINTEND[ÊE]NCIA|TRIBUNAL|UNIDADE|"
    r"UNIVERSIDADE"
    r")\b",
    re.I,
)
CONTACT_OR_ID_RE = re.compile(
    r"(https?://|\b\S+@\S+\b|\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b|"
    r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b|\bCEP\s*\d{5}-?\d{3}\b|"
    r"\b(?:\(?\d{2}\)?\s*)?\d{4,5}-?\d{4}\b)",
    re.I,
)
DATE_OR_TIME_RE = re.compile(
    r"^\s*(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\s+de\s+[a-zç]+\s+de\s+\d{4}|"
    r"\d{1,2}h\d{0,2}(?:min)?)(?:\W|$)",
    re.I,
)
SUMMARY_ENTRY_RE = re.compile(r"\.{2,}\s*\d+\s*$|^\s*\d+(?:\.\d+)*\s+.+\s+\d+\s*$")
NEGATIVE_CLAUSE_PATTERNS = [
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"^\s*\d{1,2}h\d{0,2}\b", re.I),
    re.compile(r"^\s*\d+(?:[\s-]\d+){2,}\b"),
    re.compile(r"^\s*\d+\s*%"),
    re.compile(r"^\s*https?://", re.I),
    re.compile(r"^\s*\S+@\S+"),
]
SENTENCE_END_RE = re.compile(r"[.;,]$")


def looks_upper_heading(text):
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.80 and len(text) <= 180 and not text.rstrip().endswith(".")


def clause_match(text):
    for pattern in NEGATIVE_CLAUSE_PATTERNS:
        if pattern.search(text):
            return None
    match = CLAUSE_RE.match(text)
    if not match:
        return None
    number = match.group("number").rstrip(".")
    body = match.group("body")
    if len(number.split(".")) > 8 or len(number) > 24:
        return None
    if not any(char.isalpha() for char in body):
        return None
    return match


def clause_number(text):
    match = clause_match(text)
    return match.group("number").rstrip(".") if match else None


def clause_level(number):
    return number.count(".")


def clause_tokens(number):
    parts = number.split(".")
    return [".".join(parts[: index + 1]) for index in range(len(parts))]


def clause_body(text):
    match = clause_match(text)
    return match.group("body") if match else ""


def normalized_line(text):
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text):
    return len(re.findall(r"\w+", text, re.UNICODE))


def has_explicit_body_marker(text):
    number = clause_number(text)
    if number and clause_level(number) <= 2:
        return True
    return bool(LEGAL_HEADING_RE.match(text) or ANNEX_RE.match(text) or FORMAL_DIVISION_RE.match(text))


def administrative_signal_count(text):
    signals = 0
    signals += len(ADMIN_METADATA_RE.findall(text))
    signals += len(INSTITUTION_RE.findall(text))
    if CONTACT_OR_ID_RE.search(text):
        signals += 1
    if DATE_OR_TIME_RE.search(text):
        signals += 1
    if SUMMARY_ENTRY_RE.search(text):
        signals += 1
    return signals


def has_visual_heading_evidence(block):
    metadata = getattr(block, "metadata", {}) or {}
    label = metadata.get("label", "")
    text = normalized_line(getattr(block, "text", ""))
    signals = 0
    if label in HEADING_LABELS or getattr(block, "content_type", "") in {"title", "heading"}:
        signals += 1
    if looks_upper_heading(text):
        signals += 1
    if metadata.get("bold") or metadata.get("is_bold") or metadata.get("strong"):
        signals += 1
    font_weight = metadata.get("font_weight")
    if isinstance(font_weight, int | float) and font_weight >= 600:
        signals += 1
    font_size = metadata.get("font_size")
    body_font_size = metadata.get("body_font_size")
    if isinstance(font_size, int | float) and isinstance(body_font_size, int | float):
        if font_size > body_font_size:
            signals += 1
    if metadata.get("font_size_rank") in {"largest", "large", "heading"}:
        signals += 1
    if metadata.get("visual_role") == "heading" or metadata.get("role") == "heading":
        signals += 1
    if isinstance(metadata.get("spacing_before"), int | float) and metadata["spacing_before"] > 0:
        signals += 1
    return signals >= 3


def is_front_matter_like(block):
    text = normalized_line(getattr(block, "text", ""))
    if not text:
        return True

    metadata = getattr(block, "metadata", {}) or {}
    label = metadata.get("label", "")
    if label in {"header", "footer", "page_header", "page_footer"}:
        return True
    if metadata.get("repeated_header_footer_candidate"):
        return True

    explicit_marker = has_explicit_body_marker(text)
    if word_count(text) <= 2 and not explicit_marker and not has_visual_heading_evidence(block):
        return True
    if CONTACT_OR_ID_RE.search(text) or DATE_OR_TIME_RE.search(text):
        return True
    if SUMMARY_ENTRY_RE.search(text):
        return True

    admin_signals = administrative_signal_count(text)
    if admin_signals >= 2:
        return True
    if admin_signals == 1 and not explicit_marker:
        return True
    if INSTITUTION_RE.search(text) and not explicit_marker:
        return True
    return False


def reasonable_section_title(text):
    text = normalized_line(text)
    if not text or len(text) > 220:
        return False
    if word_count(text) > 28:
        return False
    return any(char.isalpha() for char in text)


def is_body_section_start(block):
    text = normalized_line(getattr(block, "text", ""))
    if not reasonable_section_title(text):
        return False
    if is_front_matter_like(block):
        return False
    if has_explicit_body_marker(text):
        return True
    return has_visual_heading_evidence(block)


def heading_score(label, text):
    if ANNEX_RE.match(text) or LEGAL_HEADING_RE.match(text):
        return 5
    score = 0
    number = clause_number(text)
    body = clause_body(text)
    if label in HEADING_LABELS:
        score += 2
    if number and clause_level(number) == 0:
        score += 2
    if number and clause_level(number) <= 1 and len(text) <= 120:
        score += 1
    if looks_upper_heading(text):
        score += 2
    if body and looks_upper_heading(body):
        score += 1
    if text.endswith(":"):
        score += 1
    if SENTENCE_END_RE.search(text.rstrip(":")) and not text.endswith(":"):
        score -= 2
    if number and clause_level(number) >= 2:
        score -= 2
    return score


def is_structural_heading(label, text):
    if label in HEADING_LABELS and not clause_number(text):
        return True
    return heading_score(label, text) >= 3


def heading_kind(label, text):
    if not is_structural_heading(label, text):
        return None
    number = clause_number(text)
    if ANNEX_RE.match(text) or LEGAL_HEADING_RE.match(text):
        return "section"
    if not number:
        return "section"
    if clause_level(number) == 0:
        return "section"
    return "subsection"


def heading_level(text, docling_level=0):
    number = clause_number(text)
    if number:
        return clause_level(number)
    if ANNEX_RE.match(text) or LEGAL_HEADING_RE.match(text):
        return 0
    return max(docling_level, 0)


def update_section_path(section_path, title, level):
    if not title:
        return section_path
    if level <= 0:
        return [title]
    if len(section_path) < level:
        return section_path + [title]
    return section_path[:level] + [title]


def apply_sections(blocks):
    section_path = []
    subsection_title = None
    clause_path = []
    body_started = False
    for block in blocks:
        text = normalized_line(block.text)
        label = block.metadata.get("label", "")
        started_on_this_block = False
        number = clause_number(text)
        if number:
            clause_path = clause_tokens(number)
            block.metadata["clause_number"] = number
            block.metadata["clause_path"] = list(clause_path)
        elif clause_path and block.content_type != "table":
            block.metadata["clause_path"] = list(clause_path)

        if not body_started:
            if not is_body_section_start(block):
                block.section_path = []
                block.section_title = None
                continue
            body_started = True
            started_on_this_block = True

        kind = heading_kind(label, text)
        if kind == "section":
            section_path = update_section_path(
                section_path,
                text,
                heading_level(text, block.metadata.get("level", 0)),
            )
            subsection_title = None
        elif kind == "subsection":
            if not section_path:
                section_path = update_section_path(section_path, text, 0)
                subsection_title = None
            else:
                subsection_title = text
                block.metadata["subsection_title"] = subsection_title
        elif started_on_this_block and has_explicit_body_marker(text):
            section_path = update_section_path(section_path, text, 0)
            subsection_title = None

        block.section_path = list(section_path)
        block.section_title = section_path[-1] if section_path else None
        if subsection_title:
            block.metadata["subsection_title"] = subsection_title
    return blocks
