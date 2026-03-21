"""
Convierte ADAPT-ECG_Documentacion_Tecnica.md a PDF usando reportlab.
"""
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

MD_PATH  = Path("docs/ADAPT-ECG_Documentacion_Tecnica.md")
PDF_PATH = Path("docs/ADAPT-ECG_Documentacion_Tecnica.pdf")

# ── Estilos ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

STYLES = {
    "h1": ParagraphStyle("h1", parent=base["Heading1"],
                         fontSize=18, textColor=colors.HexColor("#1a237e"),
                         spaceAfter=6, spaceBefore=14),
    "h2": ParagraphStyle("h2", parent=base["Heading2"],
                         fontSize=14, textColor=colors.HexColor("#283593"),
                         spaceAfter=4, spaceBefore=12),
    "h3": ParagraphStyle("h3", parent=base["Heading3"],
                         fontSize=12, textColor=colors.HexColor("#3949ab"),
                         spaceAfter=3, spaceBefore=8),
    "h4": ParagraphStyle("h4", parent=base["Heading4"],
                         fontSize=11, textColor=colors.HexColor("#5c6bc0"),
                         spaceAfter=2, spaceBefore=6),
    "body": ParagraphStyle("body", parent=base["Normal"],
                           fontSize=10, leading=14, spaceAfter=4),
    "blockquote": ParagraphStyle("bq", parent=base["Normal"],
                                 fontSize=10, leading=13,
                                 leftIndent=16, rightIndent=16,
                                 textColor=colors.HexColor("#444444"),
                                 backColor=colors.HexColor("#e8eaf6"),
                                 spaceAfter=6, spaceBefore=4),
    "code": ParagraphStyle("code", parent=base["Code"],
                           fontSize=8, leading=11,
                           fontName="Courier",
                           backColor=colors.HexColor("#f5f5f5"),
                           leftIndent=12),
}

def escape_xml(text):
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

def inline_fmt(text):
    """Convierte **bold**, `code` inline a tags reportlab."""
    text = escape_xml(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="9">\1</font>', text)
    return text

# ── Parser ─────────────────────────────────────────────────────────────────────
def parse_md(text):
    lines = text.splitlines()
    story = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Encabezados
        if line.startswith("#### "):
            story.append(Paragraph(inline_fmt(line[5:]), STYLES["h4"]))
            i += 1; continue
        if line.startswith("### "):
            story.append(Paragraph(inline_fmt(line[4:]), STYLES["h3"]))
            i += 1; continue
        if line.startswith("## "):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#c5cae9"), spaceAfter=2))
            story.append(Paragraph(inline_fmt(line[3:]), STYLES["h2"]))
            i += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(inline_fmt(line[2:]), STYLES["h1"]))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                    color=colors.HexColor("#1a237e"), spaceAfter=4))
            i += 1; continue

        # Bloque de código
        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code_text = "\n".join(code_lines)
            story.append(Preformatted(code_text, STYLES["code"]))
            story.append(Spacer(1, 4))
            i += 1; continue

        # Blockquote
        if line.startswith("> "):
            story.append(Paragraph(inline_fmt(line[2:]), STYLES["blockquote"]))
            i += 1; continue

        # Tabla Markdown
        if "|" in line and i + 1 < len(lines) and re.match(r"[\|\s\-:]+", lines[i+1]):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            # Filtrar línea separadora
            rows = []
            for tl in table_lines:
                if re.match(r"^[\|\s\-:]+$", tl.strip()):
                    continue
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                max_cols = max(len(r) for r in rows)
                for r in rows:
                    while len(r) < max_cols: r.append("")

                col_w = (A4[0] - 4*cm) / max_cols
                tbl = Table([[Paragraph(inline_fmt(c), ParagraphStyle(
                    "tc", parent=base["Normal"], fontSize=9, leading=12))
                    for c in row] for row in rows],
                    colWidths=[col_w]*max_cols,
                    repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#3949ab")),
                    ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                    ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                    ("FONTSIZE",   (0,0), (-1,0), 9),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1),
                     [colors.HexColor("#f3f4ff"), colors.white]),
                    ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#c5cae9")),
                    ("VALIGN",     (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING",(0,0), (-1,-1), 6),
                    ("RIGHTPADDING",(0,0),(-1,-1), 6),
                    ("TOPPADDING", (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue

        # HR
        if line.strip() == "---":
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#c5cae9"),
                                    spaceBefore=6, spaceAfter=6))
            i += 1; continue

        # Lista con viñetas
        if re.match(r"^[-*] ", line):
            story.append(Paragraph("• " + inline_fmt(line[2:]), STYLES["body"]))
            i += 1; continue

        # Lista numerada
        if re.match(r"^\d+\. ", line):
            m = re.match(r"^(\d+)\. (.*)", line)
            story.append(Paragraph(f"{m.group(1)}. " + inline_fmt(m.group(2)), STYLES["body"]))
            i += 1; continue

        # Línea vacía
        if line.strip() == "":
            story.append(Spacer(1, 4))
            i += 1; continue

        # Párrafo normal
        story.append(Paragraph(inline_fmt(line), STYLES["body"]))
        i += 1

    return story

# ── Generar PDF ────────────────────────────────────────────────────────────────
def main():
    text = MD_PATH.read_text(encoding="utf-8")
    story = parse_md(text)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="ADAPT-ECG Documentación Técnica",
        author="Elias Bejarano Lozada",
    )
    doc.build(story)
    print(f"PDF generado: {PDF_PATH}")

if __name__ == "__main__":
    main()
