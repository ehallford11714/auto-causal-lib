"""Headless renderers for PDF, Markdown, HTML, and JSON report artifacts."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .models import (
    ChartSpec,
    ReportBundle,
    ReportCitation,
    ReportFact,
    ReportRenderError,
    ReportSection,
    ReportTable,
)


@dataclass
class RenderResult:
    path: Path
    format: str
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


_THEMES: dict[str, dict[str, str]] = {
    "high_contrast": {
        "primary": "#002B45",
        "secondary": "#005A78",
        "accent": "#C74700",
        "text": "#111111",
        "muted": "#4B5563",
        "background": "#FFFFFF",
        "table_header": "#002B45",
        "table_header_text": "#FFFFFF",
        "table_alt": "#EAF2F5",
        "warning_bg": "#FFF2CC",
        "danger": "#8B0000",
        "border": "#44515A",
    },
    "professional": {
        "primary": "#17324D",
        "secondary": "#2E5D73",
        "accent": "#8A4B08",
        "text": "#202124",
        "muted": "#5F6368",
        "background": "#FFFFFF",
        "table_header": "#17324D",
        "table_header_text": "#FFFFFF",
        "table_alt": "#F1F5F8",
        "warning_bg": "#FFF8E1",
        "danger": "#8B1E1E",
        "border": "#6B7780",
    },
    "monochrome": {
        "primary": "#000000",
        "secondary": "#333333",
        "accent": "#000000",
        "text": "#000000",
        "muted": "#444444",
        "background": "#FFFFFF",
        "table_header": "#000000",
        "table_header_text": "#FFFFFF",
        "table_alt": "#EEEEEE",
        "warning_bg": "#F2F2F2",
        "danger": "#000000",
        "border": "#333333",
    },
}


def _theme(name: str) -> dict[str, str]:
    return dict(_THEMES.get(name, _THEMES["high_contrast"]))


def _text(value: Any, *, limit: int = 1200) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    else:
        rendered = str(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _esc(value: Any, *, limit: int = 1200) -> str:
    return html.escape(_text(value, limit=limit), quote=False).replace("\n", "<br/>")


def render_markdown(bundle: ReportBundle, output: str | Path) -> RenderResult:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.to_markdown(), encoding="utf-8")
    return RenderResult(path=path, format="markdown")


def render_json(bundle: ReportBundle, output: str | Path) -> RenderResult:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.to_json(), encoding="utf-8")
    return RenderResult(path=path, format="json")


def _html_fact(fact: ReportFact) -> str:
    status = (
        '<span class="audit-only">audit only / excluded</span>'
        if not fact.evidence_eligible
        else ""
    )
    citations = " ".join(
        f'<a href="#citation-{html.escape(citation_id)}">[{html.escape(citation_id)}]</a>'
        for citation_id in fact.citation_ids
    )
    return (
        '<div class="fact">'
        f'<div class="fact-label">{html.escape(fact.label)}</div>'
        f'<div class="fact-value">{_esc(fact.value)} {status}</div>'
        f'<div class="provenance">Provenance: '
        f'<code>{html.escape(fact.provenance_id)}</code> {citations}</div>'
        "</div>"
    )


def _html_table(table: ReportTable) -> str:
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in table.columns)
    rows = []
    for row in table.rows:
        cells = "".join(f"<td>{_esc(row.get(column), limit=500)}</td>" for column in table.columns)
        rows.append(f"<tr>{cells}</tr>")
    footnote = f"<p class='footnote'>{html.escape(table.footnote)}</p>" if table.footnote else ""
    provenance = (
        "<p class='provenance'>Provenance: "
        + ", ".join(f"<code>{html.escape(item)}</code>" for item in table.provenance_ids)
        + "</p>"
        if table.provenance_ids
        else ""
    )
    return (
        f"<h3>{html.escape(table.title)}</h3>"
        f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{footnote}{provenance}"
    )


def _html_chart(chart: ChartSpec, warnings: list[str]) -> str:
    body = ""
    if chart.image_path:
        image = Path(chart.image_path)
        if image.is_file():
            body = (
                f'<img src="{html.escape(str(image))}" '
                f'alt="{html.escape(chart.alt_text)}"/>'
            )
        else:
            warnings.append(
                f"Chart `{chart.id}` image not found; rendered specification fallback."
            )
    if not body:
        body = (
            "<p class='chart-fallback'>Image unavailable; retained chart specification.</p>"
            f"<pre>{html.escape(json.dumps(chart.spec, indent=2, sort_keys=True, default=str))}</pre>"
        )
    return (
        f"<figure><h3>{html.escape(chart.title)}</h3>{body}"
        f"<figcaption>{html.escape(chart.caption)} "
        f"<strong>Alt text:</strong> {html.escape(chart.alt_text)}</figcaption></figure>"
    )


def _html_citation(citation: ReportCitation) -> str:
    authors = ", ".join(citation.authors)
    title = citation.title or citation.id
    url = (
        f' <a href="{html.escape(citation.url)}">{html.escape(citation.url)}</a>'
        if citation.url
        else ""
    )
    status = "verified" if citation.verified else "unverified"
    return (
        f'<li id="citation-{html.escape(citation.id)}">'
        f"[{html.escape(citation.id)}] {html.escape(authors)} "
        f"<em>{html.escape(title)}</em> {html.escape(citation.year)}{url} "
        f"({status})</li>"
    )


def render_html(bundle: ReportBundle, output: str | Path) -> RenderResult:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = _theme(bundle.policy.theme)
    warnings: list[str] = []
    toc = "".join(
        f'<li><a href="#{html.escape(section.id)}">{html.escape(section.heading)}</a></li>'
        for section in bundle.sections
        if section.id != "cover"
    )
    section_html = []
    for section in bundle.sections:
        if section.id == "cover":
            continue
        claims = "".join(
            "<li>"
            + html.escape(claim.text)
            + " <code>["
            + html.escape(", ".join(claim.fact_ids))
            + "]</code></li>"
            for claim in section.claims
        )
        facts = "".join(_html_fact(fact) for fact in section.facts)
        tables = "".join(_html_table(table) for table in section.tables)
        charts = "".join(_html_chart(chart, warnings) for chart in section.charts)
        caveats = (
            "<aside class='caveats'><h3>Caveats</h3><ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in section.caveats)
            + "</ul></aside>"
            if section.caveats
            else ""
        )
        audit = (
            "<details><summary>Section audit</summary><ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in section.audit_notes)
            + "</ul></details>"
            if section.audit_notes
            else ""
        )
        section_html.append(
            f'<section id="{html.escape(section.id)}">'
            f"<h2>{html.escape(section.heading)}</h2>"
            f"<p>{html.escape(section.summary)}</p>"
            + (f"<ul class='claims'>{claims}</ul>" if claims else "")
            + facts
            + tables
            + charts
            + caveats
            + audit
            + "</section>"
        )
    references = (
        "<section id='references'><h2>References</h2><ol>"
        + "".join(_html_citation(citation) for citation in bundle.citations)
        + "</ol></section>"
        if bundle.citations
        else ""
    )
    audit_items = list(bundle.audit_notes) + warnings
    audit = (
        "<section id='report-audit'><h2>Report audit</h2><ul>"
        + "".join(f"<li>{html.escape(item)}</li>" for item in audit_items)
        + "</ul></section>"
        if bundle.policy.include_audit_notes and audit_items
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(bundle.plan.title)}</title>
<style>
:root {{
  --primary: {colors['primary']}; --secondary: {colors['secondary']};
  --accent: {colors['accent']}; --text: {colors['text']};
  --muted: {colors['muted']}; --alt: {colors['table_alt']};
  --warning: {colors['warning_bg']}; --danger: {colors['danger']};
}}
body {{ color: var(--text); background: #fff; font: 16px/1.55 Arial, sans-serif;
  margin: 0 auto; max-width: 1100px; padding: 2.5rem; }}
h1, h2, h3 {{ color: var(--primary); line-height: 1.2; }}
h1 {{ font-size: 2.4rem; }} h2 {{ border-bottom: 2px solid var(--secondary); padding-bottom: .25rem; }}
code, pre {{ font-family: Consolas, monospace; }} pre {{ overflow-x: auto; background: var(--alt); padding: 1rem; }}
.cover {{ min-height: 55vh; border-left: 8px solid var(--accent); padding: 3rem; }}
.meta, .provenance, .footnote {{ color: var(--muted); font-size: .88rem; }}
.fact {{ border-left: 4px solid var(--secondary); margin: .8rem 0; padding: .6rem 1rem; background: var(--alt); }}
.fact-label {{ font-weight: 700; }} .audit-only {{ color: var(--danger); font-weight: 700; }}
.table-wrap {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; }}
th {{ color: #fff; background: var(--primary); }} th, td {{ border: 1px solid #65737e; padding: .45rem; text-align: left; }}
tbody tr:nth-child(even) {{ background: var(--alt); }}
.caveats {{ background: var(--warning); border: 2px solid var(--accent); padding: 1rem; margin: 1rem 0; }}
figure {{ margin: 1.5rem 0; }} img {{ max-width: 100%; height: auto; }} figcaption {{ color: var(--muted); }}
a {{ color: var(--secondary); }} section {{ margin: 2.5rem 0; }}
@media print {{ body {{ max-width: none; padding: 0; }} section {{ break-inside: auto; }} h2 {{ break-after: avoid; }} }}
</style>
</head>
<body>
<header class="cover">
  <h1>{html.escape(bundle.plan.title)}</h1>
  <p><strong>Audience:</strong> {html.escape(bundle.plan.audience)}</p>
  <p><strong>Purpose:</strong> {html.escape(bundle.plan.purpose)}</p>
  <p class="meta"><strong>Generated:</strong> {html.escape(bundle.generated_at)}<br/>
  <strong>Director:</strong> {html.escape(bundle.plan.director_backend)}<br/>
  <strong>Policy:</strong> {html.escape(bundle.policy.profile)}</p>
</header>
<nav aria-label="Table of contents"><h2>Contents</h2><ol>{toc}</ol></nav>
{''.join(section_html)}
{references}
{audit}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return RenderResult(path=path, format="html", warnings=warnings)


def render_pdf(bundle: ReportBundle, output: str | Path) -> RenderResult:
    """Render a headless ReportLab PDF with TOC, bookmarks, and page numbers."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4, LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            BaseDocTemplate,
            Flowable,
            Frame,
            Image,
            KeepTogether,
            LongTable,
            PageBreak,
            PageTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.platypus.tableofcontents import TableOfContents
    except Exception as exc:
        raise ReportRenderError(
            "PDF output requires reportlab. Install `reportlab>=4.0` or the "
            "AutoCausal reporting extra."
        ) from exc

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = _theme(bundle.policy.theme)
    page_size = LETTER if bundle.policy.page_size == "letter" else A4
    page_width, page_height = page_size
    left_margin = 0.68 * inch
    right_margin = 0.68 * inch
    top_margin = 0.72 * inch
    bottom_margin = 0.62 * inch
    usable_width = page_width - left_margin - right_margin
    warnings: list[str] = []

    class ReportDocTemplate(BaseDocTemplate):
        def __init__(self, filename: str) -> None:
            super().__init__(
                filename,
                pagesize=page_size,
                leftMargin=left_margin,
                rightMargin=right_margin,
                topMargin=top_margin,
                bottomMargin=bottom_margin,
                title=bundle.plan.title,
                author="AutoCausalLib",
                subject=bundle.plan.purpose,
                creator="AutoCausalLib ReportEngine (ReportLab)",
            )
            frame = Frame(
                self.leftMargin,
                self.bottomMargin,
                self.width,
                self.height,
                id="normal",
            )
            self.addPageTemplates(
                [
                    PageTemplate(
                        id="report",
                        frames=[frame],
                        onPage=self._on_page,
                    )
                ]
            )

        def _on_page(self, canvas: Any, doc: Any) -> None:
            canvas.saveState()
            canvas.setTitle(bundle.plan.title)
            canvas.setAuthor("AutoCausalLib")
            canvas.setSubject(bundle.plan.purpose)
            canvas.setCreator("AutoCausalLib ReportEngine (ReportLab)")
            try:
                canvas.setKeywords(
                    "AutoCausal, causal analysis, provenance, report, "
                    + ", ".join(bundle.run_ids)
                )
            except Exception:
                pass
            canvas.setStrokeColor(colors.HexColor(palette["border"]))
            canvas.setLineWidth(0.45)
            canvas.line(
                self.leftMargin,
                page_height - 0.43 * inch,
                page_width - self.rightMargin,
                page_height - 0.43 * inch,
            )
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor(palette["muted"]))
            canvas.drawString(
                self.leftMargin,
                page_height - 0.32 * inch,
                bundle.plan.title[:85],
            )
            canvas.drawRightString(
                page_width - self.rightMargin,
                0.32 * inch,
                f"Page {doc.page}",
            )
            canvas.restoreState()

        def afterFlowable(self, flowable: Any) -> None:
            if not isinstance(flowable, Paragraph):
                return
            section_id = getattr(flowable, "_report_section_id", "")
            if not section_id:
                return
            text = str(getattr(flowable, "_report_heading", flowable.getPlainText()))
            key = f"section-{section_id}"
            self.canv.bookmarkPage(key)
            try:
                self.canv.addOutlineEntry(text, key, level=0, closed=False)
            except Exception:
                pass
            self.notify("TOCEntry", (0, text, self.page, key))

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=33,
        textColor=colors.HexColor(palette["primary"]),
        alignment=TA_LEFT,
        spaceAfter=18,
    )
    cover_label = ParagraphStyle(
        "CoverLabel",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor(palette["muted"]),
        spaceAfter=7,
    )
    banner_style = ParagraphStyle(
        "EpistemicBanner",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(palette["danger"]),
        backColor=colors.HexColor(palette["warning_bg"]),
        borderColor=colors.HexColor(palette["accent"]),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=12,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "ReportSectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor(palette["primary"]),
        spaceBefore=8,
        spaceAfter=10,
        keepWithNext=True,
    )
    subheading_style = ParagraphStyle(
        "ReportSubheading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor(palette["secondary"]),
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.2,
        textColor=colors.HexColor(palette["text"]),
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "ReportSmall",
        parent=body_style,
        fontSize=7.7,
        leading=10,
        textColor=colors.HexColor(palette["muted"]),
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-8,
        bulletIndent=5,
        spaceAfter=3,
    )
    caveat_style = ParagraphStyle(
        "ReportCaveat",
        parent=body_style,
        fontSize=8.7,
        leading=12,
        textColor=colors.HexColor(palette["danger"]),
        backColor=colors.HexColor(palette["warning_bg"]),
        borderColor=colors.HexColor(palette["accent"]),
        borderWidth=0.8,
        borderPadding=6,
        spaceBefore=5,
        spaceAfter=5,
    )
    mono_style = ParagraphStyle(
        "ReportMono",
        parent=small_style,
        fontName="Courier",
        wordWrap="CJK",
        backColor=colors.HexColor(palette["table_alt"]),
        borderPadding=5,
    )
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1",
            parent=body_style,
            fontSize=10,
            leading=14,
            leftIndent=12,
            firstLineIndent=-12,
            textColor=colors.HexColor(palette["secondary"]),
        )
    ]

    def heading(text: str, section_id: str) -> Any:
        paragraph = Paragraph(html.escape(text), heading_style)
        paragraph._report_section_id = section_id
        paragraph._report_heading = text
        return paragraph

    def fact_flowable(fact: ReportFact) -> Any:
        status = (
            "<br/><font color='%s'><b>AUDIT ONLY / EXCLUDED FROM EVIDENCE</b></font>"
            % palette["danger"]
            if not fact.evidence_eligible
            else ""
        )
        citations = (
            "<br/>Citations: "
            + ", ".join(html.escape(item) for item in fact.citation_ids)
            if fact.citation_ids
            else ""
        )
        left = Paragraph(f"<b>{html.escape(fact.label)}</b>", body_style)
        right = Paragraph(
            f"{_esc(fact.value)}{status}<br/>"
            f"<font color='{palette['muted']}'>Provenance: "
            f"{html.escape(fact.provenance_id)}{citations}</font>",
            body_style,
        )
        table = Table(
            [[left, right]],
            colWidths=[usable_width * 0.28, usable_width * 0.72],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(palette["table_alt"])),
                    ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor(palette["border"])),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(palette["secondary"])),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def table_flowables(table: ReportTable) -> list[Any]:
        flows: list[Any] = [
            Paragraph(html.escape(table.title), subheading_style)
        ]
        if not table.rows:
            flows.append(Paragraph("No rows.", body_style))
            return flows
        columns = table.columns
        font_size = 6.4 if len(columns) >= 7 else 7.3
        cell_style = ParagraphStyle(
            f"TableCell{len(columns)}",
            parent=small_style,
            fontSize=font_size,
            leading=font_size + 2.2,
            textColor=colors.HexColor(palette["text"]),
            wordWrap="CJK",
        )
        header_style = ParagraphStyle(
            f"TableHeader{len(columns)}",
            parent=cell_style,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor(palette["table_header_text"]),
            alignment=TA_CENTER,
        )
        data = [
            [Paragraph(html.escape(column), header_style) for column in columns]
        ]
        for row in table.rows:
            data.append(
                [
                    Paragraph(_esc(row.get(column), limit=600), cell_style)
                    for column in columns
                ]
            )
        weights = []
        for column in columns:
            lower = column.lower()
            if any(
                token in lower
                for token in ("detail", "message", "reason", "recommendation", "finding")
            ):
                weights.append(2.1)
            elif any(token in lower for token in ("fact_id", "provenance", "citation")):
                weights.append(1.5)
            else:
                weights.append(1.0)
        total_weight = sum(weights)
        widths = [usable_width * weight / total_weight for weight in weights]
        report_table = LongTable(
            data,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
            splitByRow=1,
        )
        commands: list[tuple[Any, ...]] = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(palette["table_header"])),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(palette["table_header_text"])),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(palette["border"])),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for row_index in range(2, len(data), 2):
            commands.append(
                (
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    colors.HexColor(palette["table_alt"]),
                )
            )
        report_table.setStyle(TableStyle(commands))
        flows.append(report_table)
        if table.footnote:
            flows.append(Paragraph(f"<i>{html.escape(table.footnote)}</i>", small_style))
        if table.provenance_ids:
            flows.append(
                Paragraph(
                    "Provenance: "
                    + html.escape(", ".join(table.provenance_ids)),
                    small_style,
                )
            )
        flows.append(Spacer(1, 5))
        return flows

    def chart_flowables(chart: ChartSpec) -> list[Any]:
        flows: list[Any] = [
            Paragraph(html.escape(chart.title), subheading_style)
        ]
        rendered = False
        if chart.image_path:
            image_path = Path(chart.image_path)
            if image_path.is_file():
                suffix = image_path.suffix.lower()
                if suffix == ".svg":
                    try:
                        from svglib.svglib import svg2rlg  # type: ignore

                        drawing = svg2rlg(str(image_path))
                        if drawing is None:
                            raise ValueError("svg2rlg returned no drawing")
                        scale = min(
                            usable_width / max(float(drawing.width), 1.0),
                            4.6 * inch / max(float(drawing.height), 1.0),
                            1.0,
                        )
                        drawing.width *= scale
                        drawing.height *= scale
                        drawing.scale(scale, scale)
                        flows.append(drawing)
                        rendered = True
                    except Exception as exc:
                        warnings.append(
                            f"Chart `{chart.id}` SVG rendering unavailable "
                            f"({type(exc).__name__}); specification fallback used."
                        )
                else:
                    try:
                        image = Image(str(image_path))
                        scale = min(
                            usable_width / max(float(image.imageWidth), 1.0),
                            4.6 * inch / max(float(image.imageHeight), 1.0),
                            1.0,
                        )
                        image.drawWidth = float(image.imageWidth) * scale
                        image.drawHeight = float(image.imageHeight) * scale
                        image.hAlign = "CENTER"
                        flows.append(image)
                        rendered = True
                    except Exception as exc:
                        warnings.append(
                            f"Chart `{chart.id}` image could not be embedded "
                            f"({type(exc).__name__}); specification fallback used."
                        )
            else:
                warnings.append(
                    f"Chart `{chart.id}` image path does not exist; "
                    "specification fallback used."
                )
        if not rendered and chart.runtime_artifact is not None:
            try:
                from io import BytesIO

                buffer = BytesIO()
                artifact = chart.runtime_artifact
                if hasattr(artifact, "savefig"):
                    artifact.savefig(
                        buffer,
                        format="png",
                        bbox_inches="tight",
                        metadata={"Description": chart.alt_text},
                    )
                    image_bytes = buffer.getvalue()
                elif hasattr(artifact, "to_image"):
                    # Plotly static export uses kaleido when available.
                    image_bytes = artifact.to_image(format="png")
                    buffer = BytesIO(image_bytes)
                elif isinstance(artifact, (bytes, bytearray)):
                    image_bytes = bytes(artifact)
                    buffer = BytesIO(image_bytes)
                else:
                    raise TypeError(
                        f"unsupported runtime chart artifact {type(artifact).__name__}"
                    )
                buffer.seek(0)
                image = Image(buffer)
                image._autocausal_buffer = buffer
                scale = min(
                    usable_width / max(float(image.imageWidth), 1.0),
                    4.6 * inch / max(float(image.imageHeight), 1.0),
                    1.0,
                )
                image.drawWidth = float(image.imageWidth) * scale
                image.drawHeight = float(image.imageHeight) * scale
                image.hAlign = "CENTER"
                flows.append(image)
                rendered = True
            except Exception as exc:
                warnings.append(
                    f"Chart `{chart.id}` runtime export unavailable "
                    f"({type(exc).__name__}); install kaleido for Plotly static "
                    "images or save PNG/SVG explicitly."
                )
        if not rendered:
            warnings.append(
                f"Chart `{chart.id}` has no renderable image; chart spec/table retained."
            )
            flows.append(
                Paragraph(
                    "<b>Chart image unavailable.</b> The validated specification "
                    "and source references are retained below.",
                    caveat_style,
                )
            )
            spec_text = json.dumps(
                chart.spec,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            flows.append(Paragraph(_esc(spec_text, limit=2800), mono_style))
            if chart.source_fact_ids:
                flows.append(
                    Paragraph(
                        "Source facts: " + html.escape(", ".join(chart.source_fact_ids)),
                        small_style,
                    )
                )
            if chart.source_table_id:
                flows.append(
                    Paragraph(
                        "Source table: " + html.escape(chart.source_table_id),
                        small_style,
                    )
                )
        caption = chart.caption or "Visualization over normalized evidence."
        flows.append(
            Paragraph(
                f"<i>{html.escape(caption)}</i><br/>"
                f"<b>Alt text:</b> {html.escape(chart.alt_text)}",
                small_style,
            )
        )
        flows.append(Spacer(1, 7))
        return flows

    def citation_flowable(citation: ReportCitation) -> Any:
        authors = ", ".join(citation.authors)
        status = "verified" if citation.verified else "unverified"
        parts = [
            f"[{citation.id}]",
            authors,
            citation.title,
            citation.year,
            citation.url,
            f"({status})",
        ]
        return Paragraph(
            " ".join(html.escape(part) for part in parts if part),
            bullet_style,
            bulletText="•",
        )

    story: list[Any] = []
    story.extend(
        [
            Spacer(1, 0.65 * inch),
            Paragraph(html.escape(bundle.plan.title), title_style),
            Spacer(1, 0.08 * inch),
            Paragraph(
                f"<b>Audience:</b> {html.escape(bundle.plan.audience)}",
                cover_label,
            ),
            Paragraph(
                f"<b>Purpose:</b> {html.escape(bundle.plan.purpose)}",
                cover_label,
            ),
            Paragraph(
                f"<b>Generated:</b> {html.escape(bundle.generated_at)}",
                cover_label,
            ),
            Paragraph(
                f"<b>Report policy:</b> {html.escape(bundle.policy.profile)} "
                f"({html.escape(bundle.policy.template)} / "
                f"{html.escape(bundle.policy.theme)})",
                cover_label,
            ),
            Paragraph(
                f"<b>Director:</b> {html.escape(bundle.plan.director_backend)}",
                cover_label,
            ),
            Paragraph(
                "EPISTEMIC: AutoCausal outputs are exploratory assistance. "
                "Associations are not causal effects; predictive metrics are "
                "separate from causal estimates; synthetic instruments are "
                "excluded from production evidence.",
                banner_style,
            ),
            Spacer(1, 0.35 * inch),
            Paragraph(
                "This report contains normalized aggregate evidence and provenance "
                "references. Raw rows, raw frames, sample values, and secrets are "
                "not included.",
                body_style,
            ),
            PageBreak(),
            Paragraph("Contents", heading_style),
            Spacer(1, 8),
            toc,
            PageBreak(),
        ]
    )

    for section in bundle.sections:
        if section.id == "cover":
            continue
        story.append(heading(section.heading, section.id))
        if section.summary:
            story.append(Paragraph(html.escape(section.summary), body_style))
        if section.narrative_is_slm:
            story.append(
                Paragraph(
                    "SLM-GENERATED NARRATIVE: every statement below is constrained "
                    "to listed normalized fact/provenance ids.",
                    caveat_style,
                )
            )
        for claim in section.claims:
            references = ", ".join(claim.fact_ids)
            story.append(
                Paragraph(
                    f"{html.escape(claim.text)} "
                    f"<font color='{palette['muted']}'>"
                    f"[facts: {html.escape(references)}]</font>",
                    bullet_style,
                    bulletText="•",
                )
            )
        if section.claims:
            story.append(Spacer(1, 4))
        for fact in section.facts:
            story.extend([fact_flowable(fact), Spacer(1, 4)])
        for table in section.tables:
            story.extend(table_flowables(table))
        for chart in section.charts:
            story.extend(chart_flowables(chart))
        if section.caveats:
            story.append(Paragraph("Caveats", subheading_style))
            for caveat in section.caveats:
                story.append(Paragraph(html.escape(caveat), caveat_style))
        if section.provenance_references:
            story.append(
                Paragraph(
                    "<b>Section provenance:</b> "
                    + html.escape(", ".join(section.provenance_references)),
                    small_style,
                )
            )
        if section.audit_notes:
            story.append(Paragraph("Section audit", subheading_style))
            for note in section.audit_notes:
                story.append(
                    Paragraph(html.escape(note), bullet_style, bulletText="•")
                )
        story.append(Spacer(1, 11))

    if bundle.citations:
        story.append(heading("References", "references"))
        for citation in bundle.citations:
            story.append(citation_flowable(citation))
        story.append(Spacer(1, 8))

    if bundle.policy.include_audit_notes:
        story.append(heading("Report audit", "report-audit"))
        for note in list(bundle.audit_notes) + warnings:
            story.append(Paragraph(html.escape(note), bullet_style, bulletText="•"))

    document = ReportDocTemplate(str(path))
    try:
        document.multiBuild(story)
    except Exception as exc:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise ReportRenderError(
            f"ReportLab failed to build PDF: {type(exc).__name__}: {exc}"
        ) from exc
    page_count = int(getattr(document, "page", 0) or 0)
    if page_count > bundle.policy.max_pages:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise ReportRenderError(
            f"Rendered PDF has {page_count} pages, exceeding policy maximum "
            f"{bundle.policy.max_pages}."
        )
    try:
        prefix = path.read_bytes()[:5]
    except Exception as exc:
        raise ReportRenderError(f"Could not verify rendered PDF: {exc}") from exc
    if prefix != b"%PDF-":
        raise ReportRenderError("ReportLab output failed PDF signature validation")
    return RenderResult(
        path=path,
        format="pdf",
        page_count=page_count,
        warnings=list(dict.fromkeys(warnings)),
    )


RENDERERS: dict[str, Callable[[ReportBundle, str | Path], RenderResult]] = {
    "pdf": render_pdf,
    "markdown": render_markdown,
    "md": render_markdown,
    "html": render_html,
    "json": render_json,
}


def render_bundle(
    bundle: ReportBundle,
    output: str | Path,
    *,
    format: str | None = None,
) -> RenderResult:
    path = Path(output)
    selected = (format or path.suffix.lstrip(".") or "pdf").lower()
    if selected == "htm":
        selected = "html"
    if selected not in RENDERERS:
        raise ReportRenderError(
            f"Unsupported report format `{selected}`; use pdf, markdown, html, or json."
        )
    return RENDERERS[selected](bundle, path)


__all__ = [
    "RENDERERS",
    "RenderResult",
    "render_bundle",
    "render_html",
    "render_json",
    "render_markdown",
    "render_pdf",
]
