#!/usr/bin/env python3
"""Render blobpager/article/paper.md to a submission-grade PDF (WeasyPrint).

Mechanical conversion only: no content is retyped, reordered, or edited, so no
number can drift. Fidelity is then proven, not assumed: the script extracts the
PDF text (pypdf) and diffs number-token multisets against the source markdown,
exiting nonzero on any missing token.

Usage: ~/.venvs/pdfenv/bin/python blobpager/tools/make_pdf.py
"""
import re
import sys
from collections import Counter
from html import escape
from pathlib import Path

import markdown
from pypdf import PdfReader
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "blobpager/article/paper.md"
OUT = ROOT / "blobpager/article/paper.pdf"

RUNNING_HEAD = "BlobPager — Demand-Paged MoE Inference on a Single Consumer GPU"  # digit-free

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="author" content="Jimmy Popoola; ForgeAI">
<meta name="description" content="Demand-paged Mixture-of-Experts inference on a single consumer GPU (BlobPager, E0-E7b)">
<style>
@page {
  size: letter;
  margin: 24mm 21mm 26mm 21mm;
  @bottom-center {
    content: counter(page);
    font-family: 'DejaVu Serif';
    font-size: 8.5pt;
    color: #555;
  }
  @top-left {
    content: "__RUNHEAD__";
    font-family: 'DejaVu Serif';
    font-style: italic;
    font-size: 7.5pt;
    color: #666;
  }
}
@page :first { @top-left { content: none } }

html {
  font-family: 'DejaVu Serif';
  font-size: 10pt;
  line-height: 1.45;
  color: #111;
  font-variant-ligatures: none;
}
body { text-align: justify; hyphens: none; }

header .doctitle {
  font-size: 17pt;
  font-weight: bold;
  line-height: 1.25;
  text-align: center;
  margin: 0 6mm 10pt;
}
header .byline { text-align: center; font-size: 10pt; margin: 0 10mm 4pt; }
header .meta { text-align: center; font-size: 9pt; color: #333; margin: 0 10mm 14pt; }

h2 {
  font-size: 12.5pt;
  margin: 15pt 0 6pt;
  padding-bottom: 2pt;
  border-bottom: 0.7pt solid #888;
  break-after: avoid;
}
h3 { font-size: 11pt; margin: 11pt 0 4pt; break-after: avoid; }
p { margin: 4.5pt 0; }
ul, ol { margin: 4.5pt 0; padding-left: 16pt; }
li { margin: 2pt 0; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 7pt 0;
  font-size: 8.7pt;
  text-align: left;
}
th {
  border-bottom: 1.1pt solid #222;
  padding: 2.5pt 5pt;
  text-align: left;
  font-weight: bold;
}
td { padding: 2.3pt 5pt; border-bottom: 0.4pt solid #bbb; vertical-align: top; }
tr { break-inside: avoid; }
/* WeasyPrint repeats <thead> when a table breaks across pages; disable so the
   fidelity diff (source vs PDF tokens) stays strict and exception-free. */
thead { display: table-row-group; }

blockquote {
  margin: 7pt 10pt;
  padding: 2pt 9pt;
  border-left: 2.2pt solid #777;
  font-style: italic;
}
pre {
  font-family: 'DejaVu Sans Mono';
  font-size: 8pt;
  background: #f5f5f3;
  border: 0.5pt solid #ccc;
  padding: 6pt 8pt;
  white-space: pre-wrap;
  overflow-wrap: break-word;
  margin: 6pt 0;
}
code { font-family: 'DejaVu Sans Mono'; font-size: 88%; background: #f1f1ee; padding: 0 2pt; border-radius: 2pt; }
pre code { font-size: 100%; background: none; padding: 0; }
a { color: #17457f; text-decoration: none; }
hr { border: none; border-top: 0.6pt solid #999; margin: 10pt 0; }
p, li { orphans: 3; widows: 3; }
</style>
</head>
<body>
<header>
<div class="doctitle">__TITLE__</div>
<div class="byline">__BYLINE__</div>
<div class="meta">__META__</div>
</header>
__BODY__
</body>
</html>
"""

NUM = re.compile(r"\d[\d,]*(?:\.\d+)*")


def main() -> int:
    text = SRC.read_text()
    # Visible-to-visible fidelity: the PDF shows link labels but not link targets,
    # and renders code spans without backtick markup. Strip both from the source
    # side before counting so invisible characters cannot create false diffs.
    src_visible = re.sub(r"\]\([^)]*\)", "]", text).replace("`", "")
    lines = text.splitlines()

    title = next(l for l in lines if l.startswith("# "))[2:].strip()
    byline = next(l for i, l in enumerate(lines) if l.startswith("**Authors:**"))
    mi = next(i for i, l in enumerate(lines) if l.startswith("**Project:**"))
    meta_lines = [lines[mi]]
    j = mi + 1
    while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
        meta_lines.append(lines[j])
        j += 1
    meta = " ".join(meta_lines)

    hi = next(i for i, l in enumerate(lines) if l.startswith("## "))
    body_md = "\n".join(l for l in lines[hi:] if l.strip() != "---")

    def conv(s: str) -> str:
        return markdown.Markdown(extensions=["tables", "fenced_code"]).convert(s)

    html = (
        TEMPLATE.replace("__RUNHEAD__", RUNNING_HEAD)
        .replace("__TITLE__", escape(title))
        .replace("__BYLINE__", conv(byline))
        .replace("__META__", conv(meta))
        .replace("__BODY__", conv(body_md))
    )

    HTML(string=html, base_url=str(SRC.parent)).write_pdf(str(OUT))

    reader = PdfReader(str(OUT))
    pages = len(reader.pages)
    pdf_text = "\n".join((p.extract_text() or "") for p in reader.pages)

    # pypdf can insert a space between a digit and trailing punctuation at markup
    # boundaries ("48 ,"); strip trailing separators on BOTH sides so punctuation
    # adjacency cannot fake a diff, while any real digit change is still caught.
    cs = Counter(t.rstrip(".,") for t in NUM.findall(src_visible))
    co = Counter(t.rstrip(".,") for t in NUM.findall(pdf_text))
    missing = cs - co
    extra = co - cs
    # Footer page numbers (1..pages) are the only digits we add; exclude them.
    footer_extra = {k: v for k, v in extra.items() if k.isdigit() and 1 <= int(k) <= pages and v <= pages}
    for k, v in footer_extra.items():
        extra = extra - Counter({k: v})

    size = OUT.stat().st_size
    print(f"PDF written: {OUT.relative_to(ROOT)} | {pages} pages | {size:,} bytes")
    print(f"number tokens: source {sum(cs.values()):,} | pdf {sum(co.values()):,}")
    print(f"footer page-number tokens excluded from diff: {footer_extra}")
    print(f"MISSING from pdf ({sum(missing.values())}): {dict(missing)}")
    print(f"EXTRA in pdf beyond footer ({sum(extra.values())}): {dict(extra)}")
    if missing or extra:
        print("FIDELITY CHECK FAILED")
        return 1
    print("FIDELITY CHECK PASSED: every source number present in the PDF digit-for-digit")
    return 0


if __name__ == "__main__":
    sys.exit(main())