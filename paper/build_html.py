"""Build a printable HTML preprint from RESEARCHGATE_MANUSCRIPT.md."""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "RESEARCHGATE_MANUSCRIPT.md"
OUT = ROOT / "RESEARCHGATE_MANUSCRIPT.html"

CSS = """
:root { color-scheme: light; }
body {
  font-family: "Times New Roman", Times, serif;
  font-size: 11.5pt;
  line-height: 1.45;
  max-width: 780px;
  margin: 2rem auto;
  padding: 0 1.25rem 3rem;
  color: #111;
}
h1 { font-size: 1.55rem; line-height: 1.25; margin-top: 0; }
h2 { font-size: 1.2rem; margin-top: 1.8rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }
h3 { font-size: 1.05rem; margin-top: 1.3rem; }
p, li { margin: 0.55rem 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.9rem 0 1.1rem;
  font-size: 10.5pt;
}
th, td {
  border: 1px solid #444;
  padding: 0.28rem 0.45rem;
  text-align: left;
  vertical-align: top;
}
th { background: #f3f3f3; }
code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em;
}
pre {
  background: #f6f6f6;
  border: 1px solid #ddd;
  padding: 0.75rem;
  overflow-x: auto;
}
.meta { color: #333; margin-bottom: 1.5rem; }
.math { font-style: italic; }
@media print {
  body { margin: 0; max-width: none; font-size: 10.5pt; }
  h2 { page-break-after: avoid; }
  table { page-break-inside: avoid; }
}
"""


def md_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"\\\(([^\)]+)\\\)", r'<span class="math">\1</span>', text)
    return text


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    in_ul = False
    in_table = False

    def close_lists():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            close_lists()
            close_table()
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(html.escape(line) + "\n")
            i += 1
            continue

        if line.strip() == "---":
            close_lists()
            close_table()
            out.append("<hr/>")
            i += 1
            continue

        if line.startswith("# "):
            close_lists()
            close_table()
            out.append(f"<h1>{md_inline(line[2:])}</h1>")
            i += 1
            continue
        if line.startswith("## "):
            close_lists()
            close_table()
            out.append(f"<h2>{md_inline(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("### "):
            close_lists()
            close_table()
            out.append(f"<h3>{md_inline(line[4:])}</h3>")
            i += 1
            continue

        if "|" in line and line.strip().startswith("|"):
            close_lists()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # separator row
            if all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                i += 1
                continue
            if not in_table:
                out.append("<table><thead><tr>")
                out.append("".join(f"<th>{md_inline(c)}</th>" for c in cells))
                out.append("</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in cells) + "</tr>")
            i += 1
            continue
        else:
            close_table()

        if line.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{md_inline(line[2:])}</li>")
            i += 1
            continue
        if re.match(r"^\d+\. ", line):
            # simple ordered list as paragraphs numbered already
            close_lists()
            out.append(f"<p>{md_inline(line)}</p>")
            i += 1
            continue

        if line.strip() == "":
            close_lists()
            i += 1
            continue

        # display math block
        if line.strip().startswith("\\["):
            close_lists()
            block = [line]
            while i + 1 < len(lines) and "\\]" not in lines[i]:
                i += 1
                block.append(lines[i])
            math = "\n".join(block)
            math = math.replace("\\[", "").replace("\\]", "").strip()
            out.append(f'<p class="math">{html.escape(math)}</p>')
            i += 1
            continue

        close_lists()
        # author block lines before first --- often bold-ish meta
        out.append(f"<p>{md_inline(line)}</p>")
        i += 1

    close_lists()
    close_table()
    return "\n".join(out)


def main() -> None:
    body = convert(SRC.read_text(encoding="utf-8"))
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>L1 vs L2 Unsupervised Anomaly Detection — Melike Kaya</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
