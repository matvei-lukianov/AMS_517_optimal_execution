# Cheridito, Dupret & Wu — ABIDES-MARL materials

**PDFs (repo root):**

- [2511.02016v1.pdf](../../2511.02016v1.pdf) — preprint *ABIDES-MARL: A Multi-Agent Reinforcement Learning Environment for Endogenous Price Formation and Execution in a Limit Order Book* (arXiv:2511.02016v1).
- [ABIDES_MARL_Slides.pdf](../../ABIDES_MARL_Slides.pdf) — slides (*Optimal Execution with Endogenous Liquidity*, seminar version).

**Machine-readable extracts** (auto-generated with `pypdf`):

- `2511.02016v1_extract.txt`
- `ABIDES_MARL_Slides_extract.txt`

**Distilled notes for this project:** see [abides_marl_methodology_notes.md](abides_marl_methodology_notes.md).

To regenerate extracts (after installing `pypdf` into `AMS517/.pdf_tools`):

```bash
cd /home/matvey/AMS517
PYTHONPATH=.pdf_tools python3.9 -c "
from pathlib import Path
from pypdf import PdfReader
root = Path('.')
out_dir = Path('literature/user_pdfs')
for name in ['2511.02016v1.pdf', 'ABIDES_MARL_Slides.pdf']:
    r = PdfReader(root / name)
    parts = []
    for i, p in enumerate(r.pages):
        parts.append(f'\\n\\n--- Page {i+1} ---\\n\\n' + (p.extract_text() or ''))
    (out_dir / name.replace('.pdf', '_extract.txt')).write_text(''.join(parts), encoding='utf-8')
"
```
