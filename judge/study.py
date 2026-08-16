"""Reading the week's study material off disk and turning it into page content.

Nothing here is duplicated from the repo — the week README, the papers and the notebook
*are* the repo's `weekNN_*/` folder, served in place. Editing the README updates the site,
and there is no second copy to drift.

Everything is served locally rather than linked out to GitHub: the papers are already in
the folder, so making a reader leave the site to read them was a pointless round trip.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Only these extensions are ever served out of a papers/ directory.
SERVABLE = {".pdf": "application/pdf", ".md": "text/markdown"}


def week_folder(week: int) -> Path | None:
    hits = sorted(REPO.glob(f"week{week:02d}_*"))
    return hits[0] if hits else None


def _titles_from_index(idx: Path) -> dict[str, tuple[str, str]]:
    """Map filename -> (title, role) by parsing the generated papers/README.md.

    The index is machine-written by tools/fetch_papers.py, so this parse is against a
    known shape rather than arbitrary markdown. Anything it fails to match still shows
    up in the listing — the directory is the source of truth, this only adds titles.
    """
    out: dict[str, tuple[str, str]] = {}
    if not idx.exists():
        return out
    role = "supplement"
    for block in idx.read_text().split("\n"):
        low = block.strip().lower()
        if low.startswith("## spine"):
            role = "spine"
        elif low.startswith("## supplementary"):
            role = "supplement"
    # Titles and filenames come in pairs across two lines; match them together.
    text = idx.read_text()
    spine_at = text.lower().find("## spine")
    supp_at = text.lower().find("## supplementary")
    for m in re.finditer(r"- \*\*(.+?)\*\*\s*\n\s*(?:\[`([^`]+)`\]|`([^`]+)`)", text):
        fn = m.group(2) or m.group(3)
        role = "spine" if (spine_at != -1 and (supp_at == -1 or m.start() < supp_at)
                           and m.start() > spine_at) else "supplement"
        out[fn] = (" ".join(m.group(1).split()), role)
    return out


def papers(week: int) -> list[dict]:
    """Every readable file in the week's papers/ folder, titled where possible."""
    d = week_folder(week)
    if d is None:
        return []
    pdir = d / "papers"
    if not pdir.is_dir():
        return []
    titles = _titles_from_index(pdir / "README.md")
    out = []
    for f in sorted(pdir.iterdir()):
        if f.name == "README.md" or f.suffix.lower() not in SERVABLE:
            continue
        title, role = titles.get(f.name, (f.stem.replace("-", " "), "supplement"))
        out.append({
            "file": f.name,
            "title": title,
            "role": role,
            "kind": f.suffix.lower().lstrip("."),
            "size": f"{f.stat().st_size / 1e6:.1f} MB",
        })
    # Spine first, then alphabetical — the reading order the plan asks for.
    out.sort(key=lambda p: (p["role"] != "spine", p["title"].lower()))

    # Papers the index names but that never downloaded stay visible, marked, so the gap
    # is legible instead of silently absent.
    present = {p["file"] for p in out}
    for fn, (title, role) in titles.items():
        if fn not in present:
            out.append({"file": fn, "title": title, "role": role,
                        "kind": Path(fn).suffix.lstrip("."), "size": "", "missing": True})
    return out


def resolve_paper(week: int, name: str) -> Path | None:
    """Resolve a requested paper to a real file, or None.

    Traversal-proof by construction: the name is reduced to its basename, the extension
    must be in the allow-list, and the resolved path must still sit inside the week's
    papers directory.
    """
    d = week_folder(week)
    if d is None:
        return None
    pdir = (d / "papers").resolve()
    safe = Path(name).name
    if Path(safe).suffix.lower() not in SERVABLE:
        return None
    # The generated index is rendered by the study page itself; it is not a paper, and
    # listing and serving should agree on what exists.
    if safe.lower() == "readme.md":
        return None
    target = (pdir / safe).resolve()
    if not str(target).startswith(str(pdir) + "/") or not target.is_file():
        return None
    return target


def readme(week: int) -> str:
    d = week_folder(week)
    if d is None:
        return ""
    f = d / "README.md"
    return f.read_text() if f.exists() else ""


def notebook(week: int) -> dict:
    """Parse the week's notebook into renderable cells.

    Outputs are stripped in this repo (`make clean-nb`), so there is nothing to render
    but the source — which is the part worth reading anyway, since the notebooks ship
    scaffolded rather than solved.
    """
    d = week_folder(week)
    if d is None:
        return {}
    nb_path = d / f"{d.name}.ipynb"
    if not nb_path.exists():
        return {}
    try:
        nb = json.loads(nb_path.read_text())
    except (ValueError, OSError):
        return {"name": nb_path.name, "cells": [], "error": "could not parse"}

    cells = []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if not src.strip():
            continue
        cells.append({"type": c.get("cell_type", "code"), "source": src})
    return {"name": nb_path.name, "cells": cells,
            "n_code": sum(1 for c in cells if c["type"] == "code")}


def summary(week: int) -> dict:
    """Everything the study pages need, in one call."""
    d = week_folder(week)
    if d is None:
        return {}
    ps = papers(week)
    return {
        "folder": d.name,
        "papers": ps,
        "n_papers": sum(1 for p in ps if not p.get("missing")),
        "readme": readme(week),
        "notebook": notebook(week),
    }
