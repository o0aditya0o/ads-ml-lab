#!/usr/bin/env python3
"""Download the reading spine into each week's ``papers/`` folder.

Every entry was resolved against a live source (arXiv API, Semantic Scholar, or a
publisher host) rather than guessed -- several URLs that circulate in blog posts are
dead. Notably ``olivier.chapelle.cc`` has lapsed and the domain now serves an unrelated
content farm, so the Chapelle delayed-feedback paper is fetched from the Wayback Machine
and the repo ships open-access substitutes for that week regardless.

Each download is validated (HTTP 200 + ``%PDF`` magic bytes) before it is kept, so a
truncated or HTML-error-page download never masquerades as a paper. Re-running skips
files that are already present and valid.

Usage
-----
    python tools/fetch_papers.py              # all weeks
    python tools/fetch_papers.py --week 4     # one week
    python tools/fetch_papers.py --check      # verify what's on disk, download nothing
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ads-ml-lab/1.0"

WEEK_DIRS = {int(p.name[4:6]): p for p in sorted(REPO.glob("week[0-9][0-9]_*"))}


def arxiv(aid: str) -> str:
    return f"https://arxiv.org/pdf/{aid}"


# (week, filename, title, url, role)
#   role: "spine"      -- the one paper the plan says to read that week
#         "supplement" -- read if you have time / needed for a specific technique
PAPERS: list[tuple[int, str, str, str, str]] = [
    # ---- Week 1: baselines -------------------------------------------------------
    (1, "mcmahan2013-ftrl-view-from-the-trenches.pdf",
     "Ad Click Prediction: a View from the Trenches (McMahan et al., KDD 2013)",
     "https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf", "spine"),
    (1, "he2014-practical-lessons-facebook.pdf",
     "Practical Lessons from Predicting Clicks on Ads at Facebook (He et al., ADKDD 2014)",
     "https://quinonero.net/Publications/predicting-clicks-facebook.pdf", "supplement"),

    # ---- Week 2: deep CVR --------------------------------------------------------
    (2, "naumov2019-dlrm.pdf",
     "Deep Learning Recommendation Model for Personalization and Recommendation Systems",
     arxiv("1906.00091"), "spine"),
    (2, "cheng2016-wide-and-deep.pdf",
     "Wide & Deep Learning for Recommender Systems", arxiv("1606.07792"), "spine"),
    (2, "rendle2010-factorization-machines.pdf",
     "Factorization Machines (Rendle, ICDM 2010)",
     "https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf", "supplement"),
    (2, "guo2017-deepfm.pdf", "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction",
     arxiv("1703.04247"), "supplement"),
    (2, "wang2017-deep-and-cross.pdf", "Deep & Cross Network for Ad Click Predictions",
     arxiv("1708.05123"), "supplement"),
    (2, "shwartzziv2021-tabular-dl-is-not-all-you-need.pdf",
     "Tabular Data: Deep Learning is Not All You Need -- the trees-vs-deep evidence base",
     arxiv("2106.03253"), "supplement"),
    (2, "gorishniy2021-revisiting-tabular-dl.pdf",
     "Revisiting Deep Learning Models for Tabular Data", arxiv("2106.11959"), "supplement"),

    # ---- Week 3: calibration -----------------------------------------------------
    (3, "guo2017-on-calibration-of-modern-nns.pdf",
     "On Calibration of Modern Neural Networks", arxiv("1706.04599"), "spine"),
    (3, "kumar2019-verified-uncertainty-calibration.pdf",
     "Verified Uncertainty Calibration -- why naive binned ECE is biased (read before you trust your own ECE)",
     arxiv("1909.10155"), "supplement"),
    (3, "nixon2019-measuring-calibration.pdf",
     "Measuring Calibration in Deep Learning", arxiv("1904.01685"), "supplement"),

    # ---- Week 4: delayed feedback ------------------------------------------------
    # The author's domain (olivier.chapelle.cc) has lapsed and now serves an unrelated
    # content farm, so every mirror here is a fallback. If all of them fail, the four
    # supplements below cover the same model -- see this week's papers/README.md.
    (4, "chapelle2014-delayed-feedback.pdf",
     "Modeling Delayed Feedback in Display Advertising (Chapelle, KDD 2014)",
     ("https://web.archive.org/web/2018id_/http://olivier.chapelle.cc/pub/delayedConv.pdf",
      "https://web.archive.org/web/2id_/http://olivier.chapelle.cc/pub/delayedConv.pdf",
      "https://web.archive.org/web/2016id_/http://olivier.chapelle.cc/pub/delayedConv.pdf"),
     "spine"),
    (4, "yasui2020-nonparametric-delayed-feedback.pdf",
     "A Nonparametric Delayed Feedback Model for Conversion Rate Prediction",
     arxiv("1802.00255"), "supplement"),
    (4, "yang2021-esdfm-elapsed-time-sampling.pdf",
     "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling",
     arxiv("2012.03245"), "supplement"),
    (4, "gu2021-real-negatives-matter.pdf",
     "Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling",
     arxiv("2104.14121"), "supplement"),
    (4, "chen2022-delayed-feedback-label-correction.pdf",
     "Asymptotically Unbiased Estimation for Delayed Feedback Modeling via Label Correction",
     arxiv("2202.06472"), "supplement"),

    # ---- Week 5: missing labels / conversion modeling ----------------------------
    (5, "bekker2018-pu-learning-survey.pdf",
     "Learning from Positive and Unlabeled Data: A Survey -- the formal shape of a missing-conversion label",
     arxiv("1811.04820"), "spine"),
    (5, "kiryo2017-nnpu.pdf",
     "Positive-Unlabeled Learning with Non-Negative Risk Estimator", arxiv("1703.00593"), "supplement"),
    (5, "blind-targeting-third-party-privacy-constraints.pdf",
     "Blind Targeting: Personalization under Third-Party Privacy Constraints",
     arxiv("2507.05175"), "supplement"),
    (5, "ghazi2021-label-differential-privacy.pdf",
     "Deep Learning with Label Differential Privacy -- when only the label is the sensitive part",
     arxiv("2102.06062"), "supplement"),

    # ---- Week 6: attribution -----------------------------------------------------
    (6, "diemert2017-criteo-attribution-modeling-bidding.pdf",
     "Attribution Modeling Increases Efficiency of Bidding in Display Advertising (the dataset paper)",
     arxiv("1707.06409"), "spine"),
    (6, "ren2018-dnamta.pdf",
     "Deep Neural Net with Attention for Multi-channel Multi-touch Attribution",
     arxiv("1809.02230"), "supplement"),
    (6, "yao2022-causalmta.pdf",
     "CausalMTA: Eliminating the User Confounding Bias for Causal Multi-touch Attribution",
     arxiv("2201.00689"), "supplement"),

    # ---- Week 7: uplift ----------------------------------------------------------
    (7, "gutierrez2017-causal-inference-and-uplift-modeling.pdf",
     "Causal Inference and Uplift Modelling: A Review of the Literature (PMLR v67)",
     "https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf", "spine"),
    (7, "diemert2021-criteo-uplift-benchmark.pdf",
     "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling (the dataset paper)",
     arxiv("2111.10106"), "spine"),
    (7, "kunzel2017-metalearners.pdf",
     "Metalearners for Estimating Heterogeneous Treatment Effects -- S/T/X-learner definitions",
     arxiv("1706.03461"), "supplement"),
    (7, "betlei2020-auuc-maximization.pdf",
     "Treatment Targeting by AUUC Maximization with Generalization Guarantees",
     arxiv("2012.09897"), "supplement"),

    # ---- Week 8: auctions & bid shading ------------------------------------------
    (8, "gligorijevic2020-bid-shading-brave-new-world.pdf",
     "Bid Shading in The Brave New World of First-Price Auctions", arxiv("2009.01360"), "spine"),
    (8, "pan2021-deep-distribution-network-bid-shading.pdf",
     "An Efficient Deep Distribution Network for Bid Shading in First-Price Auctions",
     arxiv("2107.06650"), "supplement"),
    (8, "zhang2014-ipinyou-rtb-benchmark.pdf",
     "Real-Time Bidding Benchmarking with iPinYou Dataset", arxiv("1407.7073"), "supplement"),

    # ---- Week 9: pacing & CPA control --------------------------------------------
    (9, "xu2015-smart-pacing.pdf",
     "Smart Pacing for Effective Online Ad Campaign Optimization (Xu et al., KDD 2015)",
     arxiv("1506.05851"), "spine"),
    (9, "wu2018-budget-constrained-bidding-rl.pdf",
     "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising",
     arxiv("1802.08365"), "supplement"),
    (9, "cai2017-rtb-reinforcement-learning.pdf",
     "Real-Time Bidding by Reinforcement Learning in Display Advertising",
     arxiv("1701.02490"), "supplement"),
    (9, "yang2019-bid-optimization-multivariable-control.pdf",
     "Bid Optimization by Multivariable Control in Display Advertising",
     arxiv("1905.10928"), "supplement"),

    # ---- Week 10: privacy --------------------------------------------------------
    (10, "criteo2025-criteoprivatead.pdf",
     "CriteoPrivateAds: A Real-World Bidding Dataset to Design Private Advertising Systems",
     arxiv("2502.12103"), "spine"),
    (10, "abadi2016-deep-learning-with-dp.pdf",
     "Deep Learning with Differential Privacy (DP-SGD)", arxiv("1607.00133"), "supplement"),
    (10, "bittau2017-prochlo-encode-shuffle-analyze.pdf",
     "Prochlo: Strong Privacy for Analytics in the Crowd -- the shuffle model behind aggregate reporting",
     arxiv("1710.00901"), "supplement"),

    # ---- Week 11: sequences ------------------------------------------------------
    (11, "zhou2018-deep-interest-network.pdf",
     "Deep Interest Network for Click-Through Rate Prediction", arxiv("1706.06978"), "spine"),
    (11, "zhou2019-deep-interest-evolution-network.pdf",
     "Deep Interest Evolution Network for Click-Through Rate Prediction", arxiv("1809.03672"), "supplement"),
    (11, "chen2019-behavior-sequence-transformer.pdf",
     "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba",
     arxiv("1905.06874"), "supplement"),
    (11, "vaswani2017-attention-is-all-you-need.pdf",
     "Attention Is All You Need", arxiv("1706.03762"), "supplement"),
]

# Free-text appended to a week's papers/README.md.
WEEK_NOTES: dict[int, str] = {
    4: """\
### On the Chapelle paper

Chapelle 2014 is the spine of this week and it has **no stable open copy**. The author's
site (`olivier.chapelle.cc`) lapsed; the domain now hosts an unrelated content farm, and
the Wayback snapshot was returning 503 when this repo was built. If the download slot
above is still empty, get it one of these ways:

- `https://dl.acm.org/doi/10.1145/2623330.2623634` (ACM DL; free via most institutions)
- Semantic Scholar corpus id `14993056`
- retry `python tools/fetch_papers.py --week 4` later -- the Wayback mirrors are already
  wired up and it will drop into place on its own

**You are not blocked without it.** The four supplements below all restate the model
formally, and Yasui 2020 in particular gives a clean derivation of the exponential-delay
likelihood you are asked to implement. Chapelle is worth reading for the framing and the
production context, not because the maths is unavailable elsewhere.
""",
}

# Living web docs that matter more than any paper for Week 10. Markdown, not PDF.
WEB_DOCS: list[tuple[int, str, str]] = [
    (10, "attribution-reporting-api-EVENT.md",
     "https://raw.githubusercontent.com/WICG/attribution-reporting-api/main/EVENT.md"),
    (10, "attribution-reporting-api-AGGREGATE.md",
     "https://raw.githubusercontent.com/WICG/attribution-reporting-api/main/AGGREGATE.md"),
    (10, "attribution-reporting-api-README.md",
     "https://raw.githubusercontent.com/WICG/attribution-reporting-api/main/README.md"),
]


def get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(url: str | tuple[str, ...], dest: Path, want_pdf: bool,
             attempts: int = 4) -> tuple[bool, str]:
    """Fetch ``url`` to ``dest``; keep it only if it validates. Returns (ok, message).

    ``url`` may be a tuple of mirrors, tried in order.
    """
    if isinstance(url, tuple):
        notes = []
        for u in url:
            ok, note = download(u, dest, want_pdf, attempts=2)
            if ok:
                return True, note
            notes.append(note)
        return False, "; ".join(notes)

    last = "?"
    for i in range(attempts):
        try:
            body = get(url)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            # 503 from web.archive.org means "busy", worth a real wait; 404 never is.
            if e.code == 404:
                break
            time.sleep(8 * (i + 1))
            continue
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(8 * (i + 1))
            continue

        if want_pdf and not body.startswith(b"%PDF"):
            last = f"not a PDF (got {body[:16]!r}, {len(body)} bytes)"
            time.sleep(5)
            continue
        if len(body) < 2048:
            last = f"suspiciously small ({len(body)} bytes)"
            time.sleep(5)
            continue

        dest.write_bytes(body)
        return True, f"{len(body) / 1e6:.1f} MB"
    return False, last


def valid(p: Path, want_pdf: bool) -> bool:
    if not p.exists() or p.stat().st_size < 2048:
        return False
    if want_pdf:
        with p.open("rb") as fh:
            return fh.read(4) == b"%PDF"
    return True


def write_index(week: int, rows: list[tuple[str, str, str, str]]) -> None:
    """rows: (filename, title, role, status)"""
    d = WEEK_DIRS[week] / "papers"
    lines = [f"# Week {week} -- reading", "",
             "Downloaded by `python tools/fetch_papers.py`. Re-run it to repair anything missing.",
             ""]
    for role in ("spine", "supplement"):
        sel = [r for r in rows if r[2] == role]
        if not sel:
            continue
        lines.append("## Spine" if role == "spine" else "## Supplementary")
        lines.append("")
        for fn, title, _, status in sel:
            mark = f"[`{fn}`]({fn})" if status == "ok" else f"`{fn}` *(missing: {status})*"
            lines.append(f"- **{title}**  \n  {mark}")
        lines.append("")
    if week in WEEK_NOTES:
        lines.append(WEEK_NOTES[week])
    (d / "README.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, action="append", help="restrict to week N (repeatable)")
    ap.add_argument("--check", action="store_true", help="report on-disk state, download nothing")
    args = ap.parse_args()

    weeks = set(args.week) if args.week else set(WEEK_DIRS)
    by_week: dict[int, list] = {}
    failures = []

    for wk, fn, title, url, role in PAPERS:
        if wk not in weeks:
            continue
        dest = WEEK_DIRS[wk] / "papers" / fn
        if valid(dest, True):
            status, note = "ok", "cached"
        elif args.check:
            status, note = "absent", "would download"
            failures.append((wk, fn, url, note))
        else:
            print(f"  w{wk:02d} {fn}", flush=True)
            ok, note = download(url, dest, want_pdf=True)
            status = "ok" if ok else note
            if not ok:
                failures.append((wk, fn, url, note))
        by_week.setdefault(wk, []).append((fn, title, role, status))
        if not args.check:
            print(f"       -> {status} ({note})", flush=True)

    for wk, fn, url in WEB_DOCS:
        if wk not in weeks:
            continue
        dest = WEEK_DIRS[wk] / "papers" / fn
        if valid(dest, False):
            status = "ok"
        elif args.check:
            status = "absent"
        elif args.check:
            status = "absent"
            failures.append((wk, fn, url, "would download"))
        else:
            ok, note = download(url, dest, want_pdf=False)
            status = "ok" if ok else note
            if not ok:
                failures.append((wk, fn, url, note))
            print(f"  w{wk:02d} {fn} -> {status}", flush=True)
        by_week.setdefault(wk, []).append(
            (fn, "Attribution Reporting API (Privacy Sandbox) -- living spec", "supplement", status))

    for wk, rows in by_week.items():
        write_index(wk, rows)

    total = sum(len(v) for v in by_week.values())
    print(f"\n{total - len(failures)}/{total} available")
    for wk, fn, url, note in failures:
        print(f"  MISSING w{wk:02d} {fn}\n      {url}\n      {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
