#!/usr/bin/env python3
"""End-to-end smoke test against a running judge.

    make judge          # in one terminal
    python -m judge.smoke_test

Walks the whole flow a real entrant walks — signup, download, train, submit, leaderboard
— plus the rejection paths, which are the ones that actually get hit in practice and the
ones nobody tests.
"""
from __future__ import annotations

import gzip
import io
import re
import sys
import urllib.error
import urllib.request
from http.cookiejar import CookieJar

BASE = "http://127.0.0.1:8000"

# A stable account rather than a fresh one per run. Creating a new account each time
# burned the signup cap after a couple of runs, which made the suite unrunnable for an
# hour — the limiter was right and the test was wrong.
RUN = str(int(__import__("time").time()))[-6:]
# example.com is reserved and validates; .local / .test / .invalid do not —
# email_validator rejects special-use domains even with deliverability off.
EMAIL = "smoke@example.com"
NAME = "smoke tester"
PASSWORD = "smoke-test-account-pw"
PASSED, FAILED = [], []

_jar = CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Lets a test see the 307 itself instead of silently following it."""

    def redirect_request(self, *a, **k):
        return None


def req(path, data=None, headers=None, method=None):
    r = urllib.request.Request(BASE + path, data=data, headers=headers or {}, method=method)
    try:
        with _opener.open(r, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def csrf() -> str:
    _, body = req("/signup")
    m = re.search(rb'name="csrf" value="([^"]+)"', body)
    assert m, "no CSRF token in page"
    return m.group(1).decode()


def form(path, fields):
    import urllib.parse
    data = urllib.parse.urlencode(fields).encode()
    return req(path, data, {"Content-Type": "application/x-www-form-urlencoded"})


def multipart(path, fields, filename, content):
    b = "----smoke%s" % id(content)
    parts = []
    for k, v in fields.items():
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return req(path, b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={b}"})


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}{('  — ' + detail) if detail and not cond else ''}")


def main() -> int:
    import pandas as pd

    print("=" * 70, "\npublic pages\n" + "=" * 70)
    st, body = req("/")
    check("index renders", st == 200 and b"actually gets wrong" in body, f"status {st}")
    check("index lists locked weeks", body.count(b"locked") >= 11)
    st, body = req("/week/1")
    check("task page renders", st == 200 and b"CVR baseline" in body, f"status {st}")
    check("task text is rendered as HTML", b"<h3>" in body)
    check("stat strip is populated", b"train rows" in body and b"base rate" in body)
    check("theme toggle present", b"theme-toggle" in body)
    check("tab nav present on every page", body.count(b'href="/week/1') >= 4)
    check("active tab is marked", b'aria-current="page"' in body)

    print("\n" + "=" * 70, "\ntabs are their own pages\n" + "=" * 70)
    for path, needle in [("/week/1/data", b"sample_submission.csv.gz"),
                         ("/week/1/submit", b"Submit predictions"),
                         ("/week/1/leaderboard", b"Public leaderboard"),
                         ("/week/1/study", b"Reading")]:
        st, body = req(path)
        check(f"{path} renders", st == 200 and needle in body, f"status {st}")

    st, body = req("/week/1/leaderboard")
    check("calibration chart is rendered", b'class="chart"' in body)
    check("score bars are rendered", b"score-fill" in body)
    check("leaderboard shows seeded baselines", b"base_rate" in body and b"logreg" in body)

    print("\n" + "=" * 70, "\nstudy material is served inline\n" + "=" * 70)
    st, body = req("/week/1/study")
    check("papers listed with titles", b"View from the Trenches" in body)
    check("no github link in study material",
          b"github.com" not in body.split(b"<footer")[0].split(b"<main")[1])
    check("week notes are not on the study page", b"Week notes" not in body)
    check("notebook is not on the study page", b"nb-code" not in body)

    paper = b"mcmahan2013-ftrl-view-from-the-trenches.pdf".decode()
    st, body = req(f"/week/1/study/paper/{paper}")
    check("paper viewer page renders", st == 200 and b"<object" in body, f"status {st}")
    st, raw = req(f"/week/1/paper/{paper}")
    check("paper bytes served inline", st == 200 and raw[:4] == b"%PDF", f"status {st}")

    for probe in ["../../../etc/passwd", "..%2F..%2Fadslab%2Fdata.py", "README.md",
                  "week01_foundations.ipynb", "solution.parquet"]:
        st, _ = req(f"/week/1/paper/{probe}")
        check(f"paper route blocks: {probe}", st == 404, f"status {st}")

    st, _ = req("/week/99")
    check("unknown week 404s", st == 404, f"status {st}")

    print("\n" + "=" * 70, "\nsolution file is unreachable\n" + "=" * 70)
    for probe in ["solution", "solution.parquet", "../solution.parquet",
                  "..%2Fsolution.parquet", "train/../solution"]:
        st, _ = req(f"/week/1/download/{probe}")
        check(f"blocked: /download/{probe}", st == 404, f"status {st}")

    print("\n" + "=" * 70, "\ndata source: local file or Hub redirect\n" + "=" * 70)
    # Either mode is a correct deployment, so assert on the invariant that holds in both:
    # the bytes arrive and they are gzip. Then report which path served them.
    import urllib.request as _u
    _noredir = _u.build_opener(_NoRedirect, _u.HTTPCookieProcessor(_jar))
    try:
        r = _noredir.open(_u.Request(BASE + "/week/1/download/train"), timeout=60)
        code, loc = r.status, r.headers.get("location")
    except _u.HTTPError as e:
        code, loc = e.code, e.headers.get("location")
    if code in (301, 302, 307, 308):
        check("download redirects to the Hub", "huggingface.co" in (loc or ""),
              f"location={loc}")
        print(f"       serving from the Hub: {loc[:72]}...")
    else:
        check("download served locally (Hub copy not published yet)", code == 200,
              f"status {code}")
        print("       serving from the local file — publish with "
              "tools/publish_competition.py")

    print("\n" + "=" * 70, "\ndownloads\n" + "=" * 70)
    st, train_gz = req("/week/1/download/train")
    check("train downloads", st == 200 and train_gz[:2] == b"\x1f\x8b", f"status {st}")
    st, test_gz = req("/week/1/download/test")
    check("test downloads", st == 200 and test_gz[:2] == b"\x1f\x8b", f"status {st}")
    st, sample_gz = req("/week/1/download/sample")
    check("sample downloads", st == 200, f"status {st}")

    train = pd.read_csv(io.BytesIO(gzip.decompress(train_gz)))
    test = pd.read_csv(io.BytesIO(gzip.decompress(test_gz)))
    print(f"       train {train.shape}, test {test.shape}")
    check("train has the label", "conversion" in train.columns)
    leaky = {"conversion", "conversion_timestamp", "conversion_id", "attribution", "cpo"}
    found = leaky & set(test.columns)
    check("test has no leaky columns", not found, f"leaked: {found}")
    check("train precedes test in time", train.timestamp.max() < test.timestamp.min())

    print("\n" + "=" * 70, "\nauth\n" + "=" * 70)
    st, _ = multipart("/week/1/submit", {"csrf": csrf(), "label": "x"}, "s.csv", b"a,b\n1,2\n")
    check("anonymous submit is redirected to login", st in (200, 303), f"status {st}")

    tok = csrf()
    st, body = form("/signup", {"csrf": tok, "email": EMAIL,
                                "display_name": NAME, "password": "short"})
    check("short password rejected", b"at least" in body or st == 303)

    st, _ = form("/signup", {"csrf": "wrong-token", "email": f"x{RUN}@example.com",
                             "display_name": f"x{RUN}", "password": "correcthorse1"})
    check("bad CSRF token rejected", st == 403, f"status {st}")

    # Sign up the stable account, or sign in to it if a previous run already made it.
    form("/signup", {"csrf": csrf(), "email": EMAIL,
                     "display_name": NAME, "password": PASSWORD})
    st, body = req("/")
    if NAME.encode() not in body:
        form("/login", {"csrf": csrf(), "email": EMAIL, "password": PASSWORD, "next": "/"})
        st, body = req("/")
    check("signed in (signed up, or signed in to the existing account)",
          NAME.encode() in body)

    print("\n" + "=" * 70, "\nsubmission validation\n" + "=" * 70)
    tok = csrf()

    def submit(content, filename="sub.csv", label="smoke"):
        return multipart("/week/1/submit", {"csrf": tok, "label": label}, filename, content)

    st, body = submit(b"wrong,cols\n1,2\n")
    check("missing columns rejected", b"Missing the" in body, f"status {st}")

    st, body = submit(b"impression_id,prediction\n0,0.5\n")
    check("incomplete id set rejected", b"Missing" in body and b"required" in body)

    ids = test.impression_id.to_numpy()
    full = "impression_id,prediction\n" + "\n".join(f"{i},0.05" for i in ids)

    st, body = submit(full.replace("0.05", "5.0", 1).encode())
    check("out-of-range prediction rejected", b"must lie in [0, 1]" in body)

    st, body = submit((full + f"\n{ids[0]},0.1").encode())
    check("duplicate id rejected", b"duplicate" in body)

    st, body = submit((full + "\n999999999,0.1").encode())
    check("unknown id rejected", b"not in the test set" in body)

    st, body = submit(b"impression_id,prediction\n" + b"".join(
        f"{i},nan\n".encode() for i in ids))
    check("NaN predictions rejected", b"missing or" in body or b"non-numeric" in body)

    print("\n" + "=" * 70, "\nrate limiting\n" + "=" * 70)
    hit_429 = False
    for _ in range(14):
        st, _ = submit(b"nope,nope\n1,2\n")
        if st == 429:
            hit_429 = True
            break
    check("abuse limit fires on rapid submits", hit_429)
    if hit_429:
        print("       waiting out the 60s window ...", flush=True)
        import time
        time.sleep(62)

    print("\n" + "=" * 70, "\nscoring a real model\n" + "=" * 70)
    st, body = submit(gzip.compress(full.encode()), "constant.csv.gz", "constant 0.05")
    check("gzipped constant submission scores", b"Scored" in body, f"status {st}")

    from sklearn.linear_model import LogisticRegression
    from adslab import data as adsdata
    from adslab.encoders import HashingEncoder

    enc = HashingEncoder(adsdata.CAT_FEATURES, n_bits=16)
    n = min(200_000, len(train))
    sub = train.sample(n, random_state=0)
    m = LogisticRegression(solver="liblinear").fit(enc.transform(sub), sub.conversion)
    p = m.predict_proba(enc.transform(test))[:, 1]
    payload = "impression_id,prediction\n" + "\n".join(
        f"{i},{v:.6f}" for i, v in zip(ids, p))
    st, body = submit(gzip.compress(payload.encode()), "lr.csv.gz", "smoke LR 2^16")
    check("real model scores", b"Scored" in body, f"status {st}")
    m2 = re.search(rb"Public NE = ([\d.]+)", body)
    if m2:
        print(f"       scored NE = {m2.group(1).decode()}")

    st, body = req("/week/1/leaderboard")
    check("entrant appears on leaderboard", NAME.encode() in body)
    check("baselines still ranked", b"baseline" in body)

    st, body = req("/week/1/leaderboard")
    check("leaderboard page renders", st == 200 and NAME.encode() in body)

    print("\n" + "=" * 70, "\nanswer key resolution\n" + "=" * 70)
    from judge.competitions import WEEK1
    check("solution resolves", WEEK1.resolve_solution().exists())
    print(f"       from: {WEEK1.resolve_solution()}")
    check("private solution repo is configured (diskless hosts need it)",
          bool(WEEK1.hf_solution_repo))
    check("answer key is never in the public download allow-list",
          not any("solution" in v[1] for v in __import__(
              "judge.app", fromlist=["DOWNLOADABLE"]).DOWNLOADABLE.values()))

    print("\n" + "=" * 70)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
