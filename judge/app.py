"""ads-ml-lab judge — a small Kaggle-shaped competition server for the twelve weeks.

    make judge            # http://localhost:8000

Deployment notes live in judge/README.md. The short version: set JUDGE_SECRET_KEY and
JUDGE_ENV=production, put it behind TLS, and mount a volume at judge/data.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlmodel import Session, select
from starlette.middleware.sessions import SessionMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge import security as sec                                    # noqa: E402
from judge.competitions import COMPETITIONS, UPCOMING, get           # noqa: E402
from judge.models import Submission, User, engine, init_db, utcnow   # noqa: E402
from judge.scoring import Rejected, score_submission                 # noqa: E402

JUDGE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(JUDGE / "templates"))


def _markdown(text: str) -> str:
    """Render markdown, then sanitise.

    The task text and week READMEs are repo-authored, not user input, so this is
    defence in depth rather than the primary control — but rendering anything to raw
    HTML without a sanitiser downstream is a habit worth not having.
    """
    import bleach
    import markdown as md
    from markupsafe import Markup

    html = md.markdown(text or "", extensions=["fenced_code", "tables", "sane_lists"])
    allowed = set(bleach.sanitizer.ALLOWED_TAGS) | {
        "p", "pre", "h1", "h2", "h3", "h4", "h5", "table", "thead", "tbody", "tr", "th",
        "td", "br", "hr", "span", "div", "img"}
    return Markup(bleach.clean(
        html, tags=allowed,
        attributes={"a": ["href", "title", "rel", "target"], "img": ["src", "alt"],
                    "code": ["class"], "span": ["class"], "div": ["class"]},
        protocols=["http", "https", "mailto"], strip=True))


def _fromjson(s: str) -> dict:
    try:
        return json.loads(s or "{}")
    except (TypeError, ValueError):
        return {}


templates.env.filters["markdown"] = _markdown
templates.env.filters["fromjson"] = _fromjson

if sec.is_production() and not os.environ.get("JUDGE_SECRET_KEY"):
    raise RuntimeError(
        "JUDGE_ENV=production requires JUDGE_SECRET_KEY. Without a stable key every "
        "restart invalidates all sessions and multiple workers cannot agree on one.")

app = FastAPI(title="ads-ml-lab judge", docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware,
    secret_key=sec.secret_key(),
    https_only=sec.is_production(),
    same_site="lax",
    max_age=int(timedelta(days=14).total_seconds()),
)
app.mount("/static", StaticFiles(directory=str(JUDGE / "static")), name="static")

MAX_UPLOAD_BYTES = 60 * 1024 * 1024


@app.on_event("startup")
def _startup() -> None:
    init_db()
    seed_baselines()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def current_user(request: Request, db: Session) -> User | None:
    uid = request.session.get("uid")
    return db.get(User, uid) if uid else None


def render(request: Request, name: str, **ctx) -> HTMLResponse:
    with Session(engine) as db:
        user = current_user(request, db)
    ctx.setdefault("user", user)
    ctx.setdefault("csrf", sec.csrf_token(request))
    ctx.setdefault("competitions", COMPETITIONS)
    ctx.setdefault("upcoming", UPCOMING)
    ctx.setdefault("flash", request.session.pop("flash", None))
    return templates.TemplateResponse(request, name, ctx)


def flash(request: Request, message: str, kind: str = "info") -> None:
    request.session["flash"] = {"message": message, "kind": kind}


def leaderboard_rows(db: Session, week: int, reveal_private: bool = False) -> list[dict]:
    """Best public submission per entrant, plus every seeded baseline, ranked together."""
    comp = COMPETITIONS[week]
    lower_is_better = not comp.primary_metric.higher_is_better

    scored = db.exec(
        select(Submission, User)
        .join(User, User.id == Submission.user_id)
        .where(Submission.week == week, Submission.status == "scored")
    ).all()

    best: dict[int, tuple[Submission, User]] = {}
    rows: list[tuple[Submission, User]] = []
    for s, u in scored:
        if s.is_baseline:
            rows.append((s, u))
            continue
        cur = best.get(u.id)
        if cur is None or (s.public_score < cur[0].public_score) == lower_is_better:
            best[u.id] = (s, u)
    rows.extend(best.values())

    out = []
    for s, u in rows:
        m = json.loads(s.metrics_json or "{}")
        out.append({
            "name": u.display_name,
            "label": s.label or "—",
            "is_baseline": s.is_baseline,
            "score": s.public_score,
            "private": s.private_score if reveal_private else None,
            "metrics": m,
            "when": s.created_at,
            "n_entries": sum(1 for x, _ in scored if x.user_id == u.id and not x.is_baseline),
        })
    out.sort(key=lambda r: r["score"], reverse=comp.primary_metric.higher_is_better)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


# --------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------
def seed_baselines() -> None:
    """Put a base-rate and a logistic-regression entry on every prepared leaderboard.

    Runs at startup and is idempotent. A leaderboard with nothing to beat teaches nothing;
    these two are the bar the repo's ground rules ask for.
    """
    from judge.baselines import build_baseline_predictions

    with Session(engine) as db:
        system = db.exec(select(User).where(User.is_system == True)).first()  # noqa: E712
        if system is None:
            system = User(email="baselines@ads-ml-lab.local", display_name="baselines",
                          password_hash=sec.hash_password(os.urandom(32).hex()),
                          is_system=True)
            db.add(system)
            db.commit()
            db.refresh(system)

        for week, comp in COMPETITIONS.items():
            if not comp.is_prepared:
                continue
            for label in comp.baselines:
                exists = db.exec(
                    select(Submission).where(Submission.week == week,
                                             Submission.is_baseline == True,  # noqa: E712
                                             Submission.label == label)).first()
                if exists:
                    continue
                try:
                    raw = build_baseline_predictions(comp, label)
                    score = score_submission(raw, comp)
                except Exception as e:                    # never block startup on this
                    print(f"[baseline] {label} w{week} failed: {type(e).__name__}: {e}")
                    continue
                pj, qj = score.as_json()
                db.add(Submission(
                    user_id=system.id, week=week, label=label, status="scored",
                    n_rows=score.n_rows, public_score=score.public,
                    private_score=score.private, metrics_json=pj,
                    private_metrics_json=qj, is_baseline=True))
                db.commit()
                print(f"[baseline] seeded {label} w{week}: "
                      f"{comp.primary_metric.key}={score.public:.5f}")


# --------------------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    with Session(engine) as db:
        counts = dict(db.exec(
            select(Submission.week, func.count(Submission.id))
            .where(Submission.status == "scored", Submission.is_baseline == False)  # noqa: E712
            .group_by(Submission.week)).all())
        entrants = dict(db.exec(
            select(Submission.week, func.count(func.distinct(Submission.user_id)))
            .where(Submission.status == "scored", Submission.is_baseline == False)  # noqa: E712
            .group_by(Submission.week)).all())
    return render(request, "index.html", counts=counts, entrants=entrants)


@app.get("/week/{week}", response_class=HTMLResponse)
def week_page(request: Request, week: int):
    comp = get(week)
    if comp is None:
        raise HTTPException(404, f"Week {week} has no competition yet.")
    with Session(engine) as db:
        user = current_user(request, db)
        mine = []
        if user:
            mine = db.exec(
                select(Submission)
                .where(Submission.user_id == user.id, Submission.week == week)
                .order_by(Submission.created_at.desc()).limit(25)).all()
        board = leaderboard_rows(db, week)
    return render(request, "week.html", comp=comp, board=board, mine=mine,
                  study=study_material(week))


@app.get("/week/{week}/leaderboard", response_class=HTMLResponse)
def leaderboard_page(request: Request, week: int):
    comp = get(week)
    if comp is None:
        raise HTTPException(404, "No such competition.")
    with Session(engine) as db:
        board = leaderboard_rows(db, week)
    return render(request, "leaderboard.html", comp=comp, board=board)


def study_material(week: int) -> dict:
    """Pull the week's README and paper index straight off disk.

    The study material is not duplicated into the judge — it *is* the repo's week folder,
    rendered. One source of truth, so editing the README updates the site.
    """
    import re

    repo = JUDGE.parent
    hits = sorted(repo.glob(f"week{week:02d}_*"))
    if not hits:
        return {}
    d = hits[0]
    papers = []
    idx = d / "papers" / "README.md"
    if idx.exists():
        for m in re.finditer(r"- \*\*(.+?)\*\*\s*\n\s*(\[`(.+?)`\]\(.+?\)|`(.+?)`.*)", idx.read_text()):
            papers.append({"title": m.group(1), "file": m.group(3) or m.group(4),
                           "available": bool(m.group(3))})
    readme = (d / "README.md")
    return {
        "folder": d.name,
        "readme": readme.read_text() if readme.exists() else "",
        "papers": papers,
        "notebook": f"{d.name}.ipynb",
        "github": f"https://github.com/o0aditya0o/ads-ml-lab/tree/main/{d.name}",
    }


# --------------------------------------------------------------------------------------
# downloads — explicit allow-list; the solution file is unreachable by construction
# --------------------------------------------------------------------------------------
DOWNLOADABLE = {
    "train": ("train_file", "train.csv.gz"),
    "test": ("test_file", "test.csv.gz"),
    "sample": ("sample_file", "sample_submission.csv.gz"),
}


@app.get("/week/{week}/download/{what}")
def download(request: Request, week: int, what: str):
    comp = get(week)
    if comp is None:
        raise HTTPException(404, "No such competition.")
    if what not in DOWNLOADABLE:
        # Not a 403: naming the solution file in an error message is itself a hint.
        raise HTTPException(404, "No such file.")
    attr, filename = DOWNLOADABLE[what]
    path: Path = getattr(comp, attr)
    if not path.exists():
        raise HTTPException(503, "This competition's data has not been prepared yet.")
    sec.rate_limit(f"dl:{sec.client_ip(request)}", limit=40, window_s=300)
    return FileResponse(path, filename=f"week{week:02d}_{filename}",
                        media_type="application/gzip")


# --------------------------------------------------------------------------------------
# submissions
# --------------------------------------------------------------------------------------
@app.post("/week/{week}/submit")
async def submit(request: Request, week: int,
                 file: UploadFile, label: str = Form(""), csrf: str = Form("")):
    sec.require_csrf(request, csrf)
    comp = get(week)
    if comp is None:
        raise HTTPException(404, "No such competition.")

    with Session(engine) as db:
        user = current_user(request, db)
        if user is None:
            flash(request, "Sign in to submit.", "error")
            return RedirectResponse(f"/login?next=/week/{week}", status.HTTP_303_SEE_OTHER)

        if not comp.open:
            flash(request, "This competition is closed.", "error")
            return RedirectResponse(f"/week/{week}", status.HTTP_303_SEE_OTHER)

        # The daily cap counts SCORED submissions only. Its purpose is to stop
        # leaderboard overfitting, and a file rejected for a bad header taught the
        # entrant nothing about the test set — charging for it just punishes people
        # still working out the format.
        since = utcnow() - timedelta(days=1)
        today = db.exec(select(func.count(Submission.id)).where(
            Submission.user_id == user.id, Submission.week == week,
            Submission.status == "scored", Submission.created_at >= since)).one()
        if today >= comp.max_daily_submissions:
            flash(request, f"Daily limit reached ({comp.max_daily_submissions} scored "
                           f"submissions per 24h). The limit is the point — tune on your "
                           f"own validation split, not on the leaderboard.", "error")
            return RedirectResponse(f"/week/{week}", status.HTTP_303_SEE_OTHER)

        # This one is abuse protection, so it counts every attempt including rejections
        # — each still costs a parse of a multi-MB upload.
        sec.rate_limit(f"submit:{user.id}", limit=12, window_s=60)

        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            flash(request, f"File is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB. "
                           f"Gzip it — a 400k-row submission compresses to a few MB.", "error")
            return RedirectResponse(f"/week/{week}", status.HTTP_303_SEE_OTHER)

        row = Submission(user_id=user.id, week=week, label=(label or "").strip()[:60])
        try:
            score = score_submission(raw, comp)
        except Rejected as e:
            row.status, row.error = "rejected", str(e)
            db.add(row)
            db.commit()
            flash(request, f"Rejected: {e}", "error")
            return RedirectResponse(f"/week/{week}", status.HTTP_303_SEE_OTHER)
        except Exception as e:
            row.status = "rejected"
            row.error = f"Scoring failed unexpectedly: {type(e).__name__}"
            db.add(row)
            db.commit()
            flash(request, row.error, "error")
            return RedirectResponse(f"/week/{week}", status.HTTP_303_SEE_OTHER)

        pj, qj = score.as_json()
        row.status, row.n_rows = "scored", score.n_rows
        row.public_score, row.private_score = score.public, score.private
        row.metrics_json, row.private_metrics_json = pj, qj
        db.add(row)
        db.commit()

        m = comp.primary_metric
        flash(request, f"Scored. Public {m.label} = {m.fmt.format(score.public)} "
                       f"(AUC {score.public_metrics.get('auc', float('nan')):.5f}).", "ok")
    return RedirectResponse(f"/week/{week}", status.HTTP_303_SEE_OTHER)


# --------------------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------------------
@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return render(request, "signup.html")


@app.post("/signup")
def signup(request: Request, email: str = Form(...), display_name: str = Form(...),
           password: str = Form(...), csrf: str = Form("")):
    sec.require_csrf(request, csrf)
    sec.rate_limit(f"signup:{sec.client_ip(request)}", limit=5, window_s=3600)

    email = email.strip().lower()
    display_name = display_name.strip()
    if err := sec.validate_signup(email, display_name, password):
        flash(request, err, "error")
        return RedirectResponse("/signup", status.HTTP_303_SEE_OTHER)

    with Session(engine) as db:
        if db.exec(select(User).where(User.email == email)).first():
            flash(request, "An account with that email already exists.", "error")
            return RedirectResponse("/signup", status.HTTP_303_SEE_OTHER)
        if db.exec(select(User).where(User.display_name == display_name)).first():
            flash(request, "That display name is taken.", "error")
            return RedirectResponse("/signup", status.HTTP_303_SEE_OTHER)
        u = User(email=email, display_name=display_name,
                 password_hash=sec.hash_password(password))
        db.add(u)
        db.commit()
        db.refresh(u)
        request.session["uid"] = u.id
    flash(request, f"Welcome, {display_name}.", "ok")
    return RedirectResponse("/", status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    return render(request, "login.html", next=next)


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...),
          next: str = Form("/"), csrf: str = Form("")):
    sec.require_csrf(request, csrf)
    ip = sec.client_ip(request)
    sec.rate_limit(f"login:{ip}", limit=10, window_s=600)

    with Session(engine) as db:
        u = db.exec(select(User).where(User.email == email.strip().lower())).first()
        # Same message either way: distinguishing them enumerates registered addresses.
        if u is None or u.is_system or not sec.verify_password(password, u.password_hash):
            sec.rate_limit(f"login-fail:{ip}", limit=5, window_s=600)
            flash(request, "Email or password is incorrect.", "error")
            return RedirectResponse("/login", status.HTTP_303_SEE_OTHER)
        if sec.needs_rehash(u.password_hash):
            u.password_hash = sec.hash_password(password)
            db.add(u)
            db.commit()
        request.session["uid"] = u.id
    # Only same-site relative paths, so ?next= cannot bounce anyone off the site.
    dest = next if next.startswith("/") and not next.startswith("//") else "/"
    return RedirectResponse(dest, status.HTTP_303_SEE_OTHER)


@app.post("/logout")
def logout(request: Request, csrf: str = Form("")):
    sec.require_csrf(request, csrf)
    request.session.clear()
    return RedirectResponse("/", status.HTTP_303_SEE_OTHER)


@app.get("/healthz")
def healthz():
    return {"ok": True, "weeks": sorted(COMPETITIONS),
            "prepared": [w for w, c in COMPETITIONS.items() if c.is_prepared]}
