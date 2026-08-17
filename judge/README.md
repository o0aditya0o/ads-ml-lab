# ads-ml-lab judge

A small Kaggle-shaped competition server for the twelve weeks. Week 1 is live; the other
eleven are listed as locked so the shape of the course is visible from the front page.

```bash
make judge-data     # build week 1's competition files from data/raw  (one time)
make judge          # http://localhost:8000
make judge-test     # end-to-end smoke test against the running server
```

## How it works

```
judge/
  competitions.py   the registry — one Competition per week. Adding a week is an entry
                    here plus a prepare() function; nothing else knows about Week 1.
  prepare_data.py   builds train/test/sample/solution from data/raw
  scoring.py        validates and scores a submission (calls adslab.metrics)
  baselines.py      seeds every leaderboard with base-rate and logistic-regression rows
  security.py       argon2 passwords, CSRF, sliding-window rate limits
  models.py         SQLite via SQLModel: User, Submission
  app.py            FastAPI routes + Jinja rendering
  data/             generated artifacts and judge.db — gitignored, never committed
```

Scoring calls the **same** `adslab.metrics` the notebooks use. If your local number and
the leaderboard disagree, it is the split or the join, never the metric.

Study material is the week's reading, read off disk from the repo's `weekNN_*/papers/`
folder at request time — not duplicated into the judge, so there is no second copy to
drift.

**The papers are served locally.** They are already on disk, so sending a reader to
GitHub to read them was a pointless round trip. `judge/study.py` lists the directory,
titles each entry by parsing the generated index, and serves PDFs into an inline viewer;
the Privacy Sandbox markdown explainers render as prose instead.

The week README and the notebook are deliberately *not* shown here. They are working
documents for someone who has cloned the repo — a write-up template with empty sections
and a scaffold full of `TODO`s — and an entrant reading the competition page has no use
for either.

Paper serving is traversal-proof by construction: the request is reduced to a basename,
the extension must be in a two-entry allow-list, and the resolved path must still sit
inside that week's `papers/` directory. The generated `papers/README.md` index is
excluded so that listing and serving agree on what exists. The smoke test probes five
escape attempts including the notebook and the solution file.

## One page per tab

Task, Data, Submit, Leaderboard and Study material are five URLs, not five anchors on one
long scroll. `week_base.html` owns the shell — crumbs, header, stat strip, tab nav — and
each tab is a small template filling one block, so a page is only as long as the thing
you came for. Each ends with a Back/Next pager, because the tabs have a natural order:
read the task, get the data, submit, check the board, go read the papers.

## The interface

Dark-first, because the audience lives there; light is a designed mode with its own token
set, not an inverted one. The toggle in the header persists to `localStorage` and is
applied before first paint, so a light-mode user never sees a dark flash.

Numbers are the content of this product, so every figure is set in a tabular-numeral mono
face and columns of digits align by construction.

**One container level, never two.** Nested boxes were what made the first version read
like a stack of crates: a bordered panel holding bordered rows holding bordered tiles.
Now a section is either a panel or open prose, and everything inside it is separated by
hairlines and whitespace instead of its own border. The task section has no container at
all — a heading and air is enough for something you only read.

**Each week has a mark and an accent.** `judge/icons.py` holds twelve line-art SVGs, each
drawing the *idea* of its week rather than a generic glyph: week 3 is a reliability
diagram, week 7 is two diverging arms with the gap marked, week 9 is a damped oscillation
settling onto a target. All are 24x24, stroke-only, 1.5px, `currentColor`, so they inherit
whatever colour the context sets. Alongside each is an accent hue, exposed to templates as
`--wk` and used for the mark, the section rules, tints and hover states — so a week page
is quietly tinted its own colour throughout without ever depending on colour to carry
meaning.

That accent is chrome, not data. The chart keeps the validated dataviz palette regardless
of which week you are on.

**The leaderboard carries a chart**, and it is not decoration: every entry is plotted by
how well it *ranks* (AUC) against how well it is *priced* (calibration ratio), with a
dashed reference at 1.0. The two axes are independent, so an entry can sit far right and
far off the line — a model that orders impressions correctly and would still overbid every
auction it wins. The site argues its own case instead of asserting it in a paragraph.

Chart colours are the validated dataviz defaults — slot 1 blue for entrants, slot 2 orange
for you. Baselines are deliberately *not* a third hue: they are reference marks, so they
take neutral ink and a square marker, which keeps the categorical set at two and carries
identity by shape as well as colour. Both modes clear the all-pairs CVD and normal-vision
floors against their surfaces (worst pair ΔE 26.8 CVD / 31.8 normal in dark; 24.7 / 33.6
in light). Do not substitute by eye — re-run the validator.

## Design decisions worth knowing

**Ranked on normalised entropy, not AUC.** AUC is invariant to monotone rescaling: a
model calibrated 3× too high scores identically and would overbid every auction
threefold. NE is a proper scoring rule and sees both ranking and calibration. AUC and
calibration ratio are shown alongside, because their disagreement is the interesting
part. This is the whole argument of Week 3, enforced by the leaderboard.

**Public/private split, 30/70.** Deterministic on a hash of the impression id, not a
random draw, so re-preparing does not reshuffle which rows are public and invalidate
scores already on the board.

**Rate limits are split by what they protect.** Signup has a loose cap on attempts and a
strict cap on accounts actually created, applied only once the input is valid — charging
a rejected password against the account cap would mean five typos locks you out for an
hour, which punishes the wrong person. Same reasoning as the submission cap below.

**The daily cap counts scored submissions only.** Its purpose is to stop leaderboard
overfitting. A file rejected for a bad header taught the entrant nothing about the test
set, so charging for it just punishes people still working out the format. The
per-minute limit is separate, is abuse protection, and counts everything.

**The solution file is unreachable by construction.** Downloads resolve against a
three-entry allow-list of attribute names, not against a user-supplied path, so traversal
has nothing to traverse. `judge/smoke_test.py` probes five variants of the attack.

**Leaky columns are stripped from the test file** — not only `conversion` but
`conversion_timestamp`, `conversion_id` and `attribution`. That last one is the subtle
one: it is Criteo's own last-click flag and is nonzero only on converting journeys, so
shipping it would hand over the answer.

## Where the data lives

The judge no longer has to hold the competition files. `tools/publish_competition.py`
pushes them to two Hugging Face dataset repos:

| repo | visibility | contents | size |
|---|---|---|---|
| `aditagar/ads-ml-lab-week01` | **public** | `train.csv.gz`, `test.csv.gz`, `sample_submission.csv.gz` | 52 MB |
| `aditagar/ads-ml-lab-solutions` | **private** | `weekNN/solution.parquet` | 2 MB |

Downloads then 307-redirect to the Hub's CDN, and the answer key is fetched once per boot
with `HF_TOKEN`. That drops the server's storage requirement from 58 MB to **the 48 KB
database** — which is what makes hosts with no persistent disk viable, and moves all
download bandwidth off the server onto a CDN that is faster than a small VM anyway.

These files are *derived*, not copies — a time-ordered subsample with leaky columns
stripped and an `impression_id` assigned during preparation — so they have to be published
rather than linked to Criteo. The upstream dataset is **CC-BY-NC-SA 4.0**, which permits
this provided the derived data credits Criteo, stays non-commercial, and carries the same
licence; the generated dataset card does all three.

**Before publishing you need a write token.** The token used to download the source data is
read-only and cannot create repos:

```bash
# create a token with the "Write" role at https://huggingface.co/settings/tokens
hf auth login                 # `huggingface-cli` was removed in huggingface_hub 1.x
make judge-publish            # or: python tools/publish_competition.py --week 1
python tools/publish_competition.py --week 1 --verify
```

`--verify` asserts the answer key is absent from the public repo and that the solution repo
really is private. Run it after any publish.

Until this has run, nothing breaks: the download route probes the Hub once per boot and
falls back to serving local files, logging which path it took. Publishing later needs no
code change.

## Environment

| variable | required | purpose |
|---|---|---|
| `JUDGE_SECRET_KEY` | **in production** | signs session cookies; the app refuses to boot without it |
| `HF_TOKEN` | only if the answer key is not on local disk | fetches `solution.parquet` from the private repo |
| `JUDGE_DB` | no | database path; point it at a volume, or a Postgres URL later |
| `JUDGE_ENV` | no | `production` enables secure cookies and the secret-key check |

## Deploying

Built for a small deployed group. Before exposing it:

```bash
export JUDGE_ENV=production
export JUDGE_SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")
```

The app **refuses to start** in production without `JUDGE_SECRET_KEY` — without a stable
key every restart silently signs everyone out, and multiple workers would not agree on
one. Store it as a platform secret, not in a file.

### Getting the competition data onto the server

Publish it once (above) and the server fetches what it needs. No file copying, no volume
full of data.

What still needs to survive a restart is `judge.db` — the accounts and the leaderboard,
48 KB of it. Two ways:

- **A host with a disk** (Fly volume, Render paid, any VPS): point `JUDGE_DB` at it. Done.
- **A host with no disk** (Render free, Cloud Run): point SQLModel at a free Postgres.
  `judge/models.py` is plain SQLAlchemy `create_engine`, so this is a connection string
  and a driver, not a rewrite.

First boot will retrain the logistic-regression baseline on 1.5M rows before it serves,
which takes a minute or two. To avoid that, publish the baseline caches too or seed the
database once locally and copy it up.

### Docker

```bash
docker build -t ads-ml-lab-judge .
docker run -p 8000:8000 \
  -e JUDGE_ENV=production -e JUDGE_SECRET_KEY=... \
  -v $(pwd)/judge/data:/app/judge/data \
  ads-ml-lab-judge
```

`fly.toml` is included and mounts a volume at `/app/judge/data`. Create it once with
`fly volumes create judge_data --size 3`.

### Before you point a domain at it

- **TLS is not optional.** Session cookies are marked `secure` in production, so over
  plain HTTP nobody can stay signed in. Fly and Render terminate TLS for you.
- **Rate limits are in-process.** They reset on restart and do not coordinate across
  workers. Fine for one small instance; run a single worker, or move the limiter to the
  database before scaling out.
- **There is no password reset flow.** Deliberate at this size — an admin resets by hand.
  Add one before the group outgrows people you can message directly.
- **Backups.** `judge/data/judge.db` is the only irreplaceable file on the server;
  everything else can be rebuilt. `sqlite3 judge.db ".backup out.db"` on a schedule.

## Adding week 2

1. Define a `Competition` in `competitions.py` with its task text and metrics.
2. Write `prepare_weekN()` in `prepare_data.py` and register it in `PREPARERS`.
3. Optionally add baselines in `baselines.py`.
4. `python -m judge.prepare_data --week 2`, restart.

The routes, leaderboard, scoring, submission validation and templates need no changes —
that generality is the reason Week 1 was worth building carefully rather than quickly.
