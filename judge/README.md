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

Study material is read off disk from the repo's `weekNN_*/` folder at request time — the
week README and paper index are not duplicated into the judge, so editing the README
updates the site.

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

This is the one genuinely awkward part, and it is awkward for a good reason.

`solution.parquet` holds the hidden labels. **It must never be committed** — the repo is
public, and committing it would publish the answers. Regenerating it on the server is
also unattractive: `prepare_data` loads all 16.5M rows into memory and wants several GB,
which is more than a small instance has.

So prepare locally and copy the prepared directory up:

```bash
make judge-data                       # local, ~2 min
fly sftp shell                        # or scp / rsync for a plain VPS
  put judge/data/week01/train.csv.gz              /data/week01/train.csv.gz
  put judge/data/week01/test.csv.gz               /data/week01/test.csv.gz
  put judge/data/week01/sample_submission.csv.gz  /data/week01/sample_submission.csv.gz
  put judge/data/week01/solution.parquet          /data/week01/solution.parquet
  put judge/data/week01/baseline_base_rate.csv.gz /data/week01/baseline_base_rate.csv.gz
  put judge/data/week01/baseline_logreg_hashed_218.csv.gz /data/week01/baseline_logreg_hashed_218.csv.gz
```

Send the baseline caches too, or first boot will retrain the logistic regression on 1.5M
rows before it starts serving.

The whole of `judge/data/` — competition files and `judge.db` — must live on a
**persistent volume**. On an ephemeral filesystem every deploy wipes the accounts and the
leaderboard.

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
