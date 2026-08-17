# Deployment design — where everything runs, and what it costs

The judge works locally today. This document is the design for putting it on the internet
at `judge.adityanotes.com`, with the alternatives priced so the choice can be made by
reading rather than by guessing.

For what the individual pieces *are*, read [`how-it-works.md`](how-it-works.md) first.

**Status: nothing is deployed.** Everything below is a design.

---

## 1. What exists today

Three places. None of them is a server the public can reach.

```
┌──────────────────────────────────┐   ┌─────────────────────────┐
│  Your MacBook                    │   │  GitHub (public)        │
│                                  │   │  o0aditya0o/ads-ml-lab  │
│  data/raw/          4.2 GB       │   │                         │
│    Criteo source. Only used to   │   │  code, notebooks        │
│    BUILD the competition. Never  │   │  56 MB of papers        │
│    deployed anywhere.            │   │  Dockerfile             │
│                                  │   │                         │
│  judge/data/         58 MB       │   │  NOT: judge.db          │
│    competition files + judge.db  │   │  NOT: solution.parquet  │
│    (gitignored)                  │   │  NOT: raw data          │
│                                  │   └─────────────────────────┘
│  localhost:8000                  │
│    the judge — only you can      │   ┌─────────────────────────┐
│    reach it                      │   │  Hugging Face           │
└──────────────────────────────────┘   │                         │
                                       │  aditagar/…-week01      │
                                       │    PUBLIC · 52 MB       │
                                       │    train, test, sample  │
                                       │                         │
                                       │  aditagar/…-solutions   │
                                       │    PRIVATE · 2 MB       │
                                       │    the answer key       │
                                       └─────────────────────────┘
```

## 2. Target architecture

One new box. Everything else stays exactly where it is.

```
                          your visitor's browser
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  GoDaddy DNS                  │   the adityanotes.com zone
                    │  (nameservers: domaincontrol) │   — you add ONE record here
                    └───────┬───────────────┬───────┘
                            │               │
            adityanotes.com │               │ judge.adityanotes.com
             (existing A →  │               │ (NEW record → the host)
              216.198.79.1) │               │
                            ▼               ▼
            ┌───────────────────────┐   ┌──────────────────────────────────┐
            │  Vercel               │   │  Container host   ◀── TO DECIDE  │
            │                       │   │                                  │
            │  your notes site      │   │  Docker: uvicorn + FastAPI       │
            │  output:'export'      │   │  245 MB idle / 357 MB peak       │
            │  = pre-built HTML     │   │                                  │
            │  NO server, NO code   │   │  ┌────────────────────────────┐  │
            │  runs                 │   │  │ judge.db      48 KB        │  │
            │                       │   │  │ accounts + submissions     │  │
            │  free, untouched      │   │  │ THE ONLY MUTABLE STATE     │  │
            └───────────────────────┘   │  │ must survive restarts      │  │
                                        │  └────────────────────────────┘  │
                                        └──────┬─────────────────┬─────────┘
                                               │                 │
                    fetches once at boot ──────┘                 └───── 307 redirect
                                    │                                        │
                                    ▼                                        ▼
                    ┌───────────────────────────┐          ┌──────────────────────────┐
                    │  HF private repo          │          │  HF public repo + CDN    │
                    │  solution.parquet  1.9 MB │          │  train/test  52 MB       │
                    │  read-only, never served  │          │  served straight to the  │
                    │  needs HF_TOKEN           │          │  browser — never touches │
                    └───────────────────────────┘          │  your server             │
                                                           └──────────────────────────┘
```

**The most important line on that diagram:** the 52 MB of downloads goes browser →
Hugging Face directly. The server only emits a redirect. That is why the host can be tiny
and why bandwidth is not a cost.

### Why the notes site and the judge cannot share a home

`adityanotes.io` is built with `output: 'export'` — Next.js pre-renders everything to
static HTML and Vercel hands out finished files. **No server runs.** That is why it is free
and instant.

The judge is a Python process that must stay alive, hold a database, and run scikit-learn
over a 400,000-row upload. Vercel cannot host it, for three separate reasons:

| | Vercel's limit | The judge | Fixable? |
|---|---|---|---|
| Function bundle | 250 MB unzipped | **403 MB** — pyarrow 120, scipy 92, pandas 60, numpy 55, sklearn 43 | ❌ the build fails |
| Request body | 4.5 MB, infrastructure-level | submissions are **4.4 MB gzipped**, 10.4 MB raw | ❌ scraping under by 100 KB |
| Filesystem | none | SQLite file | ✅ easy — free Postgres |

Only the database has a clean solution. The dependency ceiling is the real blocker: Vercel
would refuse to build before running a line of code.

## 3. Component inventory

| What | Lives where | Size | Changes? | If it is lost |
|---|---|---|---|---|
| **judge.db** — accounts, submissions | the container host | 48 KB | **constantly** | **Unrecoverable.** Everyone re-registers, leaderboard gone. The only thing needing backup. |
| solution.parquet — answer key | HF, private | 1.9 MB | never | re-upload from your Mac |
| train / test / sample | HF, public | 52 MB | never | re-upload from your Mac |
| facts.json — header numbers | HF, public | <1 KB | never | regenerate |
| Papers (PDFs) | GitHub repo → Docker image | 56 MB | rarely | `make papers` |
| Code, templates, notebooks | GitHub → Docker image | small | on push | it is on GitHub |
| Raw Criteo data | **your Mac only** | 4.2 GB | never | `make data` |
| Session secret, HF token | host's secret store | — | — | regenerate; signs everyone out |

## 4. What actually happens on a request

**Downloading the training data**

```
browser → judge.adityanotes.com/week/1/download/train
        ← 307 Temporary Redirect
browser → huggingface.co/datasets/aditagar/ads-ml-lab-week01/resolve/main/train.csv.gz
        ← 302 to CDN → 41 MB streams from Hugging Face
```
The server transfers about 200 bytes. Verified working.

**Submitting predictions**

```
browser → POST 4.4 MB gzip → judge.adityanotes.com/week/1/submit
              ↓ validate ids, range, duplicates
              ↓ read solution.parquet   (fetched once at boot, then cached)
              ↓ score with adslab.metrics       ← peak 357 MB RAM
              ↓ INSERT row into judge.db        ← the only write
        ← 303 back to the submit page with the score
```
The only path needing real CPU, memory and durable storage.

## 5. The constraint that eliminates options: memory

Measured on a real scoring pass of the 400k-row baseline:

```
245 MB   idle (Python + pandas + numpy + scipy + sklearn + pyarrow loaded)
357 MB   peak while scoring one submission
```

Baseline seeding on a cold database is heavier still — it fits a logistic regression on
400k rows × 262k hashed features. Treat **512 MB as the floor and 1 GB as comfortable.**
This single number rules out the cheapest tier of several hosts, so check it before price.

## 6. Alternatives, priced

Three genuinely different architectures, not just three hosts.

### Design A — Server + Hugging Face data (what is built today)

The diagram in section 2. A live Python server, real accounts, instant scoring. Everything
already works; only the host is unchosen.

| Host | RAM | $/month | $/year | judge.db lives on | Ops burden | Verdict |
|---|---|---|---|---|---|---|
| **GCP e2-micro** + external IPv4 | **1 GB** | **$3.65** | **$44** | VM disk (30 GB free) | you: Docker, Caddy/TLS, patches, budget alert | cheapest always-on with enough RAM |
| **Oracle Always Free** ARM | **12 GB** | **$0** | **$0** | VM disk | same as GCP | free and roomy, *if* you can get capacity |
| **Fly.io** shared-1x/1GB | 1 GB | ~$2–6 | ~$25–70 | volume, $0.15/GB | low; CLI deploys | `fly.toml` already written |
| **Render Starter** | **512 MB** | $7 + $0.25/GB disk | ~$87 | attached disk | none | **RAM is tight — 357 MB peak in 512 MB** |
| **Render Standard** | 2 GB | $25 | $300 | attached disk | none | comfortable but overpriced here |
| **Render Free** + Neon Postgres | 512 MB | $0 | $0 | **Neon** (no disk on free) | none | ~1 min cold start; needs the Postgres migration first |

*GCP note:* the VM is free forever, the **public IPv4 is not** — $0.005/hr, with no
free-tier exemption. That $3.65 is the entire cost of making it reachable.

### Design B — No server at all: static leaderboard, batch scoring

The only design that satisfies the original wish of `adityanotes.com/ads-ml-judge`, because
with no server there is nothing Vercel cannot host.

```
entrant forks the repo, opens a Pull Request with predictions.csv.gz
        │
        ▼
GitHub Actions  ── scores it (the same adslab.metrics) ── commits results.json
        │
        ▼
static leaderboard page rebuilt ──▶ Vercel, under adityanotes.com/ads-ml-judge
```

| | |
|---|---|
| **Cost** | **$0/month, forever.** Actions is free for public repos; Vercel already hosts the site. |
| **Accounts** | None to build — identity is the GitHub account. **No passwords stored anywhere.** |
| **State** | `results.json` in git. No database, no backups, full history for free. |
| **Feedback** | Minutes, not seconds. No live leaderboard. |
| **Cost to build** | High — retires the FastAPI app, auth and upload UI. Keeps the scoring, data and design. |

Worth taking seriously if the goal is a public artifact rather than a live service. It is
strictly more robust: nothing to keep running, nothing to patch, nothing to leak.

### Design C — Cloudflare Tunnel from your Mac

```
visitor ──▶ Cloudflare ──▶ encrypted tunnel ──▶ your MacBook ──▶ the judge
```

| | |
|---|---|
| **Cost** | **$0** |
| **Code changes** | **none** — the app is untouched, SQLite keeps working |
| **Setup** | minutes |
| **Catch** | online only while the Mac is awake and the tunnel running. A named tunnel on `judge.adityanotes.com` needs the domain moved to Cloudflare; a throwaway `trycloudflare.com` URL needs nothing. |

The cheapest way to find out whether anyone actually uses this before paying for anything.

### Design D — Do not deploy yet

$0. Week 1's notebook is still scaffolded and no model has been built; the leaderboard has
two seeded baselines and nothing else. Deploying can wait until there is something to show.

## 7. Recommendation

**Design C now, Design A on GCP later.** Put it behind a Cloudflare Tunnel for nothing and
no changes; if people actually use it, move to the GCP e2-micro at $44/year — the cheapest
always-on option with enough memory.

If you would rather never run a server, **Design B** is the durable answer: nothing to keep
alive, no passwords stored, and it lands at the `adityanotes.com/ads-ml-judge` URL
originally wanted — at the price of rebuilding the submission flow.

Avoid Render Starter specifically: $87/year for 512 MB against a 357 MB peak is paying more
than GCP for less headroom.

Per year, at a glance:

```
Design C  Cloudflare Tunnel from your Mac      $0     no changes, laptop-dependent
Design D  don't deploy yet                     $0     nothing to show yet anyway
Design B  static + GitHub Actions              $0     big rework, no passwords ever
Design A  Oracle Always Free                   $0     if you can get capacity
Design A  GCP e2-micro                        $44     cheapest always-on with 1 GB
Design A  Fly.io                          $25–70     fly.toml already written
Design A  Render Starter                      $87     512 MB — too tight, avoid
Design A  Render Standard                    $300     comfortable, overpriced here
```

## 8. Gaps to fix before any deploy

Found while writing this. All small, all real, none fixed yet:

1. **`.dockerignore` excludes `**/papers/*.pdf`.** The deployed image would ship zero PDFs,
   so the study page would list every paper as missing and the viewer would 404. That rule
   predates the study page serving papers inline. Remove it and accept a 56 MB-larger image,
   or fetch papers at boot like the answer key.
2. **Rate limiting breaks behind a proxy.** `judge/security.py:client_ip` reads
   `X-Forwarded-For`, but the Dockerfile runs uvicorn without `--forwarded-allow-ips`, so
   every visitor would share one bucket — the proxy's IP. Add `--forwarded-allow-ips="*"`,
   safe when only the proxy can reach the port.
3. **Secrets.** `JUDGE_SECRET_KEY` (the app refuses to boot in production without it) and
   `HF_TOKEN` (read scope, to fetch the answer key) must be set in the host's secret store.
4. **No backup for judge.db.** The only irreplaceable file. A daily
   `sqlite3 judge.db ".backup"` copied off the host is enough.
5. **`fly.toml` is misleading** — it targets a host that may not be chosen. Delete it or
   mark it unused.

## 9. Implementation

### Design C (tunnel) — minutes, nothing to change

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000    # prints a public HTTPS URL
```

For `judge.adityanotes.com` rather than a throwaway URL, move the domain's nameservers from
GoDaddy to Cloudflare first, then create a named tunnel. That changes the live site's DNS,
so it is only worth doing once you know you want this permanently.

### Design A (a real host)

1. Fix the five gaps in section 8.
2. `docker build` and run locally — confirm papers, scoring and login work **in the
   container**, not just under `make judge`. The `.dockerignore` bug is exactly what this
   step catches.
3. Deploy; set `JUDGE_ENV=production`, `JUDGE_SECRET_KEY`, `HF_TOKEN`, and `JUDGE_DB`
   pointing at durable storage.
4. Add **one** DNS record at GoDaddy — `judge` → the host. The apex and `www` records
   pointing at Vercel are untouched, so adityanotes.com cannot break.
5. Wait for TLS, then run the smoke test against the public URL.

**GCP specifics:** VM in `us-central1` (other regions bill), install Docker, Caddy in front
for automatic TLS, a systemd unit so it survives reboot, and **a budget alert on day one** —
a wrong region, disk type or egress spike bills silently.

**Render Free specifics:** migrate SQLite to Neon Postgres *first*; free instances have no
disk, so `judge.db` vanishes on every restart.

### Design B (static, no server)

A separate piece of work, not a deployment: move scoring into a GitHub Actions workflow,
define the PR-based submission format, render the leaderboard to static HTML, point a Vercel
rewrite at it. The FastAPI app, auth and upload UI are retired; `adslab.metrics`, the data
and the visual design carry over.

## 10. Verification

Applies to Designs A and C; B needs its own plan.

- `docker run` the image locally, then sign up, download (expect a 307 to huggingface.co),
  submit the baseline file, and confirm **NE = 0.80457** — the number this has produced at
  every stage. A different number means the answer key resolved wrong.
- Watch memory during that submission (`docker stats`). If peak approaches the host's limit,
  the host is too small.
- `python -m judge.smoke_test` against the public URL — all 58 checks, including the five
  traversal probes and the answer-key-not-public assertion.
- Confirm `https://adityanotes.com` still loads and `dig judge.adityanotes.com` resolves to
  the host, with nothing else changed in the zone.
- Restart the host and confirm accounts and the leaderboard survive. This is the real test
  of whether durable storage is wired correctly.
