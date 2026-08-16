# ads-ml-lab

A 12-week hands-on curriculum: build every core model class an elite ads MLE is expected to
master — from scratch, on real public data — ending with a portfolio-grade capstone that mirrors
the privacy-era measurement problems from my Google Ads work.

**Audience for the output:** public portfolio + interviews + a possible budget-control startup.
Code is written to be read by other engineers, and each week's README is a publishable artifact.

## Repo conventions

- **One folder per week**: `week01/` … `week12/`. Commit ugly code; refactor later. Shipping beats polish.
- **Shared eval harness** lives outside the week folders (e.g. `adslab/` or `common/`) and is
  imported by every week — metrics must be comparable across all 12 weeks.
- **Splits are always by time**, never random. Random splits leak in ads data and silently inflate
  every number. If a new dataset shows up, find the timestamp column before anything else.
- **Every model gets a dumb baseline** (logistic regression, or a global average). A result only
  counts as a win if it beats the baseline *and* the reason is explainable.
- **Standard metrics everywhere**: AUC, log-loss, calibration error (ECE) + reliability diagram.
  Log them from Week 1 in a consistent format so the metrics tables stack.
- **Each week ends with a README** containing a metrics table and a short writeup of the finding.
  These compound into the portfolio; treat them as a deliverable, not a formality.

## Weekly rhythm

~8–10 hrs/week: 2 hrs deep reading (one paper), 5–6 hrs building, 1–2 hrs writing up.

---

## Phase 1 — Foundations & the CVR stack (Weeks 1–3)

### Week 1 — Data plumbing + CTR/CVR baseline
- Load **Criteo Attribution Modeling for Bidding** and **CriteoPrivateAd** (Hugging Face) into
  pandas/polars; profile feature cardinality, label sparsity, timestamp ranges.
- Build the reusable eval harness: time-based train/val/test split, AUC, log-loss, calibration
  curve plotting.
- Models: logistic regression with hashed categoricals (classic ads baseline), then LightGBM.
- **Deliverable:** repo scaffold + baseline CVR model + README metrics table.

### Week 2 — Deep CVR models: wide & deep, embeddings, FMs
- Read: DLRM (Meta) and Wide & Deep (Google) — the lineage of every production ads model.
- Build: a factorization machine, then a small wide-&-deep / DLRM-style net in PyTorch on the same
  Criteo data. Compare against Week 1 LightGBM.
- Question to answer with numbers: on tabular ads data, *when* do embeddings + MLP actually beat
  trees? (Often only at scale — measure it.)
- **Deliverable:** FM + DLRM-lite, comparison table, 1-page trees-vs-deep writeup.

### Week 3 — Calibration (the most underrated ads skill)
- Read: Guo et al., *On Calibration of Modern Neural Networks*; McMahan et al., FTRL paper
  (calibration sections).
- Build: Platt scaling and isotonic regression on top of the Week 2 model. Implement ECE and
  reliability diagrams by hand.
- Stress test: subsample positives by 50% (simulated conversion loss), watch calibration break,
  fix with a known-sampling-rate correction.
- **Deliverable:** reusable calibration module + before/after reliability diagrams.

## Phase 2 — The measurement problems (Weeks 4–7)

### Week 4 — Delayed feedback / conversion lag
- Read: Chapelle, *Modeling Delayed Feedback in Display Advertising* (canonical; from Criteo).
- Build on **CSSCL** (has real conversion timestamps): (a) naive model treating recent unconverted
  clicks as negatives, (b) Chapelle's exponential-lag delayed-feedback model, (c) optional Weibull
  survival variant.
- Show the naive model's bias on fresh traffic and how the lag model corrects it.
- **Deliverable:** delayed-feedback model + predicted-vs-actual lag distribution plot.

### Week 5 — Missing labels & conversion modeling (the Google wheelhouse, rebuilt openly)
- Simulate privacy loss on Criteo: drop conversions for a "consentless" segment (~30% of users),
  keeping ground truth for eval only.
- Build a two-model approach: observed-CVR model + a correction/imputation model that fills the
  gap — conversion modeling à la GA4/Google Ads, in miniature.
- Compare naive (biased) vs. corrected vs. oracle-on-true-labels.
- **Deliverable:** a "how conversion modeling works" writeup. Highly bloggable, interview-gold.

### Week 6 — Multi-touch attribution
- Read: Diemert et al., *Attribution Modeling Increases Efficiency of Bidding in Display Advertising*.
- Build on the **Attribution Modeling dataset**: last-click baseline → Markov-chain removal-effect
  → sampled Shapley value. Fork `criteo-research/robust-label-attribution` for scaffolding.
- Compare credit distributions across the three; connect attribution weights back to bidding value.
- **Deliverable:** attribution library with three methods + comparison analysis.

### Week 7 — Uplift / incrementality
- Read: Gutierrez & Gérardy uplift survey + the Criteo uplift dataset paper.
- Build on **Criteo Uplift Prediction** (subsample — 25M rows): T-learner, class transformation,
  X-learner. Implement Qini curves yourself.
- Internalize: "predicting conversion" ≠ "predicting incremental conversion". This distinction is
  the future of measurement.
- **Deliverable:** uplift models + Qini curve comparison.

## Phase 3 — Bidding, budgets & privacy (Weeks 8–10)

### Week 8 — Auction simulation + bid shading
- Build a first-price auction simulator with synthetic competitors (known value distributions).
- Plug in the CVR model: bid = predicted value × margin. Then bid shading — learn the win-rate
  curve, optimize surplus.
- Sanity check win-price distributions against price-paid fields in the Criteo attribution dataset.
- **Deliverable:** auction sim + bid shading model, surplus-vs-shading-aggressiveness plot.

### Week 9 — Budget pacing & CPA control (feeds the startup idea)
- Read: PID controllers for budget pacing (LinkedIn/Meta engineering blogs); Xu et al., *Smart Pacing*.
- Build: a pacing controller that spends a daily budget smoothly against simulated traffic, then a
  target-CPA controller (PID or a simple RL bandit) adjusting bid multipliers.
- Stress test: inject delayed conversion reporting (Week 4), watch the controller overspend, then
  fix it with lag-corrected CPA estimates. **This experiment is the technical heart of the
  budget-control startup idea.**
- **Deliverable:** pacing/CPA controller + overspend-under-delayed-data analysis.

### Week 10 — Privacy-constrained learning
- Read: the CriteoPrivateAd paper + Attribution Reporting API docs (Privacy Sandbox).
- Build: (a) Laplace-noised aggregate conversion counts, utility vs. epsilon; (b) train a bidding
  model on CriteoPrivateAd with cross-domain user features removed, quantify the drop; (c) an
  aggregate-only measurement model trained on noisy grouped labels.
- **Deliverable:** privacy/utility tradeoff curves — the evidence base for privacy-era pitches.

## Phase 4 — Sequence models & capstone (Weeks 11–12)

### Week 11 — Sequence models for user journeys
- Build an LSTM and a small transformer over per-user impression/click sequences (the Criteo
  attribution data has user timelines) predicting conversion. Compare to the best tabular model.
- Honest finding to chase: where do sequence models actually add lift over well-engineered tabular
  features? Quantify it.
- **Deliverable:** sequence model + tabular-vs-sequence analysis.

### Week 12 — Capstone: "privacy-era measurement stack" demo
- Wire it together: delayed-feedback-corrected CVR (W4) → calibration (W3) → attribution (W6) →
  CPA controller (W9), all running against simulated traffic with privacy-induced label loss (W5/W10).
- Ship one polished repo with a story-telling README, plus a blog post
  ("I rebuilt an ads measurement stack from scratch on open data").
- Triples as: startup prototype foundation, portfolio for early-stage ad-tech roles abroad, and
  proof-of-depth.

---

## Reading spine (one per week, in order)

1. McMahan et al., *Ad Click Prediction: a View from the Trenches* (FTRL)
2. Naumov et al., *DLRM* / Cheng et al., *Wide & Deep*
3. Guo et al., *On Calibration of Modern Neural Networks*
4. Chapelle, *Modeling Delayed Feedback in Display Advertising*
5. Diemert et al., *Attribution Modeling for Bidding* (Criteo)
6. Ji & Wang / DNAMTA / CausalMTA — skim
7. Gutierrez & Gérardy, *Causal Inference and Uplift Modeling*
8. Bid shading literature (e.g. *Bid Shading in First-Price Auctions*)
9. Xu et al., *Smart Pacing for Effective Online Ad Campaign Optimization*
10. CriteoPrivateAd paper + Attribution Reporting API docs
11. Zhou et al., *Deep Interest Network* (Alibaba)
12. Capstone week — no new reading; write instead

## Datasets & repos index

| Resource | Where | Used in |
| --- | --- | --- |
| Criteo Attribution Modeling for Bidding | ailab.criteo.com/criteo-attribution-modeling-bidding-dataset | W1, W6, W8, W11 |
| CriteoPrivateAd | huggingface.co/criteo | W1, W10 |
| Criteo Sponsored Search Conversion Log (CSSCL) | ailab.criteo.com (resources page) | W4 |
| Criteo Uplift Prediction | via `pyuplift` or HF mirrors | W7 |
| `criteo-research/robust-label-attribution` | GitHub | W6 |
| Avazu CTR (Kaggle) | kaggle.com | fallback CTR sandbox if Criteo downloads stall |
| Alibaba/Taobao Tianchi | tianchi.aliyun.com | richer sequences for W11 if desired |

## How to know it's working

- **By Week 4:** can explain calibration drift to anyone in one whiteboard sketch.
- **By Week 7:** can articulate why incrementality ≠ attribution, with own Qini curves as evidence.
- **By Week 9:** a working demo of the overspend problem observed inside Google — from the outside,
  on open data.
- **By Week 12:** a public repo + blog post that no ad-tech interviewer or investor can ignore.
