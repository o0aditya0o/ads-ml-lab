# Week 4 -- reading

Downloaded by `python tools/fetch_papers.py`. Re-run it to repair anything missing.

## Spine

- **Modeling Delayed Feedback in Display Advertising (Chapelle, KDD 2014)**  
  `chapelle2014-delayed-feedback.pdf` *(missing: absent)*

## Supplementary

- **A Nonparametric Delayed Feedback Model for Conversion Rate Prediction**  
  [`yasui2020-nonparametric-delayed-feedback.pdf`](yasui2020-nonparametric-delayed-feedback.pdf)
- **Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling**  
  [`yang2021-esdfm-elapsed-time-sampling.pdf`](yang2021-esdfm-elapsed-time-sampling.pdf)
- **Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling**  
  [`gu2021-real-negatives-matter.pdf`](gu2021-real-negatives-matter.pdf)
- **Asymptotically Unbiased Estimation for Delayed Feedback Modeling via Label Correction**  
  [`chen2022-delayed-feedback-label-correction.pdf`](chen2022-delayed-feedback-label-correction.pdf)

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
