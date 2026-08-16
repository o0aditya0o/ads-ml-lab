# Glossary

Terms as this repo uses them. Where the industry is sloppy, the entry says so — several
of these distinctions are the entire content of a week.

**Attribution** — assigning credit for a conversion across the touchpoints that preceded
it. A *bookkeeping* rule, not a causal claim. Last-click, Markov removal effect and
Shapley are three different bookkeeping rules and none of them measures causation.
(Week 6.)

**Incrementality / uplift** — the conversions that happened *because* of the ad, measured
against a randomised control. A *causal* claim. Not the same thing as attribution and not
obtainable from attribution however clever the rule. (Week 7.)

**Calibration** — whether predicted probabilities match observed frequencies. Orthogonal
to ranking: a model can rank perfectly and be uniformly 3× too high. AUC cannot see it;
your spend can. (Week 3.)

**Calibration ratio / aggregate bias ratio** — `mean(prediction) / mean(label)`. 1.0 is
unbiased on aggregate. At 1.15 every bid is 15% too high. Necessary, not sufficient — it
can be exactly 1.0 while every individual segment is wrong in offsetting directions.

**ECE (Expected Calibration Error)** — count-weighted mean absolute gap between predicted
and observed rate, per bin. Biased *downward*, and the bias depends on the bin count, so
ECE at 10 bins and ECE at 100 bins are not the same quantity. Fix the bin count once and
report it alongside the number.

**Censoring** — a label you cannot observe *yet*. A click from two hours ago has not
converted, but that is not the same as never converting. Treating censored rows as
negatives is the delayed-feedback bias. (Week 4.)

**Delayed feedback** — conversions arriving hours to weeks after the impression. On the
Criteo attribution data the median delay is ~90 hours and 66% of conversions land more
than a day out. Any model trained on "converted so far" is systematically wrong about
recent traffic, in a direction that depends on how fresh the traffic is.

**Conversion modeling** — estimating conversions you cannot observe (consent loss, cookie
loss, cross-device, signed-out) and reporting the estimate alongside the observed ones.
What Google Ads and GA4 do, rebuilt in miniature in Week 5.

**MCAR / MAR / MNAR** — missing completely at random / at random given observed features /
not at random. Privacy-driven conversion loss is usually **MNAR**: whether a user consents
correlates with things that also predict conversion. That is why simply reweighting the
observed segment does not fix it, and why Week 5 has a hard version and an easy version.

**PU learning (positive–unlabelled)** — the formal shape of the missing-conversion
problem. Your positives are trustworthy; your "negatives" are a mixture of true negatives
and unobserved positives.

**Bid shading** — bidding below your true value in a first-price auction. Necessary
because first price offers no truthful-bidding guarantee: bid your value, win, and your
surplus is exactly zero. (Week 8.)

**Win-rate curve** — `P(win | bid)`. Estimated from censored data, since you learn the
clearing price only when you win. This censoring, not the optimisation, is the hard part.

**Pacing** — spending a budget smoothly across a day instead of exhausting it by 9am. Two
families: probabilistic throttling (skip auctions) and bid modulation (bid lower). The
second is generally better because it drops the impressions you value least rather than a
random subset. (Week 9.)

**Target CPA** — a controller adjusting bids to hit a cost-per-acquisition goal. Fragile
under delayed feedback, because early-day CPA looks catastrophic when conversions have not
arrived yet, and the controller reacts to an artifact.

**Privacy Sandbox / ARA** — Chrome's Attribution Reporting API. Replaces row-level
cross-site joins with event-level reports (noisy, low-entropy) and aggregate reports
(noised, under a contribution budget). (Week 10.)

**Contribution budget** — the cap on how much one conversion can add across aggregate
keys. The consequence people miss: utility depends less on ε than on **how many keys you
split the budget across**. Same ε, ten times the keys, every aggregate drowns.

**k-anonymity** — a report is released only if at least *k* users contributed. Interacts
badly with long-tail campaigns: precisely the segments you most need to measure are the
ones suppressed.

**Base rate** — the unconditional positive rate. Attribution set: 4.90% conversion,
36.12% click. FairJob: 0.70% click. Uplift: ~0.29% conversion. Always report it next to
AUC; AUC at a 4.9% base rate and at a 0.3% base rate are not comparable numbers.

**Normalised entropy** — log-loss divided by the log-loss of predicting the base rate.
Exactly 1.0 for the constant predictor; below 1.0 means the model earns its keep. From
the Facebook "practical lessons" paper.

**Qini curve** — the uplift analogue of a lift curve: incremental conversions as a
function of the fraction of the population targeted. The control arm must be rescaled by
`n_treated / n_control` — on the Criteo data that ratio is ~5.7, and omitting it makes a
null model look excellent.
