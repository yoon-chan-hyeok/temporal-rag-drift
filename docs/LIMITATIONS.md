# Limitations

## Supported by completed experiments

- CLARK cumulative news snapshots can be compared with one fixed retriever,
  prompt and generator.
- Four answer-distribution change features transfer useful degradation-ranking
  signal from T0 to four later updates.
- Major Core4 classifiers achieve future AUROC around 0.85 and alarm risk lift
  around 3x in this setup.
- Controlled evidence interventions identify the first stage associated with
  answer recovery.

## Not supported

- A detector score proves that a particular answer is wrong.
- A previously unseen single query can be assessed without a replay, baseline
  distribution or additional perturbation/probe.
- Additive GAM is statistically superior to the other leading classifiers.
- A CLARK-selected detector transfers unchanged to another dataset, retriever,
  prompt or model.
- The first recovery stage is the unique causal root cause.
- Retrieval drift and generation drift are completely separated.
- The detector remains valid indefinitely after T4.

## Endpoint and selection constraints

- New degradation is an offline event defined from gold-validity annotations.
- Core4 uses changed questions only and treats every non-new-degradation
  operational state as negative.
- The earlier 186-case confirmatory baseline excludes persistent failures and
  therefore uses a different endpoint.
- Extra Trees won T0 F1 selection. GAM was retained post-hoc for interpretable
  diagnostic analysis and must not be presented as the prespecified winner.
- T1-T4 contain 60 positive events, with 11-22 per update; per-update estimates
  remain uncertain.

## Probe constraints

- The probe cohort is selected from detector outcomes and is not a random
  population prevalence sample.
- P5 explicitly contains the current gold fact and is an oracle upper bound.
- P2-P4 use benchmark-linked current support; automatic operational evidence
  selection is a separate unsolved problem.
- Five historical positives did not reproduce at P1, showing sampling
  stochasticity.
- False negatives are visible in retrospective analysis but would not be
  automatically probed by an alarm-only deployment.

## Reproducibility and redistribution

- Original CLARK questions and article text are not redistributed.
- Case-level response logs and predictions are withheld from the public copy.
- Hosted model aliases may change over time.
- Full scientific reproduction requires licensed CLARK inputs, local article
  materialization, cached model weights and paid generation.
- The repository's synthetic smoke run verifies wiring only.
