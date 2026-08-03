# Prior art and contribution boundary

This project does not claim to invent uplift modelling, Qini/AUUC evaluation, doubly robust policy value, or budget-constrained targeting.

## Closest verified work

| Work | Overlap | Difference from this project |
| --- | --- | --- |
| Diemert et al. (2018), *A Large Scale Benchmark for Uplift Modeling* | Introduced the Criteo randomized benchmark, its sanity checks, and uplift ranking metrics. | The present study uses the later unbiased v2.1 release and makes capacity-specific held-out policy value and uncertainty the primary decision quantities. |
| Rößler and Schoder (2022), *Bridging the Gap: A Systematic Benchmarking of Uplift Modeling and Heterogeneous Treatment Effects Methods* | Benchmarked many uplift methods, including Criteo, and discussed recommended targeting fractions. | The present study uses fewer models, a locked public reproduction, and paired DR policy-value inference across capacities. It is not a broader algorithm survey. |
| Dudík, Langford, and Li (2011), *Doubly Robust Policy Evaluation and Learning* | Provides the doubly robust off-policy evaluation basis. | The estimator is adopted, not claimed as new. The application is a binary randomized incrementality benchmark. |
| *A decision-oriented empirical comparison of predictive and uplift-based scoring under budget constraints* (2026) | Compares predictive and uplift scores with DR/AIPW value under top-k budgets. | That study uses confidential observational internet-lending data. The present study is an open replication-style analysis on public randomized advertising data. |
| *Budget-Constrained Causal Bandits* (2026 preprint) | Studies budgeted allocation on Criteo. | It is an online/cold-start bandit study. Online learning and pacing are outside this project's scope. |

## Defensible contribution

The contribution is empirical and reproducibility-oriented:

1. a public end-to-end analysis of the unbiased Criteo v2.1 randomized release;
2. a pre-specified comparison of response ranking, T-learning, and DR-learning under equal capacity;
3. paired uncertainty for held-out incremental policy value across the capacity frontier;
4. a direct comparison between conventional uplift ranking metrics and the decision value of the policies they induce;
5. an explicit separation of treatment assignment, post-assignment exposure, projected policy value, and realized business impact.

The paper must call this a benchmark, replication, or empirical evaluation unless completed evidence supports a narrower claim. A new algorithm is not part of the contribution.

## Primary links

- Criteo dataset and terms: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Diemert et al. paper page: https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018
- Rößler and Schoder DOI: https://doi.org/10.1177/10949968221111083
- Dudík et al.: https://arxiv.org/abs/1103.4601
- 2026 decision-oriented comparison: https://doi.org/10.1007/s10791-026-10251-5
- 2026 causal-bandit preprint: https://arxiv.org/abs/2604.26169
