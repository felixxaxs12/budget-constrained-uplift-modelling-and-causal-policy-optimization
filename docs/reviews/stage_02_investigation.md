# Stage 2 investigation review

Date: 2026-08-03

Reviewer: independent reviewer subagent

Scope: official-data provenance, artifact integrity, dataset citations, targeted prior art, contribution boundary, analysis timing, and architectural redundancy

## Round 1: REVISE

The first review found four blocking issues:

1. The prior-art matrix omitted the closest response-versus-uplift comparison and a foundational constrained-uplift study.
2. The provenance materials did not distinguish Criteo's requested 2018 dataset citation from the 2021 paper that documents `CRITEO-UPLIFTv2`.
3. The downloader recorded a SHA-256 but did not reject an artifact with a different checksum.
4. The methodology did not resolve whether a full-sample average treatment effect would expose test outcomes before model decisions were frozen.

The evidence matrix, provenance files, downloader, and methodology were revised to address all four findings.

## Round 2: PASS

The reviewer confirmed that:

- the closest response-versus-uplift and constrained-uplift studies are included;
- the empirical contribution is limited to an open, pre-specified replication;
- the 2018 requested citation and 2021 v2 documentation are recorded separately;
- the downloader pins the reviewed byte size and SHA-256 and fails on mismatch;
- expected and observed integrity values are distinct in the manifest; and
- the full-sample average treatment effect is deferred until the single final-results opening after model and reporting decisions are frozen.

The reviewer also confirmed that Phase 2 contains no treatment-effect, outcome-rate, model-performance, or policy-value result. Source-reported rates remain distinct from locally observed integrity checks.

## Redundancy finding

The reviewer found the phase minimally sufficient and non-redundant: one downloader, one dependency, one machine-readable manifest, one human-readable provenance document, and one targeted evidence matrix. No duplicate schema layer, synthetic data, generated cost, speculative infrastructure, or model expansion was added.
