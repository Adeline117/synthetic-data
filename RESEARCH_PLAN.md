# Membership Inference Attacks on Collaboratively Generated Synthetic Data

## Research Goal

Bridge CaPS (distributed SDG via DP-in-MPC) and MAMA-MIA (marginal-based MIA) to answer:
1. Can MAMA-MIA attack SDG algorithms not yet covered (MWEM+PGM, AIM, FLAIM)?
2. Does the distributed (CaPS) setting affect MIA success rates?
3. What new threat models emerge when data is distributed?

## Paper Structure

### Contribution 1: Extend MAMA-MIA to New Algorithms
- **MWEM+PGM**: focal points = selected marginals (worst-approximated cliques via exponential mechanism)
- **AIM**: focal points = adaptively selected cliques (exponential mechanism over L1 errors)
- **FLAIM**: focal points = AIM-style cliques in federated setting
- Compare attack accuracy (AUC) and efficiency vs MST/PrivBayes/Private-GSD

### Contribution 2: MIA in Distributed Setting
- **Threat Model A (External Attacker)**: sees only D_synth, knows algorithm used
- **Threat Model B (Malicious Data Holder)**: one of N data holders, has own D_i + D_synth
- **Threat Model C (Colluding MPC Server)**: semi-honest server, sees protocol transcripts
- Compare MIA success: centralized vs horizontal vs vertical partitioning

### Contribution 3: Analysis & Implications
- Does data distribution inherently reduce MIA risk?
- Privacy budget allocation: does splitting epsilon across parties help?
- Practical recommendations for CaPS deployments

## Experiment Plan

### Phase 1: MAMA-MIA on MWEM+PGM and AIM (Centralized Baseline)

| Experiment | SDG | Dataset | Epsilon | Config |
|-----------|-----|---------|---------|--------|
| E1.1 | MWEM+PGM | SNAKE | {0.1, 1, 10, 100, 1000} | n=1000, targets=32 |
| E1.2 | AIM | SNAKE | {0.1, 1, 10, 100, 1000} | n=1000, targets=32 |
| E1.3 | MWEM+PGM | CalHousing | {0.1, 1, 10, 100, 1000} | n=1000, targets=32 |
| E1.4 | AIM | CalHousing | {0.1, 1, 10, 100, 1000} | n=1000, targets=32 |
| E1.5 | FLAIM | SNAKE | {0.1, 1, 10, 100, 1000} | n=1000, targets=32 |

### Phase 2: MIA in Distributed Setting (CaPS)

| Experiment | SDG | Partition | N_holders | Config |
|-----------|-----|-----------|-----------|--------|
| E2.1 | AIM | Horizontal | 2 | n=1000, eps=10 |
| E2.2 | AIM | Vertical | 2 | n=1000, eps=10 |
| E2.3 | MWEM+PGM | Horizontal | 2 | n=1000, eps=10 |
| E2.4 | MWEM+PGM | Vertical | 2 | n=1000, eps=10 |
| E2.5 | AIM | Horizontal | {2,5,10} | n=1000, eps=10 |

### Phase 3: Threat Model Comparison

| Experiment | Attacker Type | Knowledge | Expected Outcome |
|-----------|---------------|-----------|------------------|
| E3.1 | External | D_synth + algorithm | Baseline attack |
| E3.2 | Data Holder | D_i + D_synth + algorithm | Stronger attack? |
| E3.3 | Colluding Server | protocol transcript + D_synth | Strongest attack? |

## Datasets

1. **SNAKE** (201,279 records, 15 features) - demographic, categorical-heavy
2. **California Housing** (20,640 records, 9 features) - continuous, numerical
3. **COMPAS** (4,120 records, 7 features) - criminal justice
4. **Diabetes** (614 records, 9 features) - healthcare

## Target Venues

- **Primary**: USENIX Security 2027 / IEEE S&P 2027 / PETS
- **Backup**: CCS 2027 / NDSS 2027

## Timeline

| Phase | Task | Duration |
|-------|------|----------|
| Phase 1 | MAMA-MIA extension (MWEM+PGM, AIM) | 4 weeks |
| Phase 2 | CaPS integration + distributed MIA | 4 weeks |
| Phase 3 | Experiments + analysis | 3 weeks |
| Phase 4 | Writing | 3 weeks |

## Code Architecture

```
src/
├── attacks/
│   ├── base.py              # Base attack class
│   ├── focal_points.py      # FP determination for all algorithms
│   ├── density.py           # Density estimation (zeta)
│   ├── mama_mia.py          # Core MAMA-MIA attack logic
│   └── scoring.py           # AUC, MA, activation functions
├── distributed/
│   ├── caps_simulator.py    # CaPS simulation wrapper
│   ├── threat_models.py     # External/insider/server threat models
│   └── partitioner.py       # Data partitioning (H/V/mixed)
├── generators/
│   ├── aim_wrapper.py       # AIM with FP tracking
│   ├── mwem_wrapper.py      # MWEM+PGM with FP tracking
│   └── flaim_wrapper.py     # FLAIM with FP tracking
└── utils/
    ├── data.py              # Data loading and encoding
    ├── metrics.py           # Evaluation metrics
    └── config.py            # Experiment configurations
```

## Key References

1. CaPS (Pentyala et al., ICML 2024) - github.com/sikhapentyala/MPC_SDG
2. MAMA-MIA (Golob et al., arXiv 2025) - github.com/steveng9/SyntheticData_MIA
3. TAPAS (Houssiau et al., NeurIPS 2022 Workshop)
4. FLAIM (Maddock et al., 2024) - openreview.net/forum?id=8hc2UvwTaL
5. AIM (McKenna et al., VLDB 2022)
6. MWEM+PGM (McKenna et al., ICML 2019)
