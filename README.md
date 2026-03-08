# Whitening Reveals Cluster Commitment as the Geometric Separator of Hallucination Types

Code and data for the paper *"Whitening Reveals Cluster Commitment as the Geometric Separator of Hallucination Types"* — Paper 3 of the CPU Trilogy.

We apply PCA-whitening and eigenspectrum decomposition to GPT-2-small's contextual hidden states to resolve the Type 1/2 collapse from [Paper 2](https://github.com/x3mm3x/llm-hallucination-induction-geometry). Whitening transforms the micro-signal regime (H(v) ≈ 0.985, max_sim ≈ 0.993) into a calibrated space where peak cluster alignment (max_sim) emerges as the theoretically correct separating metric.

- **Type 1 (Center-drift)** — minimal prompts ("The", "It is") produce generic high-frequency continuations
- **Type 2 (Wrong-well)** — ambiguous prompts ("The bank announced record levels of") force commitment to one domain
- **Type 3 (Coverage gap)** — compositionally novel prompts ("The xenoplasmic refractometry of late-Holocene") encounter no training support

Key findings across 20 independent generation runs (K=20 stability protocol):

| Finding | Whitened (N=30/group) | Spectral (N=15/group) |
|---------|----------------------|-----------------------|
| max_sim T2–T3 separation | 60% sig, 40% Holm, r=−0.31, dir 20/20 | 90% sig (tail band), 16/20 Holm |
| max_sim T1–T2 hint | 35% sig, 15% Holm, r=+0.21, dir 17/20 | 15% sig (tail), not significant |
| Predicted ordering T2>T1>T3 | Confirmed (max_sim means) | — |
| H(v) false positive (N=15→30) | Collapses: 90%→5% KW sig | Localized to PCs 1–16 |
| Spectral mixing hypothesis | — | Rejected: T1–T2 absent in all 6 bands |
| Pseudoreplication (T2–T3 max_sim) | Inverted: 0.3× (prompt > token) | ~1.0× (tail band) |

## Repository Structure

```
├── hallucination_induction_whitened.py    # Experiment 1: PCA-whitened contextual analysis (N=30)
├── hallucination_induction_spectral.py    # Experiment 2: eigenspectrum band decomposition (N=15)
├── hallucination_stats.py                 # Shared two-level statistical analysis module
├── run_multirun_whitened.py               # K-run stability wrapper — whitened (main entry point)
├── run_multirun_spectral.py               # K-run stability wrapper — spectral
├── generate_figures.py                    # Publication figures from multirun data
├── paper/
│   ├── main.tex                           # Paper source
│   └── references.bib                     # Bibliography
├── figures                                # Generated figures (after running)
├── results_multirun_whitened/             # Whitened aggregate JSON + report (after running)
├── results_multirun_spectral/             # Spectral aggregate JSON + report (after running)
├── requirements.txt
└── README.md
```

## Requirements

- Python ≥ 3.10
- CPU only, 16 GB RAM
- ~2 GB disk for GPT-2 model download (first run)

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch, Transformers (HuggingFace), scikit-learn, NumPy, SciPy, matplotlib

> **Note:** `requirements.txt` also lists `wordfreq` for compatibility with the shared `hallucination_stats.py` module across the trilogy. It is not used by any Paper 3 script and is not required for reproduction.

## Running

### Multi-run stability analysis (recommended)

The multi-run wrappers are the primary entry points. Each runs its experiment K times with different generation seeds, then aggregates results.

```bash
# Experiment 1: Whitened contextual (N=30 prompts/group)
# Edit K at line 43 of run_multirun_whitened.py (default: 20)
python run_multirun_whitened.py
# Output: ./results_multirun_whitened/

# Experiment 2: Spectral band decomposition (N=15 prompts/group)
# Edit K at line 46 of run_multirun_spectral.py (default: 20)
python run_multirun_spectral.py
# Output: ./results_multirun_spectral/
```

Runtime at K=20 on Intel i7-6700 (4 cores, 16 GB RAM):
- Whitened: ~310 s/run × 20 = **~6,450 s (~1.8 h)**
- Spectral: ~4,270 s/run × 20 = **~85,490 s (~23.7 h)**

### Individual experiments (single run)

The pipeline scripts can also be run standalone for a single generation:

```bash
# Experiment 1: PCA-whitened contextual
python hallucination_induction_whitened.py
# Output: ./results_induction_whitened/

# Experiment 2: Spectral band decomposition
python hallucination_induction_spectral.py
# Output: ./results_induction_spectral/
```

### Figures

After running both multi-run analyses:

```bash
python generate_figures.py
# Output: ./figures/
```

Produces four publication figures:
- `fig_maxsim_ordering.pdf` — Whitened max_sim condition ordering + effect-size stability (main text)
- `fig_h_collapse.pdf` — H(v) false positive collapse vs. max_sim emergence (main text)
- `fig_spectral_heatmap.pdf` — Spectral band decomposition significance heatmap (appendix)
- `fig_discordance.pdf` — Token–prompt discordance scatter across all experiments (appendix)

## Experimental Design

**Model:** GPT-2-small (124M parameters, `gpt2`), CPU only, temperature 1.0, no top-k/top-p. Manual autoregressive generation with KV cache, extracting last-layer last-position hidden states at each step.

**Experiment 1 (Whitened):** 30 prompts per condition × 60 tokens each. PCA whitening (256 components, ε=10⁻⁵) on calibration distribution (40 prompts). Clustering: MiniBatchKMeans (k=40). Metrics: whitened H(v), max_sim, whitened norm, raw norm (replication control).

**Experiment 2 (Spectral):** 15 prompts per condition × 60 tokens each. Full PCA (768 components) on calibration. Six logarithmically spaced bands (PCs 1–16 through PCs 513–768). Per-band whitening, adapted clustering (k = min(40, band_dim/2), minimum 10), sliding window scan (width 64, step 32, Bonferroni correction).

**Two-level inference:** Prompt-level means are the unit of analysis, not individual tokens. This eliminates pseudoreplication from within-prompt autocorrelation. Token-level results reported as reference only.

**Multi-run protocol:** Calibration is performed once (seed 42). Generation is repeated K times with seeds 1…K. Statistical analysis (permutation tests, bootstrap CIs) uses a fixed internal seed for determinism.

## Key Outputs

**`run_multirun_whitened.py`** produces:
- `multirun_whitened_aggregate.json` — full per-run statistics + cross-run aggregates
- `multirun_whitened_report.txt` — human-readable stability report
- `raw_results_whitened.json` — per-token data from representative run (for figures)
- `summary_whitened.json` — zone thresholds from representative run

**`run_multirun_spectral.py`** produces:
- `multirun_spectral_aggregate.json` — full per-run statistics + cross-run aggregates
- `multirun_spectral_report.txt` — human-readable stability report

**`hallucination_induction_whitened.py`** produces:
- `induction_whitened_report.txt` — full results with confusion matrix
- `raw_results_whitened.json` — per-token geometric measurements
- Diagnostic figures (norm–membership scatter, trajectories, distributions, raw vs. whitened comparison)

**`hallucination_induction_spectral.py`** produces:
- `induction_spectral_report.txt` — full results with per-band tests
- `summary_spectral.json` — machine-readable per-band results
- Diagnostic figures (band significance, sliding window scan, eigenspectrum)

## Reproducibility

Calibration and statistical analysis use `random_state=42`. Text generation is stochastic by design (temperature 1.0); the multi-run protocol quantifies this variance rather than hiding it behind a fixed seed. No GPU required. All experiments run on a single Intel Core i7-6700 (3.40 GHz, 4 cores, 8 threads, 16 GB RAM) under Ubuntu Linux.

## License

Apache 2.0
