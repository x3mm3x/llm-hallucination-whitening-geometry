#!/usr/bin/env python3
"""
Multi-Run Stability Analysis for Whitened Hallucination Induction
==================================================================

Runs the whitened contextual hallucination induction pipeline K times
with different generation seeds, then aggregates results to quantify
which findings are stable across runs vs. seed-dependent.

Architecture
------------
  - Calibration (background generation, whitening transform, clustering,
    zone thresholds) is performed ONCE with a fixed seed (42).
  - Text generation is repeated K times with seeds 1..K.
  - Statistical tests use a fixed internal seed (42 in
    hallucination_stats.py), so analysis is deterministic given
    the same generated data.

This isolates generation stochasticity as the sole source of
run-to-run variation.

Output (in ./results_multirun_whitened/)
-----------------------------------------
  multirun_whitened_aggregate.json  — full per-run + aggregate statistics
  multirun_whitened_report.txt      — human-readable stability report

Usage
-----
  python run_multirun_whitened.py

Runtime estimate: ~10–20 min × K  (with KV cache)
  K=10  → ~2–3 h   |   K=20  → ~4–6 h

Prerequisites
-------------
  hallucination_induction_whitened.py, hallucination_stats.py
  must be in the same directory.
"""

# ═══════════════════════════════════════════════════════════════
#  CHANGE THESE VALUES
# ═══════════════════════════════════════════════════════════════
K = 20
REPRESENTATIVE_SEED = 7   # Save raw per-token data for this seed (for figures)
# ═══════════════════════════════════════════════════════════════

import os
import sys
import time
import json
import gc
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from hallucination_stats import results_to_json

OUTPUT_DIR = './results_multirun_whitened'


def _save_representative(all_results, zones, experiment_name):
    """Save raw per-token results and zones from one representative run.

    These files are needed by generate_figures.py for the scatter plot.
    """
    raw_path = os.path.join(
        OUTPUT_DIR, f'raw_results_{experiment_name}.json')
    summary_path = os.path.join(
        OUTPUT_DIR, f'summary_{experiment_name}.json')

    # Serialise raw results (strip numpy types)
    raw = []
    for seq in all_results:
        entry = {
            'type': seq['type'],
            'prompt': seq['prompt'],
            'measurements': [],
        }
        for m in seq.get('measurements', []):
            entry['measurements'].append({
                k: (float(v) if hasattr(v, 'item') else v)
                for k, v in m.items()
            })
        raw.append(entry)

    with open(raw_path, 'w') as f:
        json.dump(raw, f)

    # Serialise zones (threshold dict)
    zones_ser = {k: float(v) if hasattr(v, 'item') else v
                 for k, v in zones.items()}
    with open(summary_path, 'w') as f:
        json.dump({'zones': zones_ser}, f, indent=2)

    print(f"  ✓ Saved representative raw data: {raw_path}")
    print(f"  ✓ Saved representative zones: {summary_path}")


# ──────────────────────────────────────────────────────────────
# WHITENED EXPERIMENT — K RUNS
# ──────────────────────────────────────────────────────────────

def run_whitened_multirun(K, wht_mod):
    """Run the whitened contextual pipeline K times with fixed calibration.

    Calibration (background generation, whitening transform, clustering,
    zone thresholds) is run once under seed 42.  Only experimental
    generation varies across runs (seeds 1..K).
    """
    import torch
    print("=" * 70)
    print(f"WHITENED EXPERIMENT — {K} RUNS")
    print(f"Prompts per type: {wht_mod.CONFIG['n_prompts_per_type']}")
    print(f"Calibration prompts: {len(wht_mod.CALIBRATION_PROMPTS)}")
    print("=" * 70)
    t_exp = time.time()

    # ── Calibration (once, fixed seed) ──
    np.random.seed(42)
    torch.manual_seed(42)

    model, tokenizer, static_emb = wht_mod.load_gpt2()

    calibration_hidden = wht_mod.build_calibration_distribution(
        model, tokenizer)

    whiten_mean, whiten_W, var_explained, whitened_calib = \
        wht_mod.compute_whitening_transform(
            calibration_hidden,
            wht_mod.CONFIG['whiten_n_components'],
            wht_mod.CONFIG['whiten_regularization'])

    kmeans, centroids, zones, calib_stats = \
        wht_mod.cluster_and_calibrate_whitened(whitened_calib)

    del calibration_hidden, whitened_calib
    gc.collect()

    # ── K generation runs ──
    runs = []
    for k in range(K):
        seed = k + 1
        print(f"\n{'─' * 60}")
        print(f"  WHITENED run {k + 1}/{K}  (seed={seed})")
        print(f"{'─' * 60}")
        t0 = time.time()

        torch.manual_seed(seed)

        gen1 = wht_mod.generate_experimental(
            model, tokenizer, wht_mod.TYPE1_PROMPTS, 'type1')
        gen2 = wht_mod.generate_experimental(
            model, tokenizer, wht_mod.TYPE2_PROMPTS, 'type2')
        gen3 = wht_mod.generate_experimental(
            model, tokenizer, wht_mod.TYPE3_PROMPTS, 'type3')
        all_results = gen1 + gen2 + gen3

        all_results = wht_mod.measure_geometry(
            all_results, centroids, zones, whiten_mean, whiten_W)
        confusion = wht_mod.compute_confusion_matrix(all_results)
        stat_tests, diagnostics = wht_mod.run_statistical_tests(all_results)

        run_data = {
            'seed': seed,
            'two_level_stats': results_to_json(stat_tests),
            'confusion': {
                'fractions': confusion['fractions'],
                'diagonal': [float(d) for d in confusion['diagonal']],
                'mean_diagonal': float(confusion['mean_diagonal']),
                'totals': {t: int(confusion['totals'][t])
                           for t in confusion['totals']},
            },
            'runtime_s': round(time.time() - t0, 1),
        }
        runs.append(run_data)

        # Save raw per-token data for representative run (for scatter fig)
        if seed == REPRESENTATIVE_SEED:
            _save_representative(all_results, zones, 'whitened')

        del all_results, gen1, gen2, gen3, confusion, stat_tests, diagnostics
        gc.collect()

        print(f"  ✓ Whitened run {k + 1} done in {run_data['runtime_s']:.0f}s")

    del model
    gc.collect()

    print(f"\n  Whitened experiment total: {time.time() - t_exp:.0f}s")
    return runs


# ──────────────────────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────────────────────

def _get(d, *keys, default=float('nan')):
    """Safely drill into nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, None)
        else:
            return default
        if d is None:
            return default
    return d


def _summarise(values):
    """Compute summary statistics for a list of floats."""
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return {'n': 0, 'median': float('nan'), 'mean': float('nan'),
                'std': float('nan'), 'iqr_lo': float('nan'),
                'iqr_hi': float('nan'), 'prop_sig': float('nan'),
                'values': []}
    return {
        'n': int(len(v)),
        'median': float(np.median(v)),
        'mean': float(np.mean(v)),
        'std': float(np.std(v)),
        'iqr_lo': float(np.percentile(v, 25)),
        'iqr_hi': float(np.percentile(v, 75)),
        'values': [float(x) for x in v],
    }


def _prop_sig(values, alpha=0.05):
    """Fraction of values below alpha."""
    v = [x for x in values if np.isfinite(x)]
    return float(np.mean([p < alpha for p in v])) if v else float('nan')


def aggregate_runs(runs, experiment_name):
    """Aggregate K runs into stability statistics.

    Returns a dict with:
      - per-metric omnibus p-value distributions
      - per-pair prompt-level p, effect size, Holm survival rates
      - confusion matrix mean/std
      - per-run data preserved for reference
    """
    K = len(runs)
    metrics = ['H', 'norm', 'max_sim']
    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    conditions = ['type1', 'type2', 'type3']

    # Derive N per group from first run
    n_per_group = _get(runs[0], 'two_level_stats', 'H',
                       'condition_stats', 'type1', 'n_prompts', default=0)

    agg = {
        'experiment': experiment_name,
        'K': K,
        'n_per_group': int(n_per_group),
        'seeds': [r['seed'] for r in runs],
        'runtimes_s': [r['runtime_s'] for r in runs],
    }

    # ── Per-metric aggregation ──
    for m in metrics:
        m_agg = {}

        # Omnibus KW p-values
        for level, path in [
            ('token_kw',           ('token_kw', 'p')),
            ('prompt_kw',          ('prompt_mean_kw', 'p')),
            ('prompt_kw_perm',     ('prompt_mean_kw_perm', 'p')),
            ('prompt_kw_median',   ('prompt_median_kw', 'p')),
        ]:
            vals = [_get(r, 'two_level_stats', m, *path) for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            m_agg[level] = s

        # Pairwise tests
        for pair in pairs:
            pair_agg = {}

            # Prompt-level MW p
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'prompt_mean_mw', 'p') for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['prompt_mw_p'] = s

            # Permutation p
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'prompt_mean_perm', 'p') for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['perm_p'] = s

            # Token-level MW p (for pseudoreplication comparison)
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'token', 'p') for r in runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['token_p'] = s

            # Rank-biserial r
            vals = [_get(r, 'two_level_stats', m, 'pairwise',
                         pair, 'r_ci', 'r') for r in runs]
            pair_agg['r'] = _summarise(vals)

            # Diff CI
            vals_lo = [_get(r, 'two_level_stats', m, 'pairwise',
                            pair, 'diff_ci', 'lo') for r in runs]
            vals_hi = [_get(r, 'two_level_stats', m, 'pairwise',
                            pair, 'diff_ci', 'hi') for r in runs]
            pair_agg['diff_ci_lo'] = _summarise(vals_lo)
            pair_agg['diff_ci_hi'] = _summarise(vals_hi)

            # Holm survival rate
            holm_survived = []
            for r in runs:
                holm_list = _get(r, 'two_level_stats', m, 'holm',
                                 default=[])
                survived = False
                for entry in holm_list:
                    if isinstance(entry, (list, tuple)) and pair in str(entry[0]):
                        survived = bool(entry[3])
                        break
                holm_survived.append(survived)
            pair_agg['holm_prop_sig'] = float(np.mean(holm_survived))
            pair_agg['holm_values'] = holm_survived

            m_agg[pair] = pair_agg

        # Condition means
        for cond in conditions:
            vals = [_get(r, 'two_level_stats', m, 'condition_stats',
                         cond, 'prompt_mean') for r in runs]
            m_agg[f'{cond}_prompt_mean'] = _summarise(vals)

        agg[m] = m_agg

    # ── Confusion matrix aggregation ──
    cm_agg = {}
    for typ in conditions:
        for zone in conditions + ['unclassified']:
            vals = [_get(r, 'confusion', 'fractions', typ, zone)
                    for r in runs]
            cm_agg[f'{typ}->{zone}'] = _summarise(vals)

    diags = [_get(r, 'confusion', 'mean_diagonal') for r in runs]
    cm_agg['mean_diagonal'] = _summarise(diags)
    agg['confusion'] = cm_agg

    # ── Pseudoreplication diagnostic ──
    pseudo = {}
    for m in metrics:
        tok_rates = []
        pr_rates = []
        for pair in pairs:
            tok_ps = [_get(r, 'two_level_stats', m, 'pairwise',
                           pair, 'token', 'p') for r in runs]
            pr_ps = [_get(r, 'two_level_stats', m, 'pairwise',
                          pair, 'prompt_mean_mw', 'p') for r in runs]
            tok_rates.append(_prop_sig(tok_ps))
            pr_rates.append(_prop_sig(pr_ps))
        pseudo[m] = {
            'token_sig_rate_per_pair': tok_rates,
            'prompt_sig_rate_per_pair': pr_rates,
            'pairs': pairs,
        }
    agg['pseudoreplication'] = pseudo

    return agg


# ──────────────────────────────────────────────────────────────
# REPORT FORMATTING
# ──────────────────────────────────────────────────────────────

def _stability_label(rate):
    """Classify a significance rate into a stability category."""
    if rate >= 0.80:
        return 'STABLE'
    elif rate >= 0.50:
        return 'moderate'
    elif rate >= 0.10:
        return 'unstable'
    else:
        return 'null'


def format_report(wht_agg, K):
    """Generate human-readable stability report for the whitened experiment."""
    lines = []
    w = 76

    n_per_group = wht_agg.get('n_per_group', '?')

    lines.append("=" * w)
    lines.append("MULTI-RUN STABILITY ANALYSIS — WHITENED CONTEXTUAL")
    lines.append(f"K = {K} independent generation runs")
    lines.append(f"Prompts per group: N = {n_per_group}")
    lines.append(f"Calibration: fixed (seed=42) | Generation seeds: 1..{K}")
    lines.append("=" * w)

    agg = wht_agg
    label = 'WHITENED'

    lines.append(f"\n{'=' * w}")
    lines.append(f"  {label} EXPERIMENT")
    lines.append(f"{'=' * w}")

    rts = agg.get('runtimes_s', [])
    if rts:
        lines.append(f"  Runtime per run: "
                     f"mean={np.mean(rts):.0f}s, "
                     f"total={np.sum(rts):.0f}s")

    # ── Omnibus ──
    lines.append(f"\n  OMNIBUS KRUSKAL-WALLIS (prompt-level, N={n_per_group}/group)")
    lines.append(f"  {'Metric':<10} {'Median p':>10} "
                 f"{'IQR':>16} {'Sig rate':>10} "
                 f"{'Perm med':>10} {'Perm sig':>10}")
    lines.append(f"  {'-' * 68}")

    for m in ['H', 'norm', 'max_sim']:
        pkw = agg[m]['prompt_kw']
        ppkw = agg[m]['prompt_kw_perm']
        lines.append(
            f"  {m:<10} {pkw['median']:>10.4f} "
            f"[{pkw['iqr_lo']:.3f}, {pkw['iqr_hi']:.3f}] "
            f"{pkw['prop_sig']:>8.0%}   "
            f"{ppkw['median']:>10.4f} {ppkw['prop_sig']:>8.0%}")

    # ── Pairwise ──
    lines.append(f"\n  PAIRWISE (prompt-level Mann-Whitney)")
    lines.append(
        f"  {'Metric':<8} {'Pair':<16} {'Med p':>8} "
        f"{'Sig':>6} {'Holm':>6} {'Med r':>8} "
        f"{'r SD':>8} {'Perm sig':>10}")
    lines.append(f"  {'-' * 74}")

    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    pair_short = {'type1_vs_type2': 'T1-T2',
                  'type1_vs_type3': 'T1-T3',
                  'type2_vs_type3': 'T2-T3'}

    for m in ['H', 'norm', 'max_sim']:
        for pair in pairs:
            pa = agg[m][pair]
            mw = pa['prompt_mw_p']
            pm = pa['perm_p']
            r = pa['r']
            holm = pa['holm_prop_sig']
            lines.append(
                f"  {m:<8} {pair_short[pair]:<16} "
                f"{mw['median']:>8.4f} "
                f"{mw['prop_sig']:>5.0%} "
                f"{holm:>5.0%} "
                f"{r['median']:>+8.3f} "
                f"{r['std']:>8.3f} "
                f"{pm['prop_sig']:>8.0%}")
        lines.append("")

    # ── Condition means ──
    lines.append(f"  CONDITION MEANS (prompt-level, across runs)")
    lines.append(f"  {'Metric':<10} {'Type 1':>18} "
                 f"{'Type 2':>18} {'Type 3':>18}")
    lines.append(f"  {'-' * 66}")
    for m in ['H', 'norm', 'max_sim']:
        vals = []
        for cond in ['type1', 'type2', 'type3']:
            s = agg[m][f'{cond}_prompt_mean']
            vals.append(f"{s['mean']:.4f} +/- {s['std']:.4f}")
        lines.append(f"  {m:<10} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # ── Confusion ──
    lines.append(f"\n  CONFUSION MATRIX (mean +/- std across {K} runs)")
    zones = ['type1', 'type2', 'type3', 'unclassified']
    lines.append(f"  {'':>10} {'Z1':>14} {'Z2':>14} "
                 f"{'Z3':>14} {'Unc':>14}")
    lines.append(f"  {'-' * 68}")
    for typ in ['type1', 'type2', 'type3']:
        cells = []
        for zone in zones:
            s = agg['confusion'][f'{typ}->{zone}']
            cells.append(f"{s['mean']:.3f}+/-{s['std']:.3f}")
        lines.append(f"  {typ:>10} {cells[0]:>14} {cells[1]:>14} "
                     f"{cells[2]:>14} {cells[3]:>14}")

    md = agg['confusion']['mean_diagonal']
    lines.append(f"  Mean diagonal: {md['mean']:.3f} +/- {md['std']:.3f}")

    # ── Pseudoreplication diagnostic ──
    lines.append(f"\n  PSEUDOREPLICATION DIAGNOSTIC")
    lines.append(f"  (fraction of runs where comparison is significant)")
    lines.append(f"  {'Metric':<10} {'Pair':<16} "
                 f"{'Token sig':>12} {'Prompt sig':>12} {'Ratio':>8}")
    lines.append(f"  {'-' * 60}")
    for m in ['H', 'norm', 'max_sim']:
        ps = agg['pseudoreplication'][m]
        for i, pair in enumerate(ps['pairs']):
            tok = ps['token_sig_rate_per_pair'][i]
            pr = ps['prompt_sig_rate_per_pair'][i]
            ratio = f"{tok/pr:.1f}x" if pr > 0 else ("—" if tok == 0 else "inf")
            lines.append(
                f"  {m:<10} {pair_short[pair]:<16} "
                f"{tok:>10.0%}   {pr:>10.0%}   {ratio:>8}")
        lines.append("")

    # ── Data-driven stability summary ──
    lines.append("=" * w)
    lines.append("  STABILITY SUMMARY")
    lines.append("=" * w)

    lines.append(f"\n  Key claims and their stability across {K} runs "
                 f"(N={n_per_group}/group):\n")

    # Collect all pairwise results for summary
    summary_items = []
    for m in ['H', 'norm', 'max_sim']:
        for pair in pairs:
            pa = agg[m][pair]
            mw_sig = pa['prompt_mw_p']['prop_sig']
            holm_sig = pa['holm_prop_sig']
            med_r = pa['r']['median']
            label = _stability_label(holm_sig)
            summary_items.append((m, pair_short[pair], mw_sig,
                                  holm_sig, med_r, label))

    # Type 3 separations
    lines.append("  TYPE 3 SEPARATIONS:")
    for m, pair, mw_sig, holm_sig, med_r, label in summary_items:
        if 'T3' in pair:
            lines.append(
                f"    {m:<8} {pair:<8} MW sig={mw_sig:>5.0%}  "
                f"Holm={holm_sig:>5.0%}  r={med_r:>+.3f}  [{label}]")

    # Type 1/2 non-separations
    lines.append("\n  TYPE 1/2 NON-SEPARATION:")
    for m, pair, mw_sig, holm_sig, med_r, label in summary_items:
        if pair == 'T1-T2':
            lines.append(
                f"    {m:<8} {pair:<8} MW sig={mw_sig:>5.0%}  "
                f"Holm={holm_sig:>5.0%}  r={med_r:>+.3f}  [{label}]")

    # Overall verdict
    t3_holm_rates = [holm for m, pair, mw, holm, r, lab in summary_items
                     if 'T3' in pair]
    t12_holm_rates = [holm for m, pair, mw, holm, r, lab in summary_items
                      if pair == 'T1-T2']
    best_t3 = max(t3_holm_rates) if t3_holm_rates else 0
    best_t12 = max(t12_holm_rates) if t12_holm_rates else 0

    lines.append(f"\n  VERDICT:")
    lines.append(f"    Best Type 3 Holm rate:   {best_t3:>5.0%}  "
                 f"({'CONFIRMED' if best_t3 >= 0.50 else 'WEAK'})")
    lines.append(f"    Best Type 1/2 Holm rate: {best_t12:>5.0%}  "
                 f"({'SEPARATED' if best_t12 >= 0.50 else 'NON-SEPARATION CONFIRMED'})")

    lines.append("")
    lines.append("=" * w)
    lines.append("END OF MULTI-RUN REPORT")
    lines.append("=" * w)

    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"MULTI-RUN STABILITY ANALYSIS — WHITENED  (K = {K})")
    print(f"Whitened contextual × {K} generation runs")
    est_min = K * 15  # ~15 min with KV cache and N=30
    print(f"Estimated runtime: ~{est_min // 60}h {est_min % 60}m")
    print("=" * 70)

    t_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import pipeline module (deferred to avoid torch import at parse time)
    import hallucination_induction_whitened as wht_mod

    # Disable figure generation in the module
    wht_mod.CONFIG['generate_figures'] = False

    # ── Run experiment ──
    wht_runs = run_whitened_multirun(K, wht_mod)

    # ── Aggregate ──
    print(f"\n{'=' * 70}")
    print("AGGREGATING RESULTS")
    print(f"{'=' * 70}")

    wht_agg = aggregate_runs(wht_runs, 'whitened')

    # ── Save JSON ──
    output = {
        'K': K,
        'n_per_group': wht_agg.get('n_per_group', 0),
        'total_runtime_s': round(time.time() - t_total, 1),
        'whitened': {
            'aggregate': wht_agg,
            'runs': wht_runs,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, 'multirun_whitened_aggregate.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Aggregate saved to {json_path}")

    # ── Generate report ──
    report = format_report(wht_agg, K)
    report_path = os.path.join(OUTPUT_DIR, 'multirun_whitened_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # ── Print report ──
    total_s = time.time() - t_total
    print(f"\n  Total runtime: {total_s / 3600:.1f}h ({total_s:.0f}s)")
    print(f"  Results in: {os.path.abspath(OUTPUT_DIR)}")
    print()
    print(report)


if __name__ == '__main__':
    main()
