#!/usr/bin/env python3
"""
Multi-Run Stability Analysis for Spectral Band Hallucination Induction
========================================================================

Runs the spectral band decomposition pipeline K times with different
generation seeds, then aggregates results to quantify which findings
are stable across runs vs. seed-dependent.

Architecture
------------
  - Calibration (background generation, full PCA) is performed ONCE
    with a fixed seed (42).
  - Text generation is repeated K times with seeds 1..K.
  - For each run, fixed-band analysis and sliding-window scan are
    performed on that run's experimental hidden states against the
    fixed calibration PCA.
  - Statistical tests use a fixed internal seed (42 in
    hallucination_stats.py), so analysis is deterministic given
    the same generated data.

This isolates generation stochasticity as the sole source of
run-to-run variation.

Output (in ./results_multirun_spectral/)
-----------------------------------------
  multirun_spectral_aggregate.json  — full per-run + aggregate statistics
  multirun_spectral_report.txt      — human-readable stability report

Usage
-----
  python run_multirun_spectral.py

Runtime estimate: ~30–40 min × K
  K=10  → ~6 h   |   K=20  → ~12 h

Prerequisites
-------------
  hallucination_induction_spectral.py, hallucination_stats.py
  must be in the same directory.
"""

# ═══════════════════════════════════════════════════════════════
#  CHANGE THESE VALUES
# ═══════════════════════════════════════════════════════════════
K = 20
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

OUTPUT_DIR = './results_multirun_spectral'


# ──────────────────────────────────────────────────────────────
# SPECTRAL EXPERIMENT — K RUNS
# ──────────────────────────────────────────────────────────────

def run_spectral_multirun(K, spec_mod):
    """Run the spectral band pipeline K times with fixed calibration + PCA.

    Calibration (background generation, full PCA) is run once under
    seed 42.  Only experimental generation varies across runs.
    Band analysis and sliding-window scan are then re-run per run
    using the fixed PCA against that run's experimental data.
    """
    import torch
    print("=" * 70)
    print(f"SPECTRAL EXPERIMENT — {K} RUNS")
    print("=" * 70)
    t_exp = time.time()

    # ── Calibration (once, fixed seed) ──
    np.random.seed(42)
    torch.manual_seed(42)

    model, tokenizer, static_emb = spec_mod.load_gpt2()

    calibration_hidden = spec_mod.build_calibration(model, tokenizer)
    mean, pca, eigenvalues, cumvar = spec_mod.compute_full_pca(
        calibration_hidden)

    max_available = len(eigenvalues)

    # ── K generation runs ──
    runs = []
    for k in range(K):
        seed = k + 1
        print(f"\n{'─' * 60}")
        print(f"  SPECTRAL run {k + 1}/{K}  (seed={seed})")
        print(f"{'─' * 60}")
        t0 = time.time()

        torch.manual_seed(seed)

        # Generate experimental sequences
        all_experimental = []
        for prompts, label in [(spec_mod.TYPE1_PROMPTS, 'type1'),
                               (spec_mod.TYPE2_PROMPTS, 'type2'),
                               (spec_mod.TYPE3_PROMPTS, 'type3')]:
            results = spec_mod.generate_experimental(
                model, tokenizer, prompts, label)
            all_experimental.extend(results)

        # ── Fixed-band analysis ──
        band_data = []
        for start, end, band_label in spec_mod.CONFIG['bands']:
            if start >= max_available:
                continue
            end = min(end, max_available - 1)
            br = spec_mod.analyze_band(
                calibration_hidden, all_experimental, mean, pca,
                eigenvalues, start, end, band_label)
            spec_mod.add_legacy_compat(br)

            # Extract serialisable two-level stats
            two_level_json = results_to_json(
                {k_: v for k_, v in br.items()
                 if not k_.startswith('_') and k_ not in
                 ('band_label', 'band_start', 'band_end',
                  'n_pcs', 'n_clusters', 'variance_in_band',
                  'tests', 'pairwise', 'means')})

            band_data.append({
                'meta': br.get('_meta', {}),
                'two_level_stats': two_level_json,
                'legacy_pairwise': br.get('pairwise', {}),
            })

        # ── Sliding-window scan ──
        scan_data = []
        scan_results = spec_mod.run_sliding_scan(
            calibration_hidden, all_experimental, mean, pca, eigenvalues,
            spec_mod.CONFIG['scan_width'],
            spec_mod.CONFIG['scan_step'],
            spec_mod.CONFIG['scan_max_pc'])

        for sr in scan_results:
            spec_mod.add_legacy_compat(sr)
            scan_two_level = results_to_json(
                {k_: v for k_, v in sr.items()
                 if not k_.startswith('_') and k_ not in
                 ('band_label', 'band_start', 'band_end',
                  'n_pcs', 'n_clusters', 'variance_in_band',
                  'tests', 'pairwise', 'means')})
            scan_data.append({
                'meta': sr.get('_meta', {}),
                'two_level_stats': scan_two_level,
                'legacy_pairwise': sr.get('pairwise', {}),
            })

        run_data = {
            'seed': seed,
            'bands': band_data,
            'scan': scan_data,
            'runtime_s': round(time.time() - t0, 1),
        }
        runs.append(run_data)

        del all_experimental, band_data, scan_data, scan_results
        gc.collect()

        print(f"  ✓ Spectral run {k + 1} done in {run_data['runtime_s']:.0f}s")

    del model
    gc.collect()

    # Save eigenspectrum info (constant across runs)
    eigen_info = {
        'eigenvalues_top100': eigenvalues[:100].tolist(),
        'cumvar_checkpoints': {
            str(n): float(cumvar[n - 1]) for n in [10, 50, 100, 256, 512]
            if n <= len(cumvar)
        },
        'n_components': int(max_available),
    }

    print(f"\n  Spectral experiment total: {time.time() - t_exp:.0f}s")
    return runs, eigen_info


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
                'iqr_hi': float('nan'), 'values': []}
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


def _aggregate_band_two_level(band_stats_across_runs):
    """Aggregate two-level stats from K runs for a single band.

    band_stats_across_runs: list of K dicts, each being the
    two_level_stats for one run at one band.
    """
    metrics = ['H', 'norm', 'max_sim']
    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    conditions = ['type1', 'type2', 'type3']

    agg = {}
    for m in metrics:
        m_agg = {}

        # Omnibus KW p-values
        for level, path in [
            ('token_kw',           ('token_kw', 'p')),
            ('prompt_kw',          ('prompt_mean_kw', 'p')),
            ('prompt_kw_perm',     ('prompt_mean_kw_perm', 'p')),
            ('prompt_kw_median',   ('prompt_median_kw', 'p')),
        ]:
            vals = [_get(r, m, *path) for r in band_stats_across_runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            m_agg[level] = s

        # Pairwise tests
        for pair in pairs:
            pair_agg = {}

            # Prompt-level MW p
            vals = [_get(r, m, 'pairwise', pair, 'prompt_mean_mw', 'p')
                    for r in band_stats_across_runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['prompt_mw_p'] = s

            # Permutation p
            vals = [_get(r, m, 'pairwise', pair, 'prompt_mean_perm', 'p')
                    for r in band_stats_across_runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['perm_p'] = s

            # Token-level MW p
            vals = [_get(r, m, 'pairwise', pair, 'token', 'p')
                    for r in band_stats_across_runs]
            s = _summarise(vals)
            s['prop_sig'] = _prop_sig(vals)
            pair_agg['token_p'] = s

            # Rank-biserial r
            vals = [_get(r, m, 'pairwise', pair, 'r_ci', 'r')
                    for r in band_stats_across_runs]
            pair_agg['r'] = _summarise(vals)

            # Holm survival rate
            holm_survived = []
            for r in band_stats_across_runs:
                holm_list = _get(r, m, 'holm', default=[])
                survived = False
                for entry in holm_list:
                    if isinstance(entry, (list, tuple)) and pair in str(entry[0]):
                        survived = bool(entry[3])
                        break
                holm_survived.append(survived)
            pair_agg['holm_prop_sig'] = float(np.mean(holm_survived))

            m_agg[pair] = pair_agg

        # Condition means
        for cond in conditions:
            vals = [_get(r, m, 'condition_stats', cond, 'prompt_mean')
                    for r in band_stats_across_runs]
            m_agg[f'{cond}_prompt_mean'] = _summarise(vals)

        agg[m] = m_agg

    # Pseudoreplication diagnostic
    pseudo = {}
    for m in metrics:
        tok_rates = []
        pr_rates = []
        for pair in pairs:
            tok_ps = [_get(r, m, 'pairwise', pair, 'token', 'p')
                      for r in band_stats_across_runs]
            pr_ps = [_get(r, m, 'pairwise', pair, 'prompt_mean_mw', 'p')
                     for r in band_stats_across_runs]
            tok_rates.append(_prop_sig(tok_ps))
            pr_rates.append(_prop_sig(pr_ps))
        pseudo[m] = {
            'token_sig_rate_per_pair': tok_rates,
            'prompt_sig_rate_per_pair': pr_rates,
            'pairs': pairs,
        }
    agg['pseudoreplication'] = pseudo

    return agg


def aggregate_spectral_runs(runs):
    """Aggregate K spectral runs.

    Returns per-band and per-scan-window aggregations of two-level
    test results across runs.
    """
    K = len(runs)

    # Determine band count from first run
    n_bands = len(runs[0]['bands'])
    n_scan = len(runs[0]['scan'])

    # ── Per-band aggregation ──
    band_agg = []
    for b_idx in range(n_bands):
        band_stats = [r['bands'][b_idx]['two_level_stats'] for r in runs]
        meta = runs[0]['bands'][b_idx]['meta']
        band_agg.append({
            'meta': meta,
            'aggregate': _aggregate_band_two_level(band_stats),
        })

    # ── Per-scan-window aggregation ──
    scan_agg = []
    for s_idx in range(n_scan):
        scan_stats = [r['scan'][s_idx]['two_level_stats'] for r in runs]
        meta = runs[0]['scan'][s_idx]['meta']
        scan_agg.append({
            'meta': meta,
            'aggregate': _aggregate_band_two_level(scan_stats),
        })

    return {
        'K': K,
        'seeds': [r['seed'] for r in runs],
        'runtimes_s': [r['runtime_s'] for r in runs],
        'bands': band_agg,
        'scan': scan_agg,
    }


# ──────────────────────────────────────────────────────────────
# REPORT FORMATTING
# ──────────────────────────────────────────────────────────────

def format_report(spec_agg, eigen_info, K):
    """Generate human-readable stability report for the spectral experiment."""
    lines = []
    w = 76

    pair_short = {'type1_vs_type2': 'T1-T2',
                  'type1_vs_type3': 'T1-T3',
                  'type2_vs_type3': 'T2-T3'}
    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']

    lines.append("=" * w)
    lines.append("MULTI-RUN STABILITY ANALYSIS — SPECTRAL BAND DECOMPOSITION")
    lines.append(f"K = {K} independent generation runs")
    lines.append(f"Calibration + PCA: fixed (seed=42) | Generation seeds: 1..{K}")
    lines.append("=" * w)

    rts = spec_agg.get('runtimes_s', [])
    if rts:
        lines.append(f"  Runtime per run: "
                     f"mean={np.mean(rts):.0f}s, "
                     f"total={np.sum(rts):.0f}s")

    # ── Eigenspectrum summary ──
    lines.append(f"\n  EIGENSPECTRUM (fixed across runs)")
    for n, v in eigen_info.get('cumvar_checkpoints', {}).items():
        lines.append(f"    Top {n} PCs: {v:.1%} variance")

    # ── Per-band results ──
    for band in spec_agg['bands']:
        meta = band['meta']
        agg = band['aggregate']
        label = meta.get('band_label', '').replace('\n', ' ')
        start = meta.get('band_start', 0)
        end = meta.get('band_end', 0)
        n_pcs = meta.get('n_pcs', 0)
        var = meta.get('variance_in_band', 0)

        lines.append(f"\n{'=' * w}")
        lines.append(f"  BAND: {label}  (PCs {start+1}–{end+1}, "
                     f"{n_pcs} dims, {var:.1%} var)")
        lines.append(f"{'=' * w}")

        # Omnibus
        lines.append(f"\n  OMNIBUS KW (prompt-level)")
        lines.append(f"  {'Metric':<10} {'Med p':>10} "
                     f"{'Sig rate':>10} {'Perm sig':>10}")
        lines.append(f"  {'-' * 42}")

        for m in ['H', 'norm', 'max_sim']:
            pkw = agg[m]['prompt_kw']
            ppkw = agg[m]['prompt_kw_perm']
            lines.append(
                f"  {m:<10} {pkw['median']:>10.4f} "
                f"{pkw.get('prop_sig', float('nan')):>8.0%}   "
                f"{ppkw.get('prop_sig', float('nan')):>8.0%}")

        # Pairwise
        lines.append(f"\n  PAIRWISE (prompt-level MW)")
        lines.append(
            f"  {'Metric':<8} {'Pair':<10} {'Med p':>8} "
            f"{'Sig':>6} {'Holm':>6} {'Med r':>8}")
        lines.append(f"  {'-' * 50}")

        for m in ['H', 'norm', 'max_sim']:
            for pair in pairs:
                pa = agg[m][pair]
                mw = pa['prompt_mw_p']
                holm = pa['holm_prop_sig']
                r = pa['r']
                lines.append(
                    f"  {m:<8} {pair_short[pair]:<10} "
                    f"{mw['median']:>8.4f} "
                    f"{mw.get('prop_sig', float('nan')):>5.0%} "
                    f"{holm:>5.0%} "
                    f"{r['median']:>+8.3f}")
            lines.append("")

    # ── Sliding scan stability summary ──
    lines.append(f"\n{'=' * w}")
    lines.append(f"  SLIDING WINDOW SCAN STABILITY")
    lines.append(f"{'=' * w}")
    lines.append(f"  (Fraction of {K} runs where prompt-level KW is p<0.05)\n")
    lines.append(
        f"  {'Window':<20} {'H sig':>8} {'norm sig':>10} "
        f"{'max_sim sig':>12}")
    lines.append(f"  {'-' * 52}")

    for sw in spec_agg['scan']:
        meta = sw['meta']
        agg = sw['aggregate']
        start = meta.get('band_start', 0)
        end = meta.get('band_end', 0)
        label = f"PC {start+1}–{end+1}"
        h_sig = agg['H']['prompt_kw'].get('prop_sig', float('nan'))
        n_sig = agg['norm']['prompt_kw'].get('prop_sig', float('nan'))
        ms_sig = agg['max_sim']['prompt_kw'].get('prop_sig', float('nan'))
        lines.append(
            f"  {label:<20} {h_sig:>7.0%} {n_sig:>9.0%} {ms_sig:>11.0%}")

    # ── Cross-band stability summary ──
    lines.append(f"\n{'=' * w}")
    lines.append(f"  CROSS-BAND STABILITY SUMMARY")
    lines.append(f"{'=' * w}")
    lines.append(f"\n  Key claims and their stability across {K} runs:\n")

    for band in spec_agg['bands']:
        meta = band['meta']
        agg = band['aggregate']
        label = meta.get('band_label', '').replace('\n', ' ')

        # Identify which pairs have stable separation
        stable_pairs = []
        for m in ['H', 'norm', 'max_sim']:
            for pair in pairs:
                pa = agg[m][pair]
                sig_rate = pa['prompt_mw_p'].get('prop_sig', 0)
                if sig_rate >= 0.8:
                    stable_pairs.append(
                        f"{m} {pair_short[pair]}: "
                        f"sig {sig_rate:.0%}, "
                        f"Holm {pa['holm_prop_sig']:.0%}")

        if stable_pairs:
            lines.append(f"  {label}:")
            for sp in stable_pairs:
                lines.append(f"    ✓ {sp}")
        else:
            lines.append(f"  {label}: no stably significant pairs")

    # Type 1/2 spectral mixing hypothesis check
    lines.append(f"\n  SPECTRAL MIXING HYPOTHESIS:")
    t12_any_stable = False
    for band in spec_agg['bands']:
        meta = band['meta']
        agg = band['aggregate']
        label = meta.get('band_label', '').replace('\n', ' ')
        for m in ['H', 'norm', 'max_sim']:
            pa = agg[m]['type1_vs_type2']
            sig_rate = pa['prompt_mw_p'].get('prop_sig', 0)
            if sig_rate >= 0.5:
                lines.append(
                    f"    {label} {m} T1-T2: sig {sig_rate:.0%} "
                    f"(r={pa['r']['median']:+.3f})")
                t12_any_stable = True

    if not t12_any_stable:
        lines.append(
            "    REJECTED — Type 1/2 does not survive prompt-level "
            "aggregation in any spectral band across runs")
    else:
        lines.append(
            "    PARTIAL SUPPORT — some bands show T1-T2 separation "
            "in a fraction of runs")

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
    print(f"MULTI-RUN STABILITY ANALYSIS — SPECTRAL  (K = {K})")
    print(f"Spectral band decomposition × {K} generation runs")
    est_min = K * 35
    print(f"Estimated runtime: ~{est_min // 60}h {est_min % 60}m")
    print("=" * 70)

    t_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import pipeline module (deferred to avoid torch import at parse time)
    import hallucination_induction_spectral as spec_mod

    # Disable figure generation in the module
    spec_mod.CONFIG['generate_figures'] = False

    # ── Run experiment ──
    spec_runs, eigen_info = run_spectral_multirun(K, spec_mod)

    # ── Aggregate ──
    print(f"\n{'=' * 70}")
    print("AGGREGATING RESULTS")
    print(f"{'=' * 70}")

    spec_agg = aggregate_spectral_runs(spec_runs)

    # ── Save JSON ──
    output = {
        'K': K,
        'total_runtime_s': round(time.time() - t_total, 1),
        'eigenspectrum': eigen_info,
        'spectral': {
            'aggregate': spec_agg,
            'runs': spec_runs,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, 'multirun_spectral_aggregate.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Aggregate saved to {json_path}")

    # ── Generate report ──
    report = format_report(spec_agg, eigen_info, K)
    report_path = os.path.join(OUTPUT_DIR, 'multirun_spectral_report.txt')
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
