#!/usr/bin/env python3
"""
Generate figures for Paper 3 — CPU Trilogy
==========================================

Reads multirun aggregate JSONs and representative raw data.
Produces 4 figures: 2 main paper, 2 appendix.

Data files expected (adjust paths below):
  - results_multirun_whitened/multirun_whitened_aggregate.json
  - results_multirun_whitened/raw_results_whitened.json
  - results_multirun_spectral/multirun_spectral_aggregate.json

Usage:
  python generate_paper3_figures.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — adjust paths as needed
# ══════════════════════════════════════════════════════════════

WHITENED_AGG = './results_multirun_whitened/multirun_whitened_aggregate.json'
RAW_RESULTS  = './results_multirun_whitened/raw_results_whitened.json'
SPECTRAL_AGG = './results_multirun_spectral/multirun_spectral_aggregate.json'
OUTPUT_DIR   = './figures'
DPI          = 300

# Tol colorblind-safe palette
C1 = '#4477AA'   # blue  — Type 1
C2 = '#EE6677'   # coral — Type 2
C3 = '#228833'   # green — Type 3
C_GRAY   = '#999999'
C_LIGHT  = '#DDDDDD'
C_H      = '#AA3377'   # magenta — H metric
C_MAXSIM = '#4477AA'   # blue — max_sim metric
C_NORM   = '#66CCEE'   # cyan — norm metric

CMAP = {'type1': C1, 'type2': C2, 'type3': C3}
COND_LABELS = {
    'type1': 'Type 1\n(center-drift)',
    'type2': 'Type 2\n(wrong-well)',
    'type3': 'Type 3\n(coverage gap)',
}
COND_SHORT = {'type1': 'T1', 'type2': 'T2', 'type3': 'T3'}

PAIR_LABELS = {
    'type2_vs_type3': 'T2 – T3',
    'type1_vs_type2': 'T1 – T2',
    'type1_vs_type3': 'T1 – T3',
}

# ACL column width ≈ 3.25 in; full width ≈ 6.75 in
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_all():
    with open(WHITENED_AGG) as f:
        whitened = json.load(f)
    with open(RAW_RESULTS) as f:
        raw = json.load(f)
    with open(SPECTRAL_AGG) as f:
        spectral = json.load(f)
    return whitened, raw, spectral


def extract_prompt_means(raw, metric='max_sim'):
    """Compute per-prompt means from representative-seed raw data."""
    out = {'type1': [], 'type2': [], 'type3': []}
    for seq in raw:
        vals = [m[metric] for m in seq['measurements']]
        if vals:
            out[seq['type']].append(np.mean(vals))
    return out


def extract_seed_r(runs, metric, pair):
    """Extract rank-biserial r across seeds."""
    return [r['two_level_stats'][metric]['pairwise'][pair]
            ['prompt_mean_mw']['r'] for r in runs]


def extract_seed_p(runs, metric, pair):
    """Extract prompt-level MW p across seeds."""
    return [r['two_level_stats'][metric]['pairwise'][pair]
            ['prompt_mean_mw']['p'] for r in runs]


def extract_seed_condition_means(runs, metric):
    """Extract per-seed condition means."""
    out = {'type1': [], 'type2': [], 'type3': []}
    for r in runs:
        for cond in out:
            out[cond].append(
                r['two_level_stats'][metric]
                ['condition_stats'][cond]['prompt_mean'])
    return out


def extract_seed_holm(runs, metric, pair):
    """Count Holm survivals across seeds."""
    survived = 0
    for r in runs:
        for entry in r['two_level_stats'][metric]['holm']:
            if pair in entry[0] and entry[3]:
                survived += 1
    return survived


# ══════════════════════════════════════════════════════════════
# FIGURE 1 (MAIN): max_sim Condition Ordering
# ══════════════════════════════════════════════════════════════

def figure1_maxsim_ordering(whitened, raw):
    """
    Left:  Per-prompt max_sim distributions (representative seed).
           Box + jittered strip, showing T2 > T1 > T3.
    Right: 20-seed rank-biserial r for each pair, with direction
           counts annotated.
    """
    runs = whitened['whitened']['runs']
    K = len(runs)

    # ── Left panel data: per-prompt means from representative seed ──
    prompt_ms = extract_prompt_means(raw, 'max_sim')
    # Order: T2, T1, T3 (descending max_sim to show the ordering)
    order = ['type2', 'type1', 'type3']

    # ── Right panel data: per-seed r values ──
    pairs_show = ['type2_vs_type3', 'type1_vs_type2', 'type1_vs_type3']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8),
                                    gridspec_kw={'width_ratios': [1.1, 1]})

    # ── LEFT: Box + strip ──
    positions = [0, 1, 2]
    bp_data = [prompt_ms[c] for c in order]
    bp = ax1.boxplot(bp_data, positions=positions, widths=0.5,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', linewidth=1.2),
                     whiskerprops=dict(color='#555555'),
                     capprops=dict(color='#555555'))
    for patch, c in zip(bp['boxes'], order):
        patch.set_facecolor(CMAP[c])
        patch.set_alpha(0.35)
        patch.set_edgecolor(CMAP[c])

    # Jittered strip
    rng = np.random.RandomState(42)
    for i, c in enumerate(order):
        vals = np.array(prompt_ms[c])
        jitter = rng.normal(0, 0.08, len(vals))
        ax1.scatter(positions[i] + jitter, vals, s=12, alpha=0.6,
                    color=CMAP[c], edgecolors='none', zorder=3)

    # Grand means as horizontal dashes
    for i, c in enumerate(order):
        gm = np.mean(prompt_ms[c])
        ax1.plot([positions[i] - 0.15, positions[i] + 0.15],
                 [gm, gm], color='black', linewidth=1.5, zorder=4)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([COND_LABELS[c] for c in order], fontsize=7)
    ax1.set_ylabel('Whitened max_sim\n(prompt mean)')
    ax1.set_title('(a)  Cluster commitment by condition', fontsize=9,
                  fontweight='bold', loc='left')

    # Annotate grand means
    for i, c in enumerate(order):
        gm = np.mean(prompt_ms[c])
        ax1.annotate(f'{gm:.4f}', xy=(positions[i] + 0.22, gm),
                     fontsize=6.5, color=CMAP[c], va='center',
                     fontweight='bold')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── RIGHT: 20-seed r stability ──
    y_positions = [2, 1, 0]
    pair_colors = [C3, C1, C_GRAY]  # T2-T3 green, T1-T2 blue, T1-T3 gray

    for yi, pair in zip(y_positions, pairs_show):
        rs = extract_seed_r(runs, 'max_sim', pair)
        rs = np.array(rs)
        col = pair_colors[y_positions.index(yi)]

        # Strip
        jitter = rng.normal(0, 0.06, len(rs))
        ax2.scatter(rs, yi + jitter, s=14, alpha=0.5, color=col,
                    edgecolors='none', zorder=3)

        # Median + IQR bar
        med = np.median(rs)
        q25, q75 = np.percentile(rs, [25, 75])
        ax2.plot([q25, q75], [yi, yi], color=col, linewidth=2.5,
                 solid_capstyle='round', zorder=4)
        ax2.plot(med, yi, 'o', color='white', markersize=5, zorder=5)
        ax2.plot(med, yi, 'o', color=col, markersize=3.5, zorder=6)

        # Direction and Holm annotation
        n_pos = sum(1 for r in rs if r > 0)
        n_neg = K - n_pos
        holm_n = extract_seed_holm(runs, 'max_sim', pair)
        if abs(med) > 0.01:
            sign_label = f'{n_neg}/20 –' if med < 0 else f'{n_pos}/20 +'
        else:
            sign_label = f'{n_pos}/20 +, {n_neg}/20 –'
        ann = f'dir: {sign_label}  |  Holm: {holm_n}/20'
        ax2.annotate(ann, xy=(0.98, yi + 0.25),
                     xycoords=('axes fraction', 'data'),
                     fontsize=6, ha='right', color='#444444')

    ax2.axvline(0, color=C_GRAY, linewidth=0.6, linestyle='--', zorder=1)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([PAIR_LABELS[p] for p in pairs_show], fontsize=7.5)
    ax2.set_xlabel('Rank-biserial $r$  (prompt-level)')
    ax2.set_title('(b)  Effect stability across 20 seeds', fontsize=9,
                  fontweight='bold', loc='left')
    ax2.set_ylim(-0.6, 2.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURE 2 (MAIN): H Collapse vs max_sim Emergence
# ══════════════════════════════════════════════════════════════

def figure2_h_collapse(whitened):
    """
    Left:  20-seed prompt-level p-values for H (all three pairs).
    Right: 20-seed prompt-level p-values for max_sim (all three pairs).
    Shows H scattered high (collapsed) vs max_sim T2-T3 concentrated low.
    """
    runs = whitened['whitened']['runs']
    K = len(runs)

    pairs = ['type2_vs_type3', 'type1_vs_type3', 'type1_vs_type2']
    pair_labels = [PAIR_LABELS[p] for p in pairs]
    pair_colors_map = {
        'type2_vs_type3': C3,
        'type1_vs_type3': C_GRAY,
        'type1_vs_type2': C1,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.6), sharey=True)
    rng = np.random.RandomState(42)

    for ax, metric, title, metric_color in [
        (ax1, 'H', '(a)  Whitened $H$  (collapsed at $N$=30)', C_H),
        (ax2, 'max_sim', '(b)  Whitened max_sim  (emerged at $N$=30)', C_MAXSIM),
    ]:
        for yi, pair in enumerate(pairs):
            ps = np.array(extract_seed_p(runs, metric, pair))
            col = pair_colors_map[pair]
            jitter = rng.normal(0, 0.06, len(ps))

            ax.scatter(ps, yi + jitter, s=14, alpha=0.55, color=col,
                       edgecolors='none', zorder=3)

            # Median tick
            med = np.median(ps)
            ax.plot(med, yi, '|', color='black', markersize=10,
                    markeredgewidth=1.5, zorder=5)

            # Sig count annotation
            n_sig = sum(1 for p in ps if p < 0.05)
            holm_n = extract_seed_holm(runs, metric, pair)
            ax.annotate(f'{n_sig}/20 sig, {holm_n}/20 Holm',
                        xy=(0.97, yi + 0.3),
                        xycoords=('axes fraction', 'data'),
                        fontsize=6, ha='right', color='#444444')

        # α = 0.05 line
        ax.axvline(0.05, color='#CC0000', linewidth=0.8, linestyle='--',
                   zorder=1, alpha=0.7)
        ax.text(0.06, len(pairs) - 0.3, '$\\alpha$=0.05', fontsize=6,
                color='#CC0000', alpha=0.7)

        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pair_labels, fontsize=7.5)
        ax.set_xlabel('Prompt-level MW $p$-value')
        ax.set_title(title, fontsize=8.5, fontweight='bold', loc='left')
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.6, len(pairs) - 0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURE 3 (APPENDIX): Spectral Band Heatmap
# ══════════════════════════════════════════════════════════════

def figure3_spectral_heatmap(spectral):
    """
    Heatmap: bands (rows) × metric-pair combos (columns).
    Cell colour = prompt-level sig rate across 20 seeds.
    Annotated with median r and Holm rate where sig rate ≥ 15%.
    """
    bands_data = spectral['spectral']['aggregate']['bands']
    runs = spectral['spectral']['runs']
    K = len(runs)

    metrics = ['H', 'max_sim', 'norm']
    pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
    cols = [(m, p) for m in metrics for p in pairs]

    band_labels = []
    for b in bands_data:
        meta = b['meta']
        label = meta['band_label']
        var = meta['variance_in_band']
        var_str = f"{var:.1%}" if var >= 0.001 else "<0.1%"
        band_labels.append(f"{label}  ({var_str})")

    n_bands = len(bands_data)
    n_cols = len(cols)

    # Build matrices
    sig_matrix = np.zeros((n_bands, n_cols))
    r_matrix = np.zeros((n_bands, n_cols))
    holm_matrix = np.zeros((n_bands, n_cols))

    for bi, b in enumerate(bands_data):
        agg = b['aggregate']
        for ci, (m, p) in enumerate(cols):
            pa = agg[m][p]
            sig_matrix[bi, ci] = pa['prompt_mw_p'].get('prop_sig', 0)
            r_matrix[bi, ci] = pa['r']['median']
            holm_count = 0
            for run in runs:
                tls = run['bands'][bi]['two_level_stats']
                if m in tls and 'holm' in tls[m]:
                    for entry in tls[m]['holm']:
                        if p in entry[0] and entry[3]:
                            holm_count += 1
            holm_matrix[bi, ci] = holm_count / K

    fig, ax = plt.subplots(figsize=(6.75, 3.6))

    im = ax.imshow(sig_matrix, cmap='YlOrRd', vmin=0, vmax=1,
                   aspect='auto', interpolation='nearest')

    # Annotate cells
    for bi in range(n_bands):
        for ci in range(n_cols):
            sig = sig_matrix[bi, ci]
            r = r_matrix[bi, ci]
            holm = holm_matrix[bi, ci]

            if sig >= 0.15:
                text_color = 'white' if sig >= 0.55 else 'black'
                holm_pct = int(round(holm * 100))
                label = f'{r:+.2f}\n{holm_pct}%'
                ax.text(ci, bi, label, ha='center', va='center',
                        fontsize=5, color=text_color, fontweight='bold')

    # Column labels
    col_labels = [PAIR_LABELS[p] for m, p in cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=0)
    ax.tick_params(axis='x', which='both', length=0, pad=2)

    ax.set_yticks(range(n_bands))
    ax.set_yticklabels(band_labels, fontsize=7)

    # Metric group separators
    for sep in [3, 6]:
        ax.axvline(sep - 0.5, color='white', linewidth=2)

    # Metric group labels below x tick labels
    for mi, m_name in enumerate(metrics):
        mid = mi * 3 + 1
        ax.annotate(m_name, xy=(mid, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -28), textcoords='offset points',
                    ha='center', va='top',
                    fontsize=8.5, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label('Prompt-level sig rate', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title('Spectral band decomposition ($N$=15/group, 20 seeds)\n'
                 'Cell: median $r$ / Holm survival rate',
                 fontsize=9, fontweight='bold', pad=12)

    fig.tight_layout(rect=[0, 0.06, 0.95, 1])
    return fig


# ══════════════════════════════════════════════════════════════
# FIGURE 4 (APPENDIX): Token–Prompt Discordance
# ══════════════════════════════════════════════════════════════

def figure4_discordance(whitened, spectral):
    """
    Scatter: -log10(token p) vs -log10(prompt p) for every
    metric × pair combination from whitened N=30 and spectral N=15.
    Points in lower-right = pseudoreplication artifacts.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    alpha = 0.05
    sig_line = -np.log10(alpha)

    def _collect_points(runs, metric, source_label, marker, alpha_val):
        """Collect token/prompt p-value pairs across seeds (use median)."""
        pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
        points = []
        for pair in pairs:
            tok_ps = []
            prompt_ps = []
            for r in runs:
                pw = r['two_level_stats'][metric]['pairwise'][pair]
                tok_ps.append(pw['token']['p'])
                prompt_ps.append(pw['prompt_mean_mw']['p'])
            # Use median across seeds
            med_tok = np.median(tok_ps)
            med_prompt = np.median(prompt_ps)
            points.append((med_tok, med_prompt, pair, metric))
        return points

    all_points = []

    # Whitened experiment (N=30)
    wht_runs = whitened['whitened']['runs']
    for metric in ['H', 'max_sim', 'norm']:
        pts = _collect_points(wht_runs, metric, 'whitened', 'o', 0.8)
        for tok, prompt, pair, m in pts:
            all_points.append(('whitened', m, pair, tok, prompt))

    # Raw norm from whitened runs
    pts = _collect_points(wht_runs, 'raw_norm', 'raw_norm', 's', 0.8)
    for tok, prompt, pair, m in pts:
        all_points.append(('raw_norm', m, pair, tok, prompt))

    # Spectral bands (N=15)
    spec_runs = spectral['spectral']['runs']
    spec_bands = spectral['spectral']['aggregate']['bands']
    for bi, band in enumerate(spec_bands):
        band_label = band['meta']['band_label']
        for metric in ['H', 'max_sim', 'norm']:
            pairs = ['type1_vs_type2', 'type1_vs_type3', 'type2_vs_type3']
            for pair in pairs:
                tok_ps = []
                prompt_ps = []
                for r in spec_runs:
                    pw = r['bands'][bi]['two_level_stats'][metric]['pairwise'][pair]
                    tok_ps.append(pw['token']['p'])
                    prompt_ps.append(pw['prompt_mean_mw']['p'])
                med_tok = np.median(tok_ps)
                med_prompt = np.median(prompt_ps)
                all_points.append(('spectral', metric, pair, med_tok, med_prompt))

    # Separate by pair type for colouring
    pair_colors = {
        'type1_vs_type2': C1,
        'type1_vs_type3': C_GRAY,
        'type2_vs_type3': C3,
    }
    source_markers = {'whitened': 'o', 'raw_norm': 's', 'spectral': 'D'}
    source_sizes = {'whitened': 30, 'raw_norm': 30, 'spectral': 12}
    source_alpha = {'whitened': 0.75, 'raw_norm': 0.75, 'spectral': 0.35}

    for source, m, pair, tok_p, prompt_p in all_points:
        x = -np.log10(max(tok_p, 1e-20))
        y = -np.log10(max(prompt_p, 1e-20))
        ax.scatter(x, y,
                   c=pair_colors[pair],
                   marker=source_markers[source],
                   s=source_sizes[source],
                   alpha=source_alpha[source],
                   edgecolors='none', zorder=3)

    # Quadrant lines
    ax.axhline(sig_line, color='#CC0000', linewidth=0.7, linestyle='--',
               alpha=0.5, zorder=1)
    ax.axvline(sig_line, color='#CC0000', linewidth=0.7, linestyle='--',
               alpha=0.5, zorder=1)

    # Quadrant labels
    xmax = ax.get_xlim()[1]
    ymax = ax.get_ylim()[1]
    ax.text(0.5, ymax * 0.92, 'prompt only', fontsize=6,
            color='#888888', ha='center', style='italic')
    ax.text(xmax * 0.65, 0.4, 'token only\n(pseudoreplication)',
            fontsize=6, color='#CC0000', ha='center', style='italic',
            alpha=0.7)
    ax.text(xmax * 0.65, ymax * 0.92, 'both levels', fontsize=6,
            color='#228833', ha='center', style='italic')

    ax.set_xlabel('Token-level  $-\\log_{10}\\,p$')
    ax.set_ylabel('Prompt-level  $-\\log_{10}\\,p$')
    ax.set_title('Token–prompt discordance\n(median across 20 seeds)',
                 fontsize=9, fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C1,
               markersize=5, label='T1 – T2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GRAY,
               markersize=5, label='T1 – T3'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C3,
               markersize=5, label='T2 – T3'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
               markersize=5, label='Whitened ($N$=30)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555',
               markersize=5, label='Raw norm ($N$=30)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#555555',
               markersize=4, label='Spectral ($N$=15)'),
    ]
    ax.legend(handles=legend_elements, loc='center left', fontsize=6,
              framealpha=0.85, edgecolor='#CCCCCC')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    whitened, raw, spectral = load_all()
    runs = whitened['whitened']['runs']
    K = len(runs)
    N = whitened.get('n_per_group', 30)
    print(f"  Whitened: K={K} seeds, N={N}/group")
    print(f"  Spectral: K={len(spectral['spectral']['runs'])} seeds")
    print(f"  Raw: {len(raw)} prompts (representative seed)")

    # ── Figure 1: max_sim ordering (MAIN) ──
    print("\nGenerating Figure 1: max_sim condition ordering...")
    fig1 = figure1_maxsim_ordering(whitened, raw)
    path1 = os.path.join(OUTPUT_DIR, 'fig_maxsim_ordering.pdf')
    fig1.savefig(path1)
    fig1.savefig(path1.replace('.pdf', '.png'))
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ── Figure 2: H collapse (MAIN) ──
    print("Generating Figure 2: H collapse vs max_sim emergence...")
    fig2 = figure2_h_collapse(whitened)
    path2 = os.path.join(OUTPUT_DIR, 'fig_h_collapse.pdf')
    fig2.savefig(path2)
    fig2.savefig(path2.replace('.pdf', '.png'))
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Figure 3: spectral heatmap (APPENDIX) ──
    print("Generating Figure 3: spectral band heatmap...")
    fig3 = figure3_spectral_heatmap(spectral)
    path3 = os.path.join(OUTPUT_DIR, 'fig_spectral_heatmap.pdf')
    fig3.savefig(path3)
    fig3.savefig(path3.replace('.pdf', '.png'))
    plt.close(fig3)
    print(f"  Saved: {path3}")

    # ── Figure 4: discordance (APPENDIX) ──
    print("Generating Figure 4: token-prompt discordance...")
    fig4 = figure4_discordance(whitened, spectral)
    path4 = os.path.join(OUTPUT_DIR, 'fig_discordance.pdf')
    fig4.savefig(path4)
    fig4.savefig(path4.replace('.pdf', '.png'))
    plt.close(fig4)
    print(f"  Saved: {path4}")

    print(f"\nAll figures saved to {os.path.abspath(OUTPUT_DIR)}/")
    print("  Main paper:  fig_maxsim_ordering.pdf, fig_h_collapse.pdf")
    print("  Appendix:    fig_spectral_heatmap.pdf, fig_discordance.pdf")


if __name__ == '__main__':
    main()
