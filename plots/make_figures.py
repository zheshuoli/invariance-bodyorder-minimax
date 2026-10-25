#!/usr/bin/env python3
"""Regenerate every figure and table of the paper from the JSON files in results/.

Usage (from the repository root):
    python plots/make_figures.py                 # all figures -> figures/
    python plots/make_figures.py --results DIR --out DIR

Main text:   fig2_exp1.pdf  fig3_exp23.pdf  fig4_exp4.pdf  fig5_exp5.pdf  table2_multipliers.txt
Appendix:    figA_exp1_s3.pdf  figA_exp5_s3_n6.pdf  figA_exp5_robust_n6.pdf  figA_exp5_robust_n10.pdf
             figA_k3_study.pdf  figA_nn_bridge.pdf  figA_nn_control.pdf  tableA_robustness.txt
The script also prints the numbers quoted in the text (slopes, risk ratios, selection
frequencies, multipliers) so they can be checked against the paper.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.ticker import NullFormatter, LogLocator

BASE_RC = {'font.family': 'serif', 'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'ps.fonttype': 42,
           'savefig.bbox': 'tight', 'savefig.dpi': 300}
COLORS = {'F_Kinv': '#D55E00', 'F_K': '#009E73', 'F_inv': '#0072B2', 'F_0': '#E69F00'}
MARKERS = {'F_Kinv': 'o', 'F_K': 's', 'F_inv': 'D', 'F_0': '^'}
LABELS = {'F_Kinv': r'$\mathcal{F}_{K,\mathrm{inv}}$', 'F_K': r'$\mathcal{F}_K$',
          'F_inv': r'$\mathcal{F}_{\mathrm{inv}}$', 'F_0': r'$\mathcal{F}_0$'}
LABELS_LONG = {'F_Kinv': r'$\mathcal{F}_{K,\mathrm{inv}}$ (both priors)', 'F_K': r'$\mathcal{F}_K$ (body-order only)',
               'F_inv': r'$\mathcal{F}_{\mathrm{inv}}$ (inv. only)', 'F_0': r'$\mathcal{F}_0$ (no structure)'}
CLASSES = ['F_Kinv', 'F_K', 'F_inv', 'F_0']


def load(results, name):
    with open(os.path.join(results, name)) as f:
        return json.load(f)


def rc(**kw):
    d = dict(BASE_RC); d.update(kw); plt.rcParams.update(d)


# ----------------------------------------------------------------------------- helpers
def loglog_interp_N(N, risk, target):
    """Smallest N (log-log interpolated) at which the curve `risk` drops to `target`; None if never."""
    N = np.asarray(N, float); r = np.asarray(risk, float)
    for i in range(len(N) - 1):
        if r[i] >= target > r[i + 1]:
            t = (np.log(target) - np.log(r[i])) / (np.log(r[i + 1]) - np.log(r[i]))
            return float(np.exp(np.log(N[i]) + t * (np.log(N[i + 1]) - np.log(N[i]))))
    if r[0] < target:
        return float(N[0])
    return None


def multipliers(d, anchors=(200, 500, 1000)):
    """Sample-efficiency multipliers: samples needed by class C to match F_Kinv at anchor N, divided by N."""
    a = d['analysis']; N = d['config']['N_values']
    out = {}
    for N0 in anchors:
        i = N.index(N0); target = a['F_Kinv']['risk_means'][i]
        row = {}
        for c in ['F_K', 'F_inv', 'F_0']:
            Nc = loglog_interp_N(N, a[c]['risk_means'], target)
            row[c] = None if Nc is None else Nc / N0
        out[N0] = row
    return out


def fmt_mult(x):
    return '---' if x is None else f'{x:.1f}x'


# ----------------------------------------------------------------------------- Figure 2 (Experiment 1)
def fig_exp1(results, out, fname='exp1_data.json', outname='fig2_exp1.pdf', title_suffix=''):
    rc(**{'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 15, 'xtick.labelsize': 12,
          'ytick.labelsize': 12, 'legend.fontsize': 11.5})
    d = load(results, fname); cfg, th, an = d['config'], d['theory'], d['analysis']
    N = np.array(cfg['N_values'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={'wspace': 0.35})
    for c in CLASSES:
        m, s = np.array(an[c]['means']), np.array(an[c]['stds'])
        ax1.errorbar(N, m, yerr=s, fmt=MARKERS[c] + '-', color=COLORS[c], markersize=6, linewidth=1.5,
                     capsize=2.5, label=LABELS[c], zorder=3, markeredgecolor='white', markeredgewidth=0.6)
        mid = len(N) // 2
        Nl = np.logspace(np.log10(N[0] * 0.7), np.log10(N[-1] * 1.3), 50)
        ax1.plot(Nl, m[mid] * (Nl / N[mid]) ** th[c]['slope'], '--', color=COLORS[c], alpha=0.35, linewidth=1.0)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'Sample size $N$'); ax1.set_ylabel(r'$L^2$ risk')
    ax1.set_title('(a) Learning curves' + title_suffix, fontweight='bold')
    h, l = ax1.get_legend_handles_labels()
    h.append(mpl.lines.Line2D([], [], linestyle='--', color='0.45', linewidth=1.0)); l.append('Theory slope')
    ax1.legend(h, l, loc='lower left', framealpha=0.95, edgecolor='0.8', handlelength=1.4, borderpad=0.35,
               fontsize=9.5, handletextpad=0.4)
    ax1.grid(True, which='major', alpha=0.12, linewidth=0.5)
    ax1.set_xlim(350, 80000); ax1.set_ylim(8e-4, 1.2e-1)
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    x = np.arange(4); bw = 0.30
    tv = [abs(th[c]['slope']) for c in CLASSES]; ev = [abs(an[c]['slope_full']) for c in CLASSES]
    ax2.bar(x - bw / 2, tv, bw, color='#B8B8B8', edgecolor='0.5', linewidth=0.6, zorder=2)
    for i, c in enumerate(CLASSES):
        ax2.bar(x[i] + bw / 2, ev[i], bw, color='#E8573D', edgecolor='0.35', linewidth=0.6,
                hatch='///' if c == 'F_0' else None, zorder=2)
    ax2.annotate('†', xy=(x[-1] + bw / 2, ev[-1] + 0.012), ha='center', fontsize=13, fontweight='bold', color='0.35')
    ax2.legend(handles=[mpatches.Patch(facecolor='#B8B8B8', edgecolor='0.5', linewidth=0.6, label='Theory'),
                        mpatches.Patch(facecolor='#E8573D', edgecolor='0.35', linewidth=0.6, label='Empirical'),
                        mpatches.Patch(facecolor='#E8573D', edgecolor='0.35', linewidth=0.6, hatch='///',
                                       label='† pre-asymptotic')],
               loc='upper right', framealpha=0.95, edgecolor='0.8', fontsize=9.5)
    ax2.set_xticks(x); ax2.set_xticklabels([LABELS[c] for c in CLASSES])
    ax2.set_ylabel(r'$|\mathrm{slope}|$'); ax2.set_title('(b) Rate exponents' + title_suffix, fontweight='bold')
    ax2.set_ylim(0, 0.85); ax2.grid(True, axis='y', alpha=0.12, linewidth=0.5)
    fig.savefig(os.path.join(out, outname)); plt.close(fig)
    print(f'[{outname}] empirical vs theory slopes:')
    for c in CLASSES:
        print(f'   {c:7s} empirical {an[c]["slope_full"]:+.3f}  theory {th[c]["slope"]:+.3f}  '
              f'|diff| {abs(abs(an[c]["slope_full"]) - abs(th[c]["slope"])):.3f}')


# ----------------------------------------------------------------------------- Figure 3 (Experiments 2 + 3)
def fig_exp23(results, out):
    rc(**{'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 10})
    d2 = load(results, 'exp2_data.json'); a2 = d2['analysis']; ns = d2['config']['n_values']
    d3 = load(results, 'exp3_data.json'); a3 = d3['analysis']; N3 = np.array(d3['config']['N_values'])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), gridspec_kw={'wspace': 0.42})
    ax = axes[0]
    for c in CLASSES:
        emp = [abs(a2[c][str(n)]['slope_emp']) for n in ns]; th = [abs(a2[c][str(n)]['slope_theory']) for n in ns]
        ax.plot(ns, emp, MARKERS[c] + '-', color=COLORS[c], ms=5, lw=1.4, label=LABELS[c])
        ax.plot(ns, th, '--', color=COLORS[c], alpha=0.45, lw=1.1)
    ax.set_xlabel(r'Number of factors $n$'); ax.set_ylabel(r'$|\mathrm{slope}|$'); ax.set_ylim(0, 0.72)
    ax.set_title(r'(a) Rate exponent vs $n$'); ax.grid(True, alpha=0.15)
    ax = axes[1]
    gaps = {}
    for c in CLASSES:
        g = [abs(a2[c][str(n)]['slope_emp']) - abs(a2[c][str(n)]['slope_theory']) for n in ns]; gaps[c] = g
        ax.plot(ns, g, MARKERS[c] + '-', color=COLORS[c], ms=5, lw=1.4)
    ax.axhline(0, color='0.4', ls=':', lw=0.9)
    ax.set_xlabel(r'Number of factors $n$'); ax.set_ylabel(r'$|\hat\alpha| - |\alpha_{\mathrm{theory}}|$')
    ax.set_title('(b) Slope deviation from theory'); ax.grid(True, alpha=0.15)
    ax = axes[2]
    for c in CLASSES:
        ax.errorbar(N3, a3[c]['risk_means'], yerr=a3[c]['risk_stds'], fmt=MARKERS[c] + '-', color=COLORS[c],
                    ms=4, lw=1.3, capsize=2)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Sample size $N$'); ax.set_ylabel(r'$L^2$ risk')
    ax.set_title(r'(c) Shared $f^* \in \mathcal{F}_{K,\mathrm{inv}}$'); ax.grid(True, alpha=0.15)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax = axes[3]
    sel = [5000, 10000, 20000, 50000]; idx = [list(N3).index(v) for v in sel]
    x = np.arange(len(sel)); bw = 0.27
    for j, c in enumerate(['F_K', 'F_inv', 'F_0']):
        ratios = [a3[c]['risk_means'][i] / a3['F_Kinv']['risk_means'][i] for i in idx]
        ax.bar(x + (j - 1) * bw, ratios, bw, color=COLORS[c], edgecolor='0.3', linewidth=0.5, label=LABELS[c])
    ax.axhline(1, color='0.4', ls='--', lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f'{v // 1000}k' for v in sel]); ax.set_xlabel(r'Sample size $N$')
    ax.set_ylabel(r'Risk ratio vs $\mathcal{F}_{K,\mathrm{inv}}$'); ax.set_title('(d) Risk ratio')
    ax.legend(loc='upper left', framealpha=0.95); ax.grid(True, axis='y', alpha=0.15)
    h, l = axes[0].get_legend_handles_labels()
    h.append(mpl.lines.Line2D([], [], linestyle='--', color='0.45', linewidth=1.0)); l.append('Theory')
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.42, -0.16))
    fig.savefig(os.path.join(out, 'fig3_exp23.pdf')); plt.close(fig)
    print('[fig3_exp23.pdf] Exp 2 slope-magnitude gaps |a_hat|-|a_theory| per n=%s:' % ns)
    for c in CLASSES:
        print(f'   {c:7s} ' + ' '.join(f'{g:+.3f}' for g in gaps[c]))
    print('[fig3_exp23.pdf] Exp 3 risk ratios R(F_c)/R(F_Kinv):')
    for i, n in enumerate(N3):
        print(f'   N={n:6d} ' + ' '.join(f'{c}={a3[c]["risk_means"][i] / a3["F_Kinv"]["risk_means"][i]:.2f}'
                                         for c in ['F_K', 'F_inv', 'F_0']))


# ----------------------------------------------------------------------------- Figure 4 (Experiment 4)
def fig_exp4(results, out):
    rc(**{'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9})
    d = load(results, 'exp4_data.json'); A = d['analysis']; N = np.array(A['oracle']['N_values'])
    mean = {k: np.array(A[k]['risk_means']) for k in ('oracle', 'adaptive', 'over', 'under')}
    se = {k: np.array(A[k]['risk_stds']) / np.sqrt(d['config']['M_rep']) for k in mean}
    emp = np.min(np.vstack([mean['oracle'], mean['over'], mean['under']]), axis=0)
    C_UNDER, C_K2, C_OVER, C_AD = '#D55E00', '#009E73', '#E69F00', '#0072B2'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.45, 3.3), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
    ax1.plot(N, emp, 'k--', lw=1.6, label='Empirical oracle', zorder=5)
    ax1.errorbar(N, mean['under'], yerr=se['under'], color=C_UNDER, ls=':', marker='D', ms=3.5, lw=1.2, capsize=2,
                 label=r"$K'{=}1$ (under)")
    ax1.errorbar(N, mean['oracle'], yerr=se['oracle'], color=C_K2, ls='-', marker='o', ms=3.5, lw=1.2, capsize=2,
                 label=r"$K'{=}K^*{=}2$")
    ax1.errorbar(N, mean['over'], yerr=se['over'], color=C_OVER, ls='-.', marker='^', ms=3.5, lw=1.2, capsize=2,
                 label=r"$K'{=}3$ (over)")
    ax1.errorbar(N, mean['adaptive'], yerr=se['adaptive'], color=C_AD, ls='--', marker='s', ms=3.5, lw=1.2,
                 capsize=2, label=r"Adaptive ($K_{\max}{=}3$)")
    ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.set_ylim(1.3e-3, 1.5e-1)
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_xlabel(r'Sample size $N$'); ax1.set_ylabel(r'$L^2$ risk'); ax1.set_title('(a) Candidate and adaptive risks')
    ax1.grid(True, which='major', alpha=0.4); ax1.grid(True, which='minor', alpha=0.15)
    ax1.legend(loc='upper right', framealpha=0.95)
    Ks = A['K_selected']; labels = [f'{n // 1000}k' if n >= 1000 else str(n) for n in N]
    freq = np.array([[Ks[str(n)].count(k) / len(Ks[str(n)]) for k in (1, 2, 3)] for n in N])
    x = np.arange(len(N)); bottom = np.zeros(len(N))
    for j, (k, c) in enumerate(zip((1, 2, 3), (C_UNDER, C_K2, C_AD))):
        ax2.bar(x, freq[:, j], bottom=bottom, color=c, width=0.7, label=rf'$\hat K{{=}}{k}$', edgecolor='white', lw=0.5)
        bottom += freq[:, j]
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10); ax2.set_ylim(0, 1)
    ax2.set_xlabel(r'Sample size $N$'); ax2.set_ylabel('Selection frequency')
    ax2.set_title(r'(b) Adaptive $\hat K$ distribution'); ax2.legend(loc='lower left', framealpha=0.95)
    fig.savefig(os.path.join(out, 'fig4_exp4.pdf')); plt.close(fig)
    print('[fig4_exp4.pdf] adaptive / empirical-oracle ratio and K_hat frequencies:')
    for i, n in enumerate(N):
        f = freq[i]
        print(f'   N={n:6d} ratio={mean["adaptive"][i] / emp[i]:.3f}  over/K2={mean["over"][i] / mean["oracle"][i]:.2f}'
              f'  K_hat=1:{f[0]:.0%} 2:{f[1]:.0%} 3:{f[2]:.0%}')


# ----------------------------------------------------------------------------- Figure 5 + Table 2 (Experiment 5)
def fig_exp5(results, out):
    rc(**{'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9})
    d6, d10 = load(results, 'exp5_data_n6.json'), load(results, 'exp5_data_n10.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.3), sharey=True, gridspec_kw={'wspace': 0.12})
    for ax, d, title in zip(axes, (d6, d10), ('(a) Maier-Saupe ($n=6$)', '(b) Maier-Saupe ($n=10$)')):
        N = np.array(d['config']['N_values'])
        for c in CLASSES:
            a = d['analysis'][c]
            ax.errorbar(N, a['risk_means'], yerr=a['risk_stds'], color=COLORS[c], marker=MARKERS[c], ms=4, lw=1.3,
                        capsize=2, elinewidth=0.8, label=LABELS[c])
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Sample size $N$'); ax.set_title(title)
        ax.grid(True, which='major', alpha=0.4); ax.grid(True, which='minor', alpha=0.15)
    axes[0].set_ylabel(r'$L^2$ risk'); axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(loc='upper right', framealpha=0.95)
    fig.savefig(os.path.join(out, 'fig5_exp5.pdf')); plt.close(fig)
    lines = ['Table 2: sample-efficiency multipliers (samples needed by class C to match F_Kinv at anchor N) / N', '']
    m6, m10 = multipliers(d6), multipliers(d10)
    lines.append(f'{"Anchor N":>8s} | {"F_K":>6s} {"F_inv":>6s} {"F_0":>6s} (n=6) | {"F_K":>6s} {"F_inv":>6s} {"F_0":>6s} (n=10)')
    for N0 in (200, 500, 1000):
        lines.append(f'{N0:>8d} | ' + ' '.join(f'{fmt_mult(m6[N0][c]):>6s}' for c in ('F_K', 'F_inv', 'F_0')) +
                     '       | ' + ' '.join(f'{fmt_mult(m10[N0][c]):>6s}' for c in ('F_K', 'F_inv', 'F_0')))
    txt = '\n'.join(lines)
    with open(os.path.join(out, 'table2_multipliers.txt'), 'w') as f:
        f.write(txt + '\n')
    print('[fig5_exp5.pdf] ' + txt.replace('\n', '\n   '))


# ----------------------------------------------------------------------------- Appendix: Exp 5 variants
def fig_exp5_variant(results, out, fname, outname, tag):
    rc(**{'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 8})
    d = load(results, fname); N = np.array(d['config']['N_values']); a = d['analysis']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.45, 3.4), gridspec_kw={'wspace': 0.3})
    for c in CLASSES:
        ax1.errorbar(N, a[c]['risk_means'], yerr=a[c]['risk_stds'], color=COLORS[c], marker=MARKERS[c], ms=4,
                     lw=1.3, capsize=2, elinewidth=0.8, label=LABELS_LONG[c])
    ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.set_xlabel(r'Sample size $N$'); ax1.set_ylabel(r'$L^2$ risk')
    ax1.set_title(f'(a) {tag}: risk vs $N$'); ax1.grid(True, alpha=0.2); ax1.legend(loc='lower left', framealpha=0.95)
    ax1.yaxis.set_minor_formatter(NullFormatter())
    for c in ['F_K', 'F_inv', 'F_0']:
        ax2.plot(N, np.array(a[c]['risk_means']) / np.array(a['F_Kinv']['risk_means']), marker=MARKERS[c],
                 color=COLORS[c], ms=4, lw=1.3, label=LABELS_LONG[c])
    ax2.axhline(1, color='0.4', ls='--', lw=0.9); ax2.set_xscale('log'); ax2.set_xlabel(r'Sample size $N$')
    ax2.set_ylabel(r'Risk ratio vs $\mathcal{F}_{K,\mathrm{inv}}$'); ax2.set_title(f'(b) {tag}: risk ratio')
    ax2.grid(True, alpha=0.2); ax2.legend(loc='best', framealpha=0.95)
    fig.savefig(os.path.join(out, outname)); plt.close(fig)
    return multipliers(d)


# ----------------------------------------------------------------------------- Appendix: K=3 study
def fig_k3(results, out):
    rc(**{'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9})
    d = load(results, 'k3_study_data.json'); a = d['analysis']; th = d['theory']; N = np.array(d['config']['N_values'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.45, 3.3), gridspec_kw={'wspace': 0.3})
    for c in ['F_Kinv', 'F_K']:
        m = np.array(a[c]['means']); s = np.array(a[c]['stds'])
        ax1.errorbar(N, m, yerr=s, color=COLORS[c], marker=MARKERS[c], ms=5, lw=1.5, capsize=2, label=LABELS[c])
        slope = th[c]['slope']
        if slope is not None:
            Nl = np.logspace(3, np.log10(N[-1]), 20); i0 = list(N).index(1000)
            ax1.plot(Nl, m[i0] * (Nl / 1000) ** slope, ':', color=COLORS[c], lw=1.3)
    ax1.plot([], [], ':', color='0.4', label='Theory slope')
    ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.set_xlabel(r'Sample size $N$'); ax1.set_ylabel(r'$L^2$ risk')
    ax1.set_title('(a) Learning curves'); ax1.grid(True, alpha=0.2); ax1.legend(loc='lower left', framealpha=0.95)
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ratio = np.array(a['F_K']['means']) / np.array(a['F_Kinv']['means'])
    ax2.plot(N, ratio, 'D-', color='#7B3FBF', ms=5, lw=1.8, label=r'$\mathcal{F}_K/\mathcal{F}_{K,\mathrm{inv}}$')
    ax2.axhline(1, color='0.5', ls=':', lw=1); ax2.text(N[-1], 1.03, 'no advantage', ha='right', fontsize=8, color='0.4', style='italic')
    ax2.annotate(f'{ratio[0]:.2f}' + r'$\times$', (N[0], ratio[0]), textcoords='offset points', xytext=(6, 4), color='#7B3FBF', fontweight='bold', fontsize=9)
    ax2.annotate(f'{ratio[-1]:.2f}' + r'$\times$', (N[-1], ratio[-1]), textcoords='offset points', xytext=(-32, -4), color='#7B3FBF', fontweight='bold', fontsize=9)
    ax2.set_xscale('log'); ax2.set_xlabel(r'Sample size $N$'); ax2.set_ylabel('Risk ratio'); ax2.set_title('(b) Growing advantage')
    ax2.grid(True, alpha=0.2); ax2.legend(loc='upper left', framealpha=0.95)
    fig.savefig(os.path.join(out, 'figA_k3_study.pdf')); plt.close(fig)
    print('[figA_k3_study.pdf] ratio F_K/F_Kinv: ' + ' '.join(f'{n}:{r:.2f}' for n, r in zip(N, ratio)))
    print(f'   F_Kinv fitted |slope| = {abs(a["F_Kinv"]["slope_full"]):.3f} (theory {abs(th["F_Kinv"]["slope"]):.3f})')


# ----------------------------------------------------------------------------- Appendix: neural bridge
NN_KEYS = [('inv_pair', 'F_Kinv', r'$\mathcal{F}_{K,\mathrm{inv}}$-like'), ('pair_only', 'F_K', r'$\mathcal{F}_K$-like'),
           ('inv_only', 'F_inv', r'$\mathcal{F}_{\mathrm{inv}}$-like'), ('no_struct', 'F_0', r'$\mathcal{F}_0$-like')]


def fig_nn(results, out):
    rc(**{'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9})
    d = load(results, 'nn_bridge_combined.json'); R = d['results_by_n']
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.2), sharey=True, gridspec_kw={'wspace': 0.1})
    for ax, n, ttl in zip(axes, ('6', '10'), ('(a) $n=6$', '(b) $n=10$')):
        a = R[n]['analysis']
        for key, c, lab in NN_KEYS:
            ax.errorbar(a[key]['N'], a[key]['means'], yerr=a[key]['stds'], color=COLORS[c], marker=MARKERS[c], ms=4,
                        lw=1.3, capsize=2, label=lab)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Sample size $N$'); ax.set_title(ttl)
        ax.grid(True, which='both', alpha=0.2)
    axes[0].set_ylabel('Test MSE (clean target)'); axes[1].legend(loc='lower left', framealpha=0.95)
    fig.savefig(os.path.join(out, 'figA_nn_bridge.pdf')); plt.close(fig)
    print('[figA_nn_bridge.pdf] slopes / ordering / endpoint ratio:')
    for n in ('6', '10'):
        a = R[n]['analysis']; Ns = a['inv_pair']['N']
        strict = [a['inv_pair']['means'][i] < a['pair_only']['means'][i] < a['inv_only']['means'][i] < a['no_struct']['means'][i]
                  for i in range(len(Ns))]
        first = next((Ns[i] for i in range(len(Ns)) if all(strict[i:])), None)
        print(f'   n={n}: slopes ' + ' '.join(f'{c}:{a[k]["slope"]:+.2f}' for k, c, _ in NN_KEYS) +
              f' | strict ordering from N={first} | F_0/F_Kinv at N={Ns[-1]}: {a["no_struct"]["means"][-1] / a["inv_pair"]["means"][-1]:.0f}x')
    # architectural control
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.4), sharey=True, gridspec_kw={'wspace': 0.1})
    for ax, n, ttl in zip(axes, ('6', '10'), ('(a) $n=6$', '(b) $n=10$')):
        t = load(results, f'nn_control_n{n}.json'); a = R[n]['analysis']
        ax.errorbar(t['analysis']['N'], t['analysis']['means'], yerr=t['analysis']['stds'], color=COLORS['F_Kinv'],
                    marker='D', ms=4, lw=1.5, capsize=2, label=r'Control $\mathcal{F}_{K,\mathrm{inv}}$-like')
        ax.errorbar(a['inv_pair']['N'], a['inv_pair']['means'], yerr=a['inv_pair']['stds'], color='#CC79A7', marker='o',
                    ms=4, lw=1.3, capsize=2, label=r'Bridge $\mathcal{F}_{K,\mathrm{inv}}$-like')
        for key, c, lab in NN_KEYS[2:]:
            ax.errorbar(a[key]['N'], a[key]['means'], yerr=a[key]['stds'], color=COLORS[c], marker=MARKERS[c], ms=4,
                        lw=1.3, capsize=2, label=lab)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Sample size $N$'); ax.set_title(ttl)
        ax.grid(True, which='both', alpha=0.2)
    axes[0].set_ylabel('Test MSE (clean target)')
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(os.path.join(out, 'figA_nn_control.pdf')); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(os.path.dirname(__file__), '..', 'results'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'figures'))
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    fig_exp1(args.results, args.out)
    fig_exp23(args.results, args.out)
    fig_exp4(args.results, args.out)
    fig_exp5(args.results, args.out)
    fig_exp1(args.results, args.out, fname='exp1_s3_data.json', outname='figA_exp1_s3.pdf', title_suffix=' ($s=3$)')
    rows = []
    rows.append(('n=6, s=2, lambda=0   (main)', multipliers(load(args.results, 'exp5_data_n6.json'))[500]))
    rows.append(('n=6, s=2, lambda=0.3       ', fig_exp5_variant(args.results, args.out, 'exp5_data_n6_robust.json', 'figA_exp5_robust_n6.pdf', r'$\lambda=0.3$, $n=6$')[500]))
    rows.append(('n=6, s=3, lambda=0         ', fig_exp5_variant(args.results, args.out, 'exp5_s3_data_n6.json', 'figA_exp5_s3_n6.pdf', r'$s=3$, $n=6$')[500]))
    rows.append(('n=10, s=2, lambda=0  (main)', multipliers(load(args.results, 'exp5_data_n10.json'))[500]))
    rows.append(('n=10, s=2, lambda=0.3      ', fig_exp5_variant(args.results, args.out, 'exp5_data_n10_robust.json', 'figA_exp5_robust_n10.pdf', r'$\lambda=0.3$, $n=10$')[500]))
    txt = 'Appendix robustness table: multipliers at anchor N=500\n' + '\n'.join(
        f'{name} | ' + ' '.join(f'{fmt_mult(r[c]):>6s}' for c in ('F_K', 'F_inv', 'F_0')) for name, r in rows)
    with open(os.path.join(args.out, 'tableA_robustness.txt'), 'w') as f:
        f.write(txt + '\n')
    print('[tableA_robustness.txt] ' + txt.replace('\n', '\n   '))
    fig_k3(args.results, args.out)
    fig_nn(args.results, args.out)
    print('done ->', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
