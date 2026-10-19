#!/usr/bin/env python3
"""Final-pipeline areas on the example series versus Glahn et al. (Lasers Surg Med 2025;57:112-120).

Glahn et al. (PMID 39682024): horizontal NBTC serials every 20 um from 0 to 500 um,
10 randomly selected lesions per punch followed through every section, areas outlined manually in NDP.view;
ablation zone (AZ) area = ablated tissue enclosed by the coagulation zone (CZ);
CZ area = denatured tissue minus AZ; absent features recorded as zero.
Published means +- SD (mm^2): control AZ 0.0091 +- 0.004, CZ 0.051 +- 0.024;
papain AZ 0.0089 +- 0.006, CZ 0.044 +- 0.014; papain-urea AZ 0.0103 +- 0.022,
CZ 0.028 +- 0.014.

Here: the same endpoints from the final-pipeline lesion tables of the three
example series (predictions/r3_example_series), interior lesions (not touching
the section edge) of sections at 0-500 um, AZ = 0 when no lumen. Reported over
every lesion and with Glahn's sampling (10 random lesions per section, 2,000
repetitions). Writes predictions/r3_glahn_comparison.json and
manuscript/generated/table_glahn.tex.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / 'predictions' / 'r3_example_series'
SERIES = [('laser_only_CCC', 'Laser only', 'control'), ('papain_CPC', 'Laser + papain', 'papain'),
          ('papain_urea_CPVB', 'Laser + papain--urea', 'papain_urea')]
GLAHN = {'control': (0.0091, 0.004, 0.051, 0.024), 'papain': (0.0089, 0.006, 0.044, 0.014),
         'papain_urea': (0.0103, 0.022, 0.028, 0.014)}  # AZ mean, SD, CZ mean, SD (mm^2)
MAX_DEPTH = 500
rng = np.random.default_rng(42)


def lesions(series):
    sec = pd.read_csv(EX / f'{series}_sections.csv')
    out = []
    for _, r in sec[sec.depth_um <= MAX_DEPTH].iterrows():
        f = EX / series / 'pred' / r.section / 'csv' / 'lesion_summary.csv'
        d = pd.read_csv(f)
        d = d[~d.Touches_Edge.astype(str).str.lower().isin(['true', '1'])].copy()
        d['az_mm2'] = np.where(d.Has_Ablation.astype(str).str.lower().isin(['true', '1']), d.Ablation_Area_um2, 0.0) / 1e6
        d['cz_mm2'] = d.Coagulation_Area_um2 / 1e6
        d['section'] = r.section; d['depth_um'] = r.depth_um
        out.append(d[['section', 'depth_um', 'az_mm2', 'cz_mm2']])
    return pd.concat(out, ignore_index=True)


def main():
    res = {}
    for s, label, key in SERIES:
        d = lesions(s)
        groups = [g for _, g in d.groupby('section') if len(g) >= 10]
        reps = []
        for _ in range(2000):  # Glahn sampling: 10 random lesions per section
            smp = pd.concat([g.sample(10, random_state=int(rng.integers(1 << 31))) for g in groups])
            reps.append((smp.az_mm2.mean(), smp.cz_mm2.mean()))
        reps = np.array(reps)
        res[key] = {'label': label, 'n_lesions': int(len(d)), 'n_sections': int(d.section.nunique()),
                    'az_mean': float(d.az_mm2.mean()), 'az_sd': float(d.az_mm2.std(ddof=1)),
                    'cz_mean': float(d.cz_mm2.mean()), 'cz_sd': float(d.cz_mm2.std(ddof=1)),
                    'sample10_az_mean': float(reps[:, 0].mean()), 'sample10_az_95': [float(x) for x in np.percentile(reps[:, 0], [2.5, 97.5])],
                    'sample10_cz_mean': float(reps[:, 1].mean()), 'sample10_cz_95': [float(x) for x in np.percentile(reps[:, 1], [2.5, 97.5])],
                    'glahn': dict(zip(('az_mean', 'az_sd', 'cz_mean', 'cz_sd'), GLAHN[key]))}
    c, pu, pa = res['control'], res['papain_urea'], res['papain']
    res['cz_change_papain_urea_pct'] = 100 * (pu['cz_mean'] / c['cz_mean'] - 1)
    res['cz_change_papain_pct'] = 100 * (pa['cz_mean'] / c['cz_mean'] - 1)
    res['az_change_papain_urea_pct'] = 100 * (pu['az_mean'] / c['az_mean'] - 1)
    res['glahn_cz_change_papain_urea_pct'] = 100 * (0.028 / 0.051 - 1)
    res['glahn_cz_change_papain_pct'] = 100 * (0.044 / 0.051 - 1)
    res['glahn_az_change_papain_urea_pct'] = 100 * (0.0103 / 0.0091 - 1)

    f = lambda x: f'{x:.4f}'
    L = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
         r' & \multicolumn{2}{c}{Ablation-zone area (mm$^2$)} & \multicolumn{2}{c}{Coagulation-zone area (mm$^2$)} \\',
         r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
         r'Series & Glahn et al. & This work & Glahn et al. & This work \\', r'\midrule']
    for s, label, key in SERIES:
        r = res[key]; g = r['glahn']
        L.append(f"{label} & {f(g['az_mean'])} $\\pm$ {g['az_sd']:.3f} & {f(r['az_mean'])} $\\pm$ {r['az_sd']:.3f} & "
                 f"{g['cz_mean']:.3f} $\\pm$ {g['cz_sd']:.3f} & {r['cz_mean']:.3f} $\\pm$ {r['cz_sd']:.3f} \\\\")
    res['az_change_papain_pct'] = 100 * (pa['az_mean'] / c['az_mean'] - 1); res['glahn_az_change_papain_pct'] = 100 * (0.0089 / 0.0091 - 1)
    L += [r'\addlinespace', r'\multicolumn{5}{@{}l}{\emph{Change relative to laser only}} \\',
          f"Laser + papain & ${res['glahn_az_change_papain_pct']:+.0f}$\\% & ${res['az_change_papain_pct']:+.0f}$\\% & "
          f"${res['glahn_cz_change_papain_pct']:+.0f}$\\% & ${res['cz_change_papain_pct']:+.0f}$\\% \\\\",
          f"Laser + papain--urea & ${res['glahn_az_change_papain_urea_pct']:+.0f}$\\% & ${res['az_change_papain_urea_pct']:+.0f}$\\% & "
          f"${res['glahn_cz_change_papain_urea_pct']:+.0f}$\\% & ${res['cz_change_papain_urea_pct']:+.0f}$\\% \\\\",
          r'\bottomrule', r'\end{tabular}']
    json.dump(res, open(ROOT / 'predictions' / 'r3_glahn_comparison.json', 'w'), indent=1)
    (ROOT / 'manuscript' / 'generated' / 'table_glahn.tex').write_text('\n'.join(L).replace('-', '-') + '\n')
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'glahn'}) for k, v in res.items()}, indent=1))


if __name__ == '__main__':
    main()
