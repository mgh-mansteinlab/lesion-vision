#!/usr/bin/env python3
"""Export every manuscript table and plotted figure data as CSV (source data).

Tables are parsed from the generated LaTeX (manuscript/generated/*.tex) and from
the tables typed in the manuscript, so the CSVs match the PDF exactly.
Output: manuscript/source_data/.
"""
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / 'manuscript'; OUT = M / 'source_data'; P = ROOT / 'predictions'


def expand_multicolumn(cell):
    m = re.match(r'\s*\\multicolumn\{(\d+)\}\{.*?\}\{(.*)\}\s*$', cell, re.S)
    return [m.group(2)] + [''] * (int(m.group(1)) - 1) if m else [cell]


def clean(c):
    c = re.sub(r'\\(?:emph|textbf|textit)\{([^}]*)\}', r'\1', c)
    c = re.sub(r'\\cite\{[^}]*\}', '', c)
    for a, b in [(r'$\pm$', '±'), (r'$\mu$m', 'µm'), (r'\mu', 'µ'), ('$', ''), ('--', '–'), (r'\%', '%'), (r'\_', '_'),
                 (r'\textasciitilde', '~'), (r'\rho', 'ρ'), (r'\times', '×'), (r'\ ', ' '), ('~', ' '), (r'\&', '&'), ('{', ''), ('}', ''), ('mm^2', 'mm²'), ('^2', '²'), (r'\le', '≤'), (r'\ge', '≥'), (r'\kappa', 'κ'), ('CO_2', 'CO2')]:
        c = c.replace(a, b)
    c = re.sub(r'\\multicolumn\d+[lcr@]*', '', c)
    c = re.sub(r'\\[a-zA-Z]+', '', c)
    return ' '.join(c.split()).strip()


def tabular_to_df(tex):
    body = tex[tex.index(r'\toprule') + len(r'\toprule'): tex.rindex(r'\bottomrule')]
    rows = []
    for line in body.split(r'\\'):
        line = re.sub(r'\\(midrule|addlinespace(\[[^\]]*\])?|cmidrule(\([a-z]*\))?\{[^}]*\})', '', line).strip()
        if not line:
            continue
        rows.append([clean(y) for x in re.split(r'(?<!\\)&', line) for y in expand_multicolumn(x)])
    w = max(len(r) for r in rows)
    rows = [r + [''] * (w - len(r)) for r in rows]
    head, rows = rows[0], rows[1:]
    if rows and head[0] == '' and rows[0][0]:  # two-row header (spanning group + sub-columns)
        grp, cur = [], ''
        for h in head:
            cur = h or cur; grp.append(cur)
        head = [f'{g}: {r}' if g and g != r else r for g, r in zip(grp, rows[0])]; rows = rows[1:]
    df = pd.DataFrame(rows, columns=[c or f'col{i}' for i, c in enumerate(head)])
    keep = [c for c in df.columns if not (c.startswith('col') and df[c].astype(str).str.strip().eq('').all())]
    return df[keep]


def main():
    OUT.mkdir(exist_ok=True)
    g = M / 'generated'
    tables = [('Table1_models', 'table_models'), ('Table2_depth_strata', 'table_depth'), ('Table3_census', 'table_census'),
              ('Table4_geometry', 'table_geometry'), ('Table5_sampling', 'table_sampling'), ('Table6_cv_example_series', 'table_cv'),
              ('Table7_glahn_comparison', 'table_glahn')]
    for name, f in tables:
        tabular_to_df((g / f'{f}.tex').read_text()).to_csv(OUT / f'{name}.csv', index=False)
    tex = (M / 'manu_scientific_reports.tex').read_text()
    for name, label in [('Table8_benchmark', 'tab:benchmark'), ('Table9_comparison_manual_safhir', 'tab:comparison')]:
        i = tex.index(r'\label{' + label + '}'); j = tex.index(r'\end{tabular*}', i)
        tabular_to_df(tex[i:j + 14]).to_csv(OUT / f'{name}.csv', index=False)
    # figure data
    pairs = pd.read_csv(P / 'r3_greedy' / 'stats_filtered_nested_filtered_nested_pairs.csv')
    order = sorted(pairs.section_id.unique(), key=lambda x: tuple(map(int, re.findall(r'\d+', x))))
    pairs.insert(0, 'section_serial_rank', pairs.section_id.map({k: i + 1 for i, k in enumerate(order)}))
    pairs = pairs.drop(columns=['section_id'])
    pairs.to_csv(OUT / 'Fig4_bland_altman_matched_lesion_pairs.csv', index=False)
    prof = []
    for s, lab in [('laser_only_CCC', 'Laser only'), ('papain_CPC', 'Laser + papain'), ('papain_urea_CPVB', 'Laser + papain-urea')]:
        d = pd.read_csv(P / 'r3_example_series' / f'{s}_sections.csv').drop(columns=['section'])
        d.insert(0, 'series', lab); prof.append(d)
    pd.concat(prof).to_csv(OUT / 'Fig5_depth_profiles_per_section.csv', index=False)
    (OUT / 'README.txt').write_text(
        'Source data for the manuscript.\n'
        'Table1-9: values exactly as printed in the tables.\n'
        'Fig4_bland_altman_matched_lesion_pairs.csv: one row per matched lesion (expert gt_*, automated pr_* in um).\n'
        'Fig5_depth_profiles_per_section.csv: per-section means of the three example series (depth_um = serial rank x 20 um).\n'
        'Figures 1, 2, 3 and 6 are images of histology and segmentation masks; they have no tabular source data.\n')
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
