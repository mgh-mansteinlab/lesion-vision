#!/usr/bin/env python3
"""Generate every round-3 manuscript table and in-text number from result files.

Writes manuscript/generated/{numbers.tex, table_models.tex, table_depth.tex,
table_census.tex, table_geometry.tex, table_sampling.tex}. Numbers that do not
exist yet render as a red [pending] marker, so nothing is ever typed by hand.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'predictions'
OUT = ROOT / 'manuscript' / 'generated'
PENDING = r'\textcolor{red}{[pending]}'
# slide/series identifiers are internal; the manuscript uses neutral labels
SERIES_LABEL = {'20H': 'A', 'CPBSV': 'B', 'CUA': 'C', 'JAETH': 'D', 'RH': 'E', 'V': 'F'}
EXAMPLE_SERIES = ('laser_only_CCC', 'papain_CPC', 'papain_urea_CPVB')


def series_of(slide):
    for k in sorted(SERIES_LABEL, key=len, reverse=True):
        if slide.startswith(k):
            return k


def section_label(slide, all_slides):
    k = series_of(slide)
    same = sorted(x for x in all_slides if series_of(x) == k)
    return f'Series {SERIES_LABEL[k]}' + (f', section {same.index(slide) + 1}' if len(same) > 1 else '')

MODELS = [  # id, label, architecture/encoder, params(M), tag (None = M1 files)
    ('M1', 'Pre-specified', 'UNet / ConvNeXt-B', 92.7, None),
    ('M2', '+ lesion-free sections', 'UNet / ConvNeXt-B', 92.7, 'r3m2'),
    ('C1', 'Comparison', 'UPerNet / ConvNeXt-B', 97.7, 'r3_c1_upernet_convnextb'),
    ('C2', 'Comparison', 'SegFormer / MiT-B3', 44.6, 'r3_c2_segformer_mitb3'),
    ('C3', 'Comparison', 'DeepLabV3+ / ResNet-50', 26.7, 'r3_c3_deeplabv3plus_resnet50'),
    ('C4', 'Comparison', 'UNet++ / EfficientNet-B3', 13.6, 'r3_c4_unetpp_effb3'),
]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def f3(x):
    return PENDING if x is None else f'{x:.3f}'


def pct(x):
    return PENDING if x is None else f'{100 * x:.1f}'


def pipe(stats, name):
    return (stats or {}).get('pipelines', {}).get(name)


def summarise(p):
    if not p:
        return None
    a, o = p['pixel']['all'], p['object']['tol_250um']
    return {'macro': a['macro_dice'], 'ci': a['macro_dice_ci95'], 'abl': a['per_class']['ablation']['dice'],
            'coag': a['per_class']['coagulation']['dice'], 'bands': {b: p['pixel'][b] for b in ('superficial', 'middle', 'deep')},
            'rec': o['all']['recall'], 'prec': o['all']['precision'], 'obj': p['object'], 'ba': p['object']['bland_altman']}


def model_rows():
    rows = []
    for mid, role, arch, params, tag in MODELS:
        if tag is None:
            st = load(P / 'r3_stats.json')
            chosen = summarise(pipe(st, 'tta')); default = summarise(pipe(st, 'frozen'))
            filt = summarise(pipe(load(P / 'r3m1' / 'stats_filtered_nested.json'), 'filtered_nested'))
        else:
            sm = load(P / tag / 'summary.json') or {}
            st = load(P / tag / 'stats.json')
            chosen = summarise(pipe(st, sm.get('chosen_config', 'tta_raw'))) if sm.get('status') == 'done' else None
            default = None
            filt = summarise(pipe(load(P / tag / 'stats_filtered_nested.json'), 'filtered_nested'))
        rows.append((mid, role, arch, params, chosen, filt, default))
    ens = summarise(pipe(load(P / 'r3_ensemble' / 'stats_filtered_nested.json'), 'filtered_nested'))
    fin = summarise(pipe(load(P / 'r3_greedy' / 'stats_filtered_nested.json'), 'filtered_nested'))
    sel = load(P / 'r3_greedy' / 'selection.json')
    rows.append(('E5', 'Ensemble, equal weights', 'M2+C1+C2+C3+C4', None, None, ens, None))
    rows.append(('F', 'Final (selected on validation)', '+'.join(sel['selected']) if sel else PENDING, None, None, fin, None))
    return rows, fin, sel


def ci(s):
    return PENDING if not s else f"{s['macro']:.3f} ({s['ci'][0]:.3f}--{s['ci'][1]:.3f})"


def table_models(rows):
    L = [r'\begin{tabular}{@{}lllcccccc@{}}', r'\toprule',
         r'Model & Architecture / encoder & Params & Macro-Dice (95\% CI) & Abl. & Coag. & Deep band & Recall (\%) & Precision (\%) \\',
         r'\midrule', r'\multicolumn{9}{@{}l}{\emph{Inference setting chosen on validation}} \\']
    for mid, role, arch, params, ch, fl, df in rows:
        if mid in ('E5', 'F'):
            continue
        L.append(f"{mid} & {arch} & {params:.1f}M & {ci(ch)} & {f3(ch and ch['abl'])} & {f3(ch and ch['coag'])} & "
                 f"{f3(ch and ch['bands']['deep']['macro_dice'])} & {pct(ch and ch['rec'])} & {pct(ch and ch['prec'])} \\\\")
    L += [r'\addlinespace', r'\multicolumn{9}{@{}l}{\emph{Nested decoding + lesion filter tuned on validation}} \\']
    for mid, role, arch, params, ch, fl, df in rows:
        p = f'{params:.1f}M' if params else '--'
        L.append(f"{mid} & {arch} & {p} & {ci(fl)} & {f3(fl and fl['abl'])} & {f3(fl and fl['coag'])} & "
                 f"{f3(fl and fl['bands']['deep']['macro_dice'])} & {pct(fl and fl['rec'])} & {pct(fl and fl['prec'])} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L)


def table_depth(m1, fin):
    L = [r'\begin{tabular}{@{}llccccc@{}}', r'\toprule',
         r'Pipeline & Depth band & Macro-Dice & Abl.\ Dice & Coag.\ Dice & Coag.\ recall & Coag.\ width MAE \\', r'\midrule']
    for lab, s in (('Pre-specified (M1)', m1), ('Final', fin)):
        for b, rng in (('superficial', '0--220 $\\mu$m'), ('middle', '240--460 $\\mu$m'), ('deep', '480--680 $\\mu$m')):
            if s:
                x = s['bands'][b]; mae = s['obj']['coag_median_abs_err_by_band_um'].get(b)
                L.append(f"{lab if b == 'superficial' else ''} & {rng} & {x['macro_dice']:.3f} & {x['per_class']['ablation']['dice']:.3f} & "
                         f"{x['per_class']['coagulation']['dice']:.3f} & {x['per_class']['coagulation']['recall']:.3f} & "
                         f"{'--' if mae is None else f'{mae:.1f} $\\mu$m'} \\\\")
            else:
                L.append(f"{lab if b == 'superficial' else ''} & {rng} & {PENDING} & & & & \\\\")
        L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L)


def table_census():
    st = load(P / 'r3_greedy' / 'stats_filtered_nested.json')
    g = (st or load(P / 'r3_stats.json'))['census']['gt']
    c = (st or {}).get('census', {}).get('filtered_nested')
    def v(d, k, sub=None):
        if d is None:
            return PENDING
        x = d[k] if sub is None else (d[k] or {}).get(sub)
        return PENDING if x is None else (f'${x:.3f}$' if isinstance(x, float) and abs(x) < 10 else f'{x:.0f}')
    rows = [('Detections per section, superficial', 'Mean count per section', 'detections_per_section_superficial3', None),
            ('Within-section ablation CV, superficial', 'SD/mean of ablation diameter, interior lesions', 'superficial3_within_section_abl_cv', None),
            ('Nearest-neighbor pitch, superficial', 'Median center-to-nearest-center distance ($\\mu$m)', 'superficial3_nn_pitch_um', None),
            ('Coagulation/ablation width ratio', 'Median over lesions, superficial', 'superficial3_coag_over_abl_ratio', 'median'),
            ('Coagulation vs ablation', 'Pearson $r$ across lesions, superficial', 'superficial3_coag_vs_abl_pearson', 'r'),
            ('Ablation CV vs depth', 'Spearman $\\rho$ across sections', 'within_section_abl_cv_vs_depth_spearman', 'rho'),
            ('Lumen fraction vs depth', 'Spearman $\\rho$ across sections', 'lumen_fraction_vs_depth_spearman', 'rho'),
            ('Detections vs depth', 'Spearman $\\rho$ across sections', 'counts_vs_depth_spearman', 'rho')]
    L = [r'\begin{tabular}{@{}llcc@{}}', r'\toprule', r'Census descriptor (evaluation punch) & Metric & Expert annotation & Final pipeline \\', r'\midrule']
    for lab, met, k, sub in rows:
        L.append(f'{lab} & {met} & {v(g, k, sub)} & {v(c, k, sub)} \\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L)


def table_geometry():
    rows = list(csv.DictReader(open(P / 'r3_cpu_report' / 'tables' / 'gt_section_geometry_pitch.csv')))
    by = {}
    for r in rows:
        by.setdefault(r['series'], []).append(r)
    import statistics as S
    L = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
         r'Series & Sections & Section eq.\ diameter (mm) & Lesions per mm$^2$ & Nearest-neighbor pitch ($\mu$m) \\', r'\midrule']
    for s in sorted(by, key=lambda k: (k != 'PVcont', SERIES_LABEL.get(k, k))):
        rr = by[s]; med = lambda c: S.median(float(r[c]) for r in rr if r[c] not in ('', 'nan'))
        if s == 'PVcont':
            L += [f"Evaluation punch & {len(rr)} & {med('eq_diam_mm'):.2f} & {med('lesions_per_mm2'):.2f} & {med('nn_pitch_median_um'):.0f} \\\\", r'\addlinespace']
        else:
            L.append(f"Series {SERIES_LABEL[s]} & {len(rr)} & {med('eq_diam_mm'):.2f} & {med('lesions_per_mm2'):.2f} & {med('nn_pitch_median_um'):.0f} \\\\")
    L += [r'\midrule', r'Nominal treatment grid & & 8 (punch) & 1.96 & 714 \\', r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L)


def table_sampling():
    rows = list(csv.DictReader(open(P / 'r3_manual_sampling' / 'manual_sampling_simulation.csv')))
    L = [r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
         r'Section & Lesions & Full-field CV & Random 10: 95\% range & Within $\pm$20\% & Well-formed 10: CV bias \\', r'\midrule']
    allsec = [r['section'] for r in rows]
    for r in sorted(rows, key=lambda r: section_label(r['section'], allsec)):
        L.append(f"{section_label(r['section'], allsec)} & {r['n_lesions']} & {float(r['full_cv']):.3f} & {float(r['random10_cv_p2.5']):.3f}--{float(r['random10_cv_p97.5']):.3f} & "
                 f"{100 * float(r['random10_cv_within_20pct']):.0f}\\% & ${float(r['wellformed10_cv_bias_pct']):+.0f}$\\% \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L)


def numbers(rows, fin, sel):
    m1 = rows[0][4]; m1d = rows[0][6]
    samp = load(P / 'r3_manual_sampling' / 'summary.json') or {}
    d = {}
    def put(k, x, fmt='{:.3f}'):
        d[k] = PENDING if x is None else fmt.format(x).replace('-', '$-$', 1)
    put('MOneDefaultMacro', m1d and m1d['macro']); put('MOneMacro', m1 and m1['macro'])
    put('MOneCILo', m1 and m1['ci'][0]); put('MOneCIHi', m1 and m1['ci'][1])
    put('FinalMacro', fin and fin['macro']); put('FinalCILo', fin and fin['ci'][0]); put('FinalCIHi', fin and fin['ci'][1])
    for b, k in (('superficial', 'Sup'), ('middle', 'Mid'), ('deep', 'Deep')):
        put(f'Final{k}', fin and fin['bands'][b]['macro_dice'])
    put('FinalAbl', fin and fin['abl']); put('FinalCoag', fin and fin['coag'])
    put('FinalRecall', fin and fin['rec'] * 100, '{:.1f}'); put('FinalPrecision', fin and fin['prec'] * 100, '{:.1f}')
    for cls, k in (('ablation_diameter_um', 'Abl'), ('coagulation_width_um', 'Coag')):
        b = fin and fin['ba'][cls]
        put(f'Final{k}Bias', b and b['bias'], '{:+.1f}'); put(f'Final{k}LoaLo', b and b['loa'][0], '{:+.1f}'); put(f'Final{k}LoaHi', b and b['loa'][1], '{:+.1f}')
        put(f'Final{k}N', b and b['n'], '{:d}')
    d['FinalMembers'] = '+'.join(sel['selected']) if sel else PENDING
    chosen = [r[4]['macro'] for r in rows[:6] if r[4]]
    put('ArchMacroMin', min(chosen) if chosen else None); put('ArchMacroMax', max(chosen) if chosen else None)
    put('SampWithinTwenty', samp.get('median_prob_random10_cv_within_20pct', None) and samp['median_prob_random10_cv_within_20pct'] * 100, '{:.0f}')
    put('SampWellformedBias', samp.get('median_wellformed10_cv_bias_pct'), '{:+.0f}')
    put('SampMeanBias', samp.get('median_wellformed10_mean_bias_pct'), '{:+.0f}')
    # unfiltered / filtered lesion precision and filtered recall across the six models
    pr0 = [r[4]['prec'] * 100 for r in rows[:6] if r[4]]; pr1 = [r[5]['prec'] * 100 for r in rows[:6] if r[5]]
    rc1 = [r[5]['rec'] * 100 for r in rows[:6] if r[5]]
    put('PrecRawMin', min(pr0), '{:.0f}'); put('PrecRawMax', max(pr0), '{:.0f}')
    put('PrecFiltMin', min(pr1), '{:.0f}'); put('PrecFiltMax', max(pr1), '{:.0f}'); put('RecFiltMin', min(rc1), '{:.0f}')
    # census of the evaluation punch (expert annotation and final pipeline)
    st = load(P / 'r3_greedy' / 'stats_filtered_nested.json') or {}
    g = st.get('census', {}).get('gt'); c = st.get('census', {}).get('filtered_nested')
    if g:
        put('CenSupCV', g['superficial3_within_section_abl_cv']); put('CenSupDet', g['detections_per_section_superficial3'], '{:.1f}')
        put('CenSupPitch', g['superficial3_nn_pitch_um'], '{:,.0f}')
        put('CenSupCVLo', g['superficial3_per_section_cv_range'][0], '{:.2f}'); put('CenSupCVHi', g['superficial3_per_section_cv_range'][1], '{:.2f}')
        put('CenDetSup', g['by_band']['superficial']['detections_per_section'], '{:.1f}'); put('CenDetDeep', g['by_band']['deep']['detections_per_section'], '{:.1f}')
        put('CenLumSup', g['by_band']['superficial']['lumen_fraction'], '{:.2f}'); put('CenLumDeep', g['by_band']['deep']['lumen_fraction'], '{:.2f}')
        put('CenCVRho', g['within_section_abl_cv_vs_depth_spearman']['rho'], '{:.2f}')
        put('CenCVSup', g['by_band']['superficial']['within_section_abl_cv_mean'], '{:.2f}'); put('CenCVMid', g['by_band']['middle']['within_section_abl_cv_mean'], '{:.2f}')
    if c:
        put('CenPredSupCV', c['superficial3_within_section_abl_cv']); put('CenPredDetDeep', c['by_band']['deep']['detections_per_section'], '{:.1f}')
    # section geometry of all annotated slides of verified scale
    import statistics as S
    geo = list(csv.DictReader(open(P / 'r3_cpu_report' / 'tables' / 'gt_section_geometry_pitch.csv')))
    by = {}
    for r in geo:
        by.setdefault(r['series'], []).append(r)
    med = lambda rr, col: S.median(float(r[col]) for r in rr if r[col] not in ('', 'nan'))
    oth = [k for k in by if k != 'PVcont']
    pit = [med(by[k], 'nn_pitch_median_um') for k in oth]; den = [med(by[k], 'lesions_per_mm2') for k in oth]
    dia = [med(by[k], 'eq_diam_mm') for k in oth]
    put('OthPitchMin', min(pit), '{:.0f}'); put('OthPitchMax', max(pit), '{:.0f}')
    put('OthDenMin', min(den), '{:.1f}'); put('OthDenMax', max(den), '{:.1f}')
    put('OthDiamMin', min(dia), '{:.1f}'); put('OthDiamMax', max(dia), '{:.1f}')
    put('OthPitchRedMin', 100 * (1 - max(pit) / 714), '{:.0f}'); put('OthPitchRedMax', 100 * (1 - min(pit) / 714), '{:.0f}')
    put('PVPitch', med(by['PVcont'], 'nn_pitch_median_um'), '{:,.0f}'); put('PVDen', med(by['PVcont'], 'lesions_per_mm2'), '{:.2f}')
    put('PVDiam', med(by['PVcont'], 'eq_diam_mm'), '{:.2f}')
    put('PVShrinkPct', 100 * (1 - med(by['PVcont'], 'eq_diam_mm') / 8.0), '{:.1f}')
    put('PVPredPitch', 714 * med(by['PVcont'], 'eq_diam_mm') / 8.0, '{:.0f}')
    put('PVPitchOverPred', med(by['PVcont'], 'nn_pitch_median_um') / (714 * med(by['PVcont'], 'eq_diam_mm') / 8.0), '{:.1f}')
    put('PVPitchNominal', med(by['PVcont'], 'nn_pitch_median_um') / 714, '{:.1f}')
    put('PVPitchRatioLo', med(by['PVcont'], 'nn_pitch_median_um') / max(pit), '{:.1f}'); put('PVPitchRatioHi', med(by['PVcont'], 'nn_pitch_median_um') / min(pit), '{:.1f}')
    put('NOtherSeries', len(oth), '{:d}')
    # ten-lesion sampling, ranges over sections
    sm = list(csv.DictReader(open(P / 'r3_manual_sampling' / 'manual_sampling_simulation.csv')))
    col = lambda k: [float(r[k]) for r in sm]
    put('SampLesMin', min(col('n_lesions')), '{:.0f}'); put('SampLesMax', max(col('n_lesions')), '{:.0f}')
    put('SampCVMin', min(col('full_cv')), '{:.2f}'); put('SampCVMax', max(col('full_cv')), '{:.2f}')
    put('SampWithinMin', 100 * min(col('random10_cv_within_20pct')), '{:.0f}'); put('SampWithinMax', 100 * max(col('random10_cv_within_20pct')), '{:.0f}')
    put('SampWellMin', min(col('wellformed10_cv_bias_pct')), '{:+.0f}'); put('SampWellMax', max(col('wellformed10_cv_bias_pct')), '{:+.0f}')
    put('SampNSections', len(sm), '{:d}')
    if sel:
        put('ValFinal', sel['val_macro_dice'], '{:.4f}'); put('ValFinalBg', sel.get('val_macro_dice_interior_bg'), '{:.4f}')
        bs = max(sel['single_scores'], key=sel['single_scores'].get); put('ValBestSingle', sel['single_scores'][bs], '{:.4f}')
        d['ValBestSingleName'] = bs
        d['FinalDecoding'] = 'nested decoding with interior-background re-decoding' if sel.get('interior_bg') else 'nested decoding'
    o = (pipe(load(P / 'r3_greedy' / 'stats_filtered_nested.json'), 'filtered_nested') or {}).get('object', {})
    if 'tol_100um' in o:
        put('FinalRecHundred', o['tol_100um']['all']['recall'] * 100, '{:.1f}'); put('FinalPrecHundred', o['tol_100um']['all']['precision'] * 100, '{:.1f}')
        d['FinalTolOneFiftySame'] = 'identical' if o['tol_150um']['all'] == o['tol_250um']['all'] else 'different'
    e5 = load(P / 'r3_ensemble' / 'filter_params_nested.json'); fv = load(P / 'r3_greedy' / 'filter_params_nested.json')
    if e5 and fv:
        put('ValEFiveFilt', e5['chosen']['macro_dice']); put('ValFinalFilt', fv['chosen']['macro_dice'])
    fs = load(ROOT / 'manuscript' / 'figures' / 'r3_figure_sources.json')
    if fs:
        put('FigSegMacro', fs['segmentation']['macro']); put('FigSegAbl', fs['segmentation']['dice_per_class'][3]); put('FigSegCoag', fs['segmentation']['dice_per_class'][2])
    gl = load(P / 'r3_glahn_comparison.json')
    if gl:
        c, pa, pu = gl['control'], gl['papain'], gl['papain_urea']
        put('GlCtlAZ', c['az_mean'], '{:.4f}'); put('GlCtlCZ', c['cz_mean']); put('GlPapCZ', pa['cz_mean']); put('GlPUCZ', pu['cz_mean'])
        put('GlCtlAZDiff', 100 * (c['az_mean'] / 0.0091 - 1), '{:+.0f}'); put('GlCtlCZDiff', 100 * (c['cz_mean'] / 0.051 - 1), '{:+.0f}')
        put('GlPapCZDiff', 100 * (pa['cz_mean'] / 0.044 - 1), '{:+.0f}')
        put('GlPapChange', gl['cz_change_papain_pct'], '{:+.0f}'); put('GlPUChange', gl['cz_change_papain_urea_pct'], '{:+.0f}')
        put('GlPUAZChange', gl['az_change_papain_urea_pct'], '{:+.0f}')
        d['GlNLesions'] = f"{c['n_lesions'] + pa['n_lesions'] + pu['n_lesions']:,}"
    # unannotated example series (final pipeline): superficial within-section CV, range of series means
    ex = []
    for e in EXAMPLE_SERIES:
        f = P / 'r3_example_series' / f'{e}_sections.csv'
        if f.exists():
            rr = [r for r in csv.DictReader(open(f)) if float(r['depth_um']) < 60 and float(r['n_lumen']) >= 10]
            v = lambda k: S.mean(float(r[k]) for r in rr if r[k] not in ('', 'nan'))
            ex.append((v('cv_abl'), v('cv_coag'), v('cv_les')))
    if ex:
        for i, k in enumerate(('Abl', 'Coag', 'Les')):
            put(f'ExSup{k}CVMin', min(x[i] for x in ex), '{:.2f}'); put(f'ExSup{k}CVMax', max(x[i] for x in ex), '{:.2f}')
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in d.items())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, fin, sel = model_rows()
    (OUT / 'numbers.tex').write_text(numbers(rows, fin, sel) + '\n')
    (OUT / 'table_models.tex').write_text(table_models(rows) + '\n')
    (OUT / 'table_depth.tex').write_text(table_depth(rows[0][4], fin) + '\n')
    (OUT / 'table_census.tex').write_text(table_census() + '\n')
    (OUT / 'table_geometry.tex').write_text(table_geometry() + '\n')
    (OUT / 'table_sampling.tex').write_text(table_sampling() + '\n')
    print('wrote', OUT)
    print((OUT / 'numbers.tex').read_text())


if __name__ == '__main__':
    main()
