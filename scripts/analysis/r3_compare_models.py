#!/usr/bin/env python3
"""Multi-model comparison on the test punch (PVcont), from finished evaluations.

Collects M1 (predictions/r3_stats*.json) and every model evaluated by
scripts/r3_eval_model.py (predictions/<tag>/summary.json + stats.json), and
writes predictions/r3_model_comparison/{COMPARISON.md, comparison.csv,
comparison.png}. Models not yet evaluated are listed as pending.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'predictions'
OUT = P / 'r3_model_comparison'
MODELS = [  # id, tag, architecture, encoder, params (M), role
    ('M1', None, 'UNet', 'ConvNeXt-Base', 92.7, 'pre-specified primary'),
    ('M2', 'r3m2', 'UNet', 'ConvNeXt-Base', 92.7, '+ hard negatives'),
    ('C1', 'r3_c1_upernet_convnextb', 'UPerNet', 'ConvNeXt-Base', 97.7, 'decoder'),
    ('C2', 'r3_c2_segformer_mitb3', 'SegFormer', 'MiT-B3', 44.6, 'transformer'),
    ('C3', 'r3_c3_deeplabv3plus_resnet50', 'DeepLabV3+', 'ResNet-50', 26.7, 'classic CNN'),
    ('C4', 'r3_c4_unetpp_effb3', 'UNet++', 'EfficientNet-B3', 13.6, 'lightweight CNN'),
]
FILTER_TAGS = {'M1': 'r3m1', 'M2': 'r3m2', 'C1': 'r3_c1_upernet_convnextb', 'C2': 'r3_c2_segformer_mitb3',
               'C3': 'r3_c3_deeplabv3plus_resnet50', 'C4': 'r3_c4_unetpp_effb3', 'ENS': 'r3_ensemble', 'FINAL': 'r3_greedy'}


def rows_from_stats(stats, mid, arch, enc, params, role, extra):
    out = []
    for pipe, p in stats['pipelines'].items():
        a, o = p['pixel']['all'], p['object']['tol_250um']
        ba = p['object']['bland_altman']
        out.append({'model': mid, 'architecture': arch, 'encoder': enc, 'params_M': params, 'role': role, 'pipeline': pipe, **extra,
                    'macro_dice': a['macro_dice'], 'ci95_lo': a['macro_dice_ci95'][0], 'ci95_hi': a['macro_dice_ci95'][1],
                    'abl_dice': a['per_class']['ablation']['dice'], 'coag_dice': a['per_class']['coagulation']['dice'],
                    'superficial': p['pixel']['superficial']['macro_dice'], 'middle': p['pixel']['middle']['macro_dice'],
                    'deep': p['pixel']['deep']['macro_dice'],
                    'det_recall': o['all']['recall'], 'det_precision': o['all']['precision'],
                    'deep_precision': o['deep']['precision'],
                    'abl_bias_um': ba['ablation_diameter_um']['bias'], 'abl_loa_width_um': ba['ablation_diameter_um']['loa'][1] - ba['ablation_diameter_um']['loa'][0],
                    'coag_bias_um': ba['coagulation_width_um']['bias'], 'coag_loa_width_um': ba['coagulation_width_um']['loa'][1] - ba['coagulation_width_um']['loa'][0]})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pending = [], []
    for mid, tag, arch, enc, params, role in MODELS:
        if mid == 'M1':
            st = json.load(open(P / 'r3_stats.json'))
            names = {'frozen': 'notta_pp', 'tta': 'tta_raw'}
            st['pipelines'] = {names[k]: v for k, v in st['pipelines'].items()}
            s80 = P / 'r3_stats_size80.json'
            if s80.exists():
                st['pipelines']['notta_pp_size80'] = json.load(open(s80))['pipelines']['frozen_size80']
            rows += rows_from_stats(st, mid, arch, enc, params, role, {'selected_epoch': 5, 'chosen_config': 'tta_raw'})
            continue
        sp, stp = P / tag / 'summary.json', P / tag / 'stats.json'
        if not (sp.exists() and stp.exists() and json.load(open(sp)).get('status') == 'done'):
            pending.append(mid); continue
        sm = json.load(open(sp))
        rows += rows_from_stats(json.load(open(stp)), mid, arch, enc, params, role,
                                {'selected_epoch': sm['selected_epoch'], 'chosen_config': sm['chosen_config'],
                                 'val_macro_dice': sm['val_checkpoints'][str(sm['selected_epoch'])]['macro_dice']})
    # secondary: object filter (tuned on validation) on argmax of probabilities, and the ensemble
    for mid, tag in FILTER_TAGS.items():
        for sfx in ('', '_nested'):
            fs, fp = P / tag / f'stats_filtered{sfx}.json', P / tag / f'filter_params{sfx}.json'
            if not fs.exists():
                continue
            sel = P / 'r3_greedy' / 'selection.json'
            final_members = '+'.join(json.load(open(sel))['selected']) if sel.exists() else '?'
            info = {m[0]: m for m in MODELS}.get(mid, (mid, None, 'Ensemble', 'M2+C1-C4', None, 'mean probabilities') if mid == 'ENS'
                    else (mid, None, 'Final (validation-selected)', final_members, None, 'greedy selection on validation'))
            prm = json.load(open(fp))['chosen'] if fp.exists() else {}
            extra = {'filter_tau': prm.get('tau'), 'filter_d_um': prm.get('d_um'), 'val_filtered_f1': prm.get('f1')}
            st = json.load(open(fs))
            st['pipelines'] = {f'probs_{k}': v for k, v in st['pipelines'].items()}
            rows += rows_from_stats(st, mid, info[2], info[3], info[4], info[5], extra)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'comparison.csv', index=False)
    show = ['model', 'architecture', 'encoder', 'params_M', 'selected_epoch', 'macro_dice', 'ci95_lo', 'ci95_hi', 'abl_dice', 'coag_dice',
            'superficial', 'middle', 'deep', 'det_recall', 'det_precision', 'deep_precision', 'abl_bias_um', 'coag_bias_um']
    lines = ['# Round-3 model comparison on PVcont (35 sections, full depth)', '',
             'Protocol: experiments/protocol_r3_comparison_*. Every model: same data, split and schedule; checkpoint and inference '
             'configuration chosen on validation; PVcont scored once. M1 is the pre-specified primary.', '']
    for title, pipe_filter in (('Default inference (no TTA, post-processing on): equal terms for all models', lambda s: s == 'notta_pp'),
                               ('Configuration chosen on validation (per model)', lambda s: not s.endswith('size80')),
                               ('Secondary: + 80 um size rule (default inference)', lambda s: s == 'notta_pp_size80'),
                               ('Secondary: probabilities, unfiltered argmax (no TTA, no post-processing)', lambda s: s == 'probs_argmax'),
                               ('Secondary: object filter tuned on validation (size, confidence, tissue edge); ENS = ensemble of M2+C1-C4', lambda s: s == 'probs_filtered'),
                               ('Nested decoding (complete >= 10 um coagulation rim), no filter', lambda s: s == 'probs_nested'),
                               ('Nested decoding + object filter tuned on validation; ENS = pre-specified equal-weight ensemble; FINAL = validation-selected pipeline', lambda s: s == 'probs_filtered_nested')):
        sub = df[df.pipeline.map(pipe_filter)].copy() if len(df) else df
        if title.startswith('Configuration'):
            sub = sub[sub.apply(lambda r: r.pipeline == r.chosen_config, axis=1)]
        lines += [f'## {title}', '', sub[show + (['pipeline'] if title.startswith('Configuration') else [])].round(3).to_markdown(index=False) if len(sub) else '(none yet)', '']
    lines += [f'Pending (not yet trained/evaluated): {", ".join(pending) if pending else "none"}', '',
              'Columns: superficial/middle/deep = macro-Dice by depth band (0-220, 240-460, 480-680 um); det = lesion detection at 250 um; '
              'bias = Bland-Altman mean difference (automated - expert).']
    (OUT / 'COMPARISON.md').write_text('\n'.join(lines))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        d = df[df.pipeline == 'notta_pp']
        if len(d):
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
            lab = d.model + '\n' + d.architecture
            ax[0].bar(lab, d.macro_dice, yerr=[d.macro_dice - d.ci95_lo, d.ci95_hi - d.macro_dice], capsize=4, color='tab:blue')
            ax[0].set_ylim(0.8, 1); ax[0].set_title('PVcont macro-Dice (95% CI)')
            for b, c in (('superficial', 'tab:green'), ('middle', 'tab:orange'), ('deep', 'tab:red')):
                ax[1].plot(lab, d[b], 'o-', color=c, label=b)
            ax[1].set_title('macro-Dice by depth band'); ax[1].legend()
            ax[2].bar(lab, d.det_precision, color='tab:purple', label='precision'); ax[2].plot(lab, d.det_recall, 'ko-', label='recall')
            ax[2].set_ylim(0, 1.05); ax[2].set_title('lesion detection'); ax[2].legend()
            fig.suptitle('Default inference (no TTA, post-processing on)'); fig.tight_layout(); fig.savefig(OUT / 'comparison.png', dpi=110)
    except Exception as e:
        print('plot skipped:', e)
    print((OUT / 'COMPARISON.md').read_text())


if __name__ == '__main__':
    main()
