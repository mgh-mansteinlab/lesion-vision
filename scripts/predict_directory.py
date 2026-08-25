#!/usr/bin/env python
"""Headless batch prediction over a directory of slides (multi-sample aware).

This is the canonical non-interactive batch predictor (the interactive entry
point is ``scripts/pipeline_cli.py``). It supersedes the old, CPBS-specific
``predict_cpbs.py``.

NDPI files often contain several tissue samples side by side. Every sample is
extracted to its own TIF (``<base>_sample_01.tif``, ``_02``, ...) and predicted
separately so nothing is skipped. Each sample's prediction is written to its
own subfolder (``<out>/<base>_sample_NN/images/...``) with the colored mask,
overlay, and visualization triptych.

Work is sharded one-slide-per-GPU across the available GPUs; the model is loaded
once per worker and reused for all of that worker's samples.

Optionally collects all ``*_pred.png`` masks into one folder and chains the 3D
reconstruction pipeline when the collected stack has >= 2 sections.

Examples:
    python scripts/predict_directory.py --data-dir /path/to/slides
    python scripts/predict_directory.py --data-dir /path/to/slides \
        --model models/single_scale_448/best_model.pth \
        --collect-dir /path/to/slides/all_predictions --reconstruct
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))

import argparse
import glob
import multiprocessing as mp
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_MODEL = os.path.join(REPO, 'models', 'single_scale_448', 'best_model.pth')

_DEVICE = None
_MODEL_PATH = None
_PREDICTOR = None


def _init_worker(gpu_queue, model_path):
    global _DEVICE, _MODEL_PATH
    import torch
    gid = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gid)
        _DEVICE = f'cuda:{gid}'
    else:
        _DEVICE = 'cpu'
    _MODEL_PATH = model_path
    print(f"[worker pid={os.getpid()}] device={_DEVICE}", flush=True)


def _get_predictor():
    """Lazily build the predictor once per worker process."""
    global _PREDICTOR
    if _PREDICTOR is None:
        from src.prediction import LesionPredictor
        _PREDICTOR = LesionPredictor(_MODEL_PATH, device=_DEVICE)
    return _PREDICTOR


def _samples_for(image_path, out_dir, ndpi_level):
    """Return list of TIF paths to predict for one input file.

    For NDPI inputs, extract every detected sample to <out_dir> as
    <base>_sample_NN.tif. For plain images, just return the path itself.
    """
    if not image_path.lower().endswith('.ndpi'):
        return [image_path]
    from src.ndpi_processor import NDPIExtractor
    extractor = NDPIExtractor(image_path, out_dir)
    return extractor.extract_to_tif(level=ndpi_level)


def _process_one(args):
    image_path, out_dir, tile_size, overlap, post, ndpi_level = args
    import torch
    base = os.path.basename(image_path)
    results = []
    try:
        sample_paths = _samples_for(image_path, out_dir, ndpi_level)
    except Exception as e:
        traceback.print_exc()
        return [(base, f'EXTRACT_ERROR: {e}')]

    predictor = _get_predictor()
    for sp in sample_paths:
        sname = os.path.basename(sp)
        try:
            output_path = os.path.join(out_dir, os.path.splitext(sname)[0] + '_pred.png')
            predictor.save_prediction(
                sp, output_path,
                tile_size=tile_size, overlap=overlap,
                summary_metrics=False, figures=False,
                create_subdir=True, use_tta=False, use_post_process=post,
            )
            results.append((sname, 'OK'))
        except Exception as e:
            traceback.print_exc()
            results.append((sname, f'PREDICT_ERROR: {e}'))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description="Headless multi-sample batch prediction over a directory")
    ap.add_argument('--data-dir', required=True, help='Directory of .tif/.ndpi slides')
    ap.add_argument('--out-dir', default=None,
                    help='Write extracted TIFs and predictions here (default: data-dir)')
    ap.add_argument('--recursive', action='store_true',
                    help='Find .tif/.ndpi in subdirectories as well')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--tile-size', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--ndpi-level', type=int, default=2)
    ap.add_argument('--gpus', type=int, default=None)
    ap.add_argument('--no-post-process', action='store_true')
    ap.add_argument('--collect-dir', default=None,
                    help='If set, copy every *_pred.png here after prediction')
    ap.add_argument('--reconstruct', action='store_true',
                    help='Chain 3D reconstruction on the collected stack (implies --collect-dir)')
    args = ap.parse_args()

    import torch
    n_gpus = args.gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    n_gpus = max(1, n_gpus)

    images = []
    pattern_root = args.data_dir
    for ext in ('*.tif', '*.ndpi'):
        if args.recursive:
            images.extend(sorted(glob.glob(os.path.join(pattern_root, '**', ext), recursive=True)))
        else:
            images.extend(sorted(glob.glob(os.path.join(pattern_root, ext))))
    # Skip already-extracted per-sample tifs to avoid double work on re-runs.
    images = sorted(set(p for p in images if '_sample_' not in os.path.basename(p)))
    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Model: {args.model}")
    print(f"Found {len(images)} source slides in {args.data_dir}, using {n_gpus} GPUs")
    print(f"Output: {out_dir}")
    for im in images:
        print("  ", os.path.basename(im))

    n_workers = min(n_gpus, len(images)) or 1
    ctx = mp.get_context('spawn')
    gpu_queue = ctx.Queue()
    for i in range(n_workers):
        gpu_queue.put(i % (torch.cuda.device_count() if torch.cuda.is_available() else 1))

    post = not args.no_post_process
    tasks = [(im, out_dir, args.tile_size, args.overlap, post, args.ndpi_level)
             for im in images]

    all_results = []
    with ctx.Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(gpu_queue, args.model)) as pool:
        for res in pool.imap_unordered(_process_one, tasks):
            for name, status in res:
                print(f"  -> {name}: {status}", flush=True)
            all_results.extend(res)

    ok = sum(1 for _, s in all_results if s == 'OK')
    print(f"\nDone: {ok}/{len(all_results)} samples predicted successfully.")
    for name, s in all_results:
        if s != 'OK':
            print(f"  FAILED {name}: {s}")

    collect_dir = args.collect_dir
    if args.reconstruct and not collect_dir:
        collect_dir = os.path.join(args.data_dir, 'all_predictions')

    if collect_dir:
        from scripts.collect_predictions import collect_predictions
        n = collect_predictions(out_dir, collect_dir)
        print(f"Collected {n} prediction mask(s) into {collect_dir}")

        if args.reconstruct:
            from src.reconstruction_3d import MIN_STACK_SIZE, ReconstructionError, run_reconstruction
            if n < MIN_STACK_SIZE:
                print(f"Skipping 3D: need >= {MIN_STACK_SIZE} sections, have {n}.")
            else:
                try:
                    out = run_reconstruction(collect_dir)
                    print(f"3D reconstruction complete -> {out}/model/lesion_3d_surfaces.html")
                except ReconstructionError as e:
                    print(f"3D reconstruction failed: {e}")


if __name__ == '__main__':
    main()
