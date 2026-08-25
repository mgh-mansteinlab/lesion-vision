import math
import os
import re
from itertools import product
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# Default number of worker processes (e.g. 170 out of 192 CPUs; set via --workers)
DEFAULT_N_WORKERS = 170

# Directories to skip when scanning for images/masks
EXCLUDE_DIRS = frozenset(("tile_mask_bkp", "predictions", "test"))


class TileGenerator:
    """
    Generates aligned tile pairs from an image (.tif) and its mask (.png).
    At each position the mask tile is checked first — if it is pure white
    (background), both the image tile and mask tile are skipped.
    """
    DEFAULT_OVERLAP_PCTS = (20, 40, 60, 80)

    def __init__(self, image_path, mask_path, tile_size, overlap_pcts=None, output_root=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.tile_size = tile_size
        self.overlap_pcts = overlap_pcts if overlap_pcts is not None else list(self.DEFAULT_OVERLAP_PCTS)
        self.output_root = os.path.abspath(output_root) if output_root else None
        self.name, _ = os.path.splitext(os.path.basename(image_path))
        self.img = None
        self.mask = None

    def _create_output_dirs(self, shift_folder):
        base = self.output_root if self.output_root is not None else os.path.dirname(self.image_path)
        tile_dir = os.path.join(base, f"tile_{self.tile_size}", shift_folder)
        mask_dir = os.path.join(base, f"mask_{self.tile_size}", shift_folder)
        os.makedirs(tile_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        return tile_dir, mask_dir

    def _get_shift_folder(self, shift_y, shift_x):
        if shift_y == 0 and shift_x == 0:
            return "no_shift"
        pct = round(100 * shift_y / self.tile_size) if self.tile_size else 0
        if pct in self.overlap_pcts:
            return f"ovlp_{pct}"
        return None

    def _get_shifts(self):
        shifts = [(0, 0)]
        for pct in sorted(self.overlap_pcts):
            if pct <= 0 or pct >= 100:
                continue
            shift = (self.tile_size * pct) // 100
            step = self.tile_size - shift
            if step <= 0:
                continue
            shifts.append((shift, shift))
        return shifts

    def _tile_positions(self, length, step):
        max_start = max(0, length - self.tile_size)
        if max_start < 0:
            return []
        if step <= 0:
            return [0] if self.tile_size <= length else []
        positions = list(range(0, max_start + 1, step))
        if positions and positions[-1] != max_start:
            positions.append(max_start)
        return positions

    @staticmethod
    def _is_background_tile(mask_tile, threshold=240, min_bg_pct=0.995):
        """Return True if the mask tile is almost entirely white (background)."""
        arr = np.asarray(mask_tile)
        if arr.ndim == 3:
            white = np.all(arr > threshold, axis=-1)
        else:
            white = arr > threshold
        return white.mean() >= min_bg_pct

    def _process_tile(self, i, j, tile_dir, mask_dir):
        if i + self.tile_size > self.img.height or j + self.tile_size > self.img.width:
            return
        box = (j, i, j + self.tile_size, i + self.tile_size)
        mask_tile = self.mask.crop(box)
        if self._is_background_tile(mask_tile):
            self._skipped_bg += 1
            return
        img_tile = self.img.crop(box)
        img_tile.save(os.path.join(tile_dir, f'{self.name}_{i}_{j}.tif'))
        mask_tile.save(os.path.join(mask_dir, f'{self.name}_{i}_{j}.png'))

    def generate_tiles(self, disable_tqdm=False):
        self.img = Image.open(self.image_path)
        self.mask = Image.open(self.mask_path).convert("RGB")

        if self.img.size != self.mask.size:
            if not disable_tqdm:
                tqdm.write(f"  {self.name}: resizing mask {self.mask.size} -> {self.img.size}")
            self.mask = self.mask.resize(self.img.size, Image.NEAREST)

        if not disable_tqdm:
            tqdm.write(f"{self.name} {self.img.size}")

        self._skipped_bg = 0
        total_tiles = 0
        for shift_y, shift_x in self._get_shifts():
            shift_folder = self._get_shift_folder(shift_y, shift_x)
            if not shift_folder:
                continue
            tile_dir, mask_dir = self._create_output_dirs(shift_folder)
            step_y = self.tile_size - shift_y
            step_x = self.tile_size - shift_x
            positions_i = self._tile_positions(self.img.height, step_y)
            positions_j = self._tile_positions(self.img.width, step_x)
            pairs = list(product(positions_i, positions_j))
            total_tiles += len(pairs)
            for i, j in tqdm(pairs, desc=f"{self.name} {shift_folder}", unit="tile", leave=False, disable=disable_tqdm):
                self._process_tile(i, j, tile_dir, mask_dir)

        kept = total_tiles - self._skipped_bg
        if not disable_tqdm:
            tqdm.write(f"  {self.name}: {kept}/{total_tiles} tiles kept, "
                       f"{self._skipped_bg} background tiles skipped")


def discover_images_and_masks(root_dir):
    """
    Recursively find all .tif (images) and .png (masks) under root_dir,
    skipping paths that contain any of EXCLUDE_DIRS (tile_mask_bkp, predictions, test).
    Returns (images_list, masks_list) with absolute paths, sorted.
    """
    root_dir = os.path.abspath(root_dir)
    images = []
    masks = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories (don't descend into them)
        rel_parts = os.path.relpath(dirpath, root_dir).split(os.sep)
        if any(d in EXCLUDE_DIRS for d in rel_parts):
            continue
        for f in filenames:
            path = os.path.join(dirpath, f)
            if f.lower().endswith(".tif"):
                images.append(path)
            elif f.lower().endswith(".png"):
                masks.append(path)
    return (sorted(images), sorted(masks))


def find_image_mask_combinations(root_dir):
    """
    Discover images and masks under root_dir (with exclusions), then pair them
    by base name. Pairs (image_path, mask_path); mask_path is None if no mask found.
    Pairing rules: same dir same base name, or image in .../images/ -> mask in .../masks/.
    Returns list of (image_path, mask_path or None), and list of standalone masks
    (masks with no paired image) for reporting.
    """
    images, masks = discover_images_and_masks(root_dir)
    mask_by_basename = {}  # (dir, base) -> path for masks in that "logical" dir
    for m in masks:
        d = os.path.dirname(m)
        base = os.path.splitext(os.path.basename(m))[0]
        key = (d, base)
        mask_by_basename[key] = m
    # Also index masks by (parent_dir, base) when mask is under parent/masks/
    for m in masks:
        d = os.path.dirname(m)
        base = os.path.splitext(os.path.basename(m))[0]
        parent = os.path.dirname(d)
        if os.path.basename(d) == "masks":
            mask_by_basename[(parent, base)] = m

    combinations = []
    for img_path in images:
        d = os.path.dirname(img_path)
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = mask_by_basename.get((d, base))
        if mask_path is None:
            parent = os.path.dirname(d)
            mask_path = mask_by_basename.get((parent, base))
        combinations.append((img_path, mask_path))
    return combinations


def inspect_tile_output(output_root, tile_sizes=(256, 384, 448, 512, 640, 768, 1024)):
    """
    Dry run: scan output_root for existing tile/mask files and report how much
    processing has completed. Use when the main process appears stuck to see
    if workers are writing tiles to disk.
    """
    output_root = os.path.abspath(output_root)
    if not os.path.isdir(output_root):
        print(f"[TileGen inspect] Output root does not exist: {output_root}")
        return

    # Tile filenames: {name}_{i}_{j}.tif or .png -> extract name for task identity
    name_suffix_re = re.compile(r"_(\d+)_(\d+)\.(tif|png)$", re.IGNORECASE)

    total_tile_files = 0
    completed_tasks = set()  # (basename, tile_size, 'tile'|'mask')
    by_size_type = {}  # (tile_size, type) -> file count

    for tile_size in tile_sizes:
        for prefix in ("tile", "mask"):
            top = os.path.join(output_root, f"{prefix}_{tile_size}")
            if not os.path.isdir(top):
                continue
            for shift_name in os.listdir(top):
                shift_dir = os.path.join(top, shift_name)
                if not os.path.isdir(shift_dir):
                    continue
                for f in os.listdir(shift_dir):
                    if not (f.endswith(".tif") or f.endswith(".png")):
                        continue
                    total_tile_files += 1
                    key = (tile_size, prefix)
                    by_size_type[key] = by_size_type.get(key, 0) + 1
                    m = name_suffix_re.search(f)
                    if m:
                        name = f[: m.start()]
                        completed_tasks.add((name, tile_size, prefix))

    # Expected total tasks (same logic as run_tile_gen_for_combinations)
    n_expected = 98 * len(tile_sizes)  # 98 files * 7 sizes = 686
    n_completed = len(completed_tasks)

    print(f"[TileGen inspect] Output root: {output_root}")
    print(f"[TileGen inspect] Total tile files on disk: {total_tile_files}")
    print(f"[TileGen inspect] Tasks with at least one tile: {n_completed} / {n_expected}")
    if by_size_type:
        print("[TileGen inspect] By (tile_size, type):")
        for (ts, typ), cnt in sorted(by_size_type.items()):
            print(f"  {typ}_{ts}: {cnt} files")
    print("[TileGen inspect] Done.")


def _tile_gen_worker(args):
    """
    Worker entry point for multiprocessing.Pool.
    Each worker tiles one image–mask pair at one tile size.
    """
    image_path, mask_path, tile_size, overlap_pcts, output_root = args
    TileGenerator(
        image_path, mask_path, tile_size,
        overlap_pcts=overlap_pcts,
        output_root=output_root,
    ).generate_tiles(disable_tqdm=True)
    return image_path


def run_tile_gen_for_combinations(root_dir, tile_sizes=(448,), overlap_pcts=None, output_root=None, n_workers=None):
    """
    Discover image–mask pairs under root_dir, then tile each pair jointly.
    The mask is checked at every tile position; pure-white (background) tiles
    are skipped for both image and mask so the output stays perfectly paired.
    """
    combinations = find_image_mask_combinations(root_dir)
    pairs = [(img, msk) for img, msk in combinations if msk is not None]
    unpaired = [img for img, msk in combinations if msk is None]
    if unpaired:
        print(f"[TileGen] Warning: {len(unpaired)} image(s) have no mask and will be skipped:")
        for p in unpaired:
            print(f"  {p}")

    out_root = os.path.abspath(output_root) if output_root else None
    overlap_list = list(overlap_pcts) if overlap_pcts is not None else list(TileGenerator.DEFAULT_OVERLAP_PCTS)

    tasks = []
    for img_path, mask_path in pairs:
        for ts in tile_sizes:
            tasks.append((img_path, mask_path, ts, overlap_list, out_root))

    n_workers = n_workers if n_workers is not None else DEFAULT_N_WORKERS

    print(f"[TileGen] Input: {len(pairs)} image–mask pairs.")
    print(f"[TileGen] Tile sizes: {tile_sizes} -> {len(tile_sizes)} size(s) per pair.")
    print(f"[TileGen] Total tasks: {len(tasks)} (each task = one pair at one tile size).")

    if n_workers <= 1:
        print("[TileGen] Running sequentially (n_workers <= 1).")
        for args in tqdm(tasks, desc="Tasks", unit="task"):
            _tile_gen_worker(args)
        print("[TileGen] Sequential run complete.")
        return

    print(f"[TileGen] Starting multiprocessing with {n_workers} worker processes...")
    with Pool(n_workers) as pool:
        print(f"[TileGen] Pool created. Submitting {len(tasks)} tasks (imap_unordered).")
        completed = list(tqdm(
            pool.imap_unordered(_tile_gen_worker, tasks),
            total=len(tasks),
            desc="Tasks",
            unit="task",
        ))
    print(f"[TileGen] Multiprocessing complete. Processed {len(completed)} tasks.")


MASK_COLORS = {
    0: (255, 255, 255),  # Background – white
    1: (0, 255, 0),      # Tissue – green
    2: (0, 0, 255),      # Coagulation – blue
    3: (255, 0, 0),      # Ablation – red
}
CLASS_NAMES = {0: "BG", 1: "Tissue", 2: "Coag", 3: "Ablat"}


def _mask_to_rgb(mask_img):
    """Convert a class-coloured mask PIL image to a labelled RGB array and class stats string."""
    arr = np.asarray(mask_img.convert("RGB"))
    h, w = arr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    stats_parts = []
    for cls_idx, color in sorted(MASK_COLORS.items()):
        if cls_idx == 0:
            hit = np.all(arr > 240, axis=-1)
        else:
            r, g, b = color
            hit = (
                (np.abs(arr[..., 0].astype(int) - r) < 30)
                & (np.abs(arr[..., 1].astype(int) - g) < 30)
                & (np.abs(arr[..., 2].astype(int) - b) < 30)
            )
        pct = hit.mean() * 100
        if pct > 0.05:
            stats_parts.append(f"{CLASS_NAMES[cls_idx]}:{pct:.0f}%")
        canvas[hit] = color
    return canvas, "  ".join(stats_parts)


def visualize_sample_tiles(output_root, tile_size, shift="no_shift",
                           num_samples=16, save_path=None):
    """
    Save a grid of randomly sampled tile–mask pairs from the TileGen output.
    Returns the save path or None if no tiles found.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tile_dir = os.path.join(output_root, f"tile_{tile_size}", shift)
    mask_dir = os.path.join(output_root, f"mask_{tile_size}", shift)
    if not os.path.isdir(tile_dir):
        print(f"[visualize] tile dir not found: {tile_dir}")
        return None

    tile_files = sorted(f for f in os.listdir(tile_dir) if f.endswith(".tif"))
    if not tile_files:
        print("[visualize] No tiles found to visualize.")
        return None

    rng = np.random.default_rng(42)
    n = min(num_samples, len(tile_files))
    selected = list(rng.choice(tile_files, n, replace=False))

    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5.5, rows * 3))
    if rows == 1 and cols * 2 == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    for idx, fname in enumerate(selected):
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(tile_dir, fname)
        msk_path = os.path.join(mask_dir, base + ".png")

        img_arr = np.asarray(Image.open(img_path))
        ax_img = axes_flat[idx * 2]
        ax_img.imshow(img_arr)
        ax_img.set_title(base, fontsize=7)
        ax_img.axis("off")

        ax_msk = axes_flat[idx * 2 + 1]
        if os.path.isfile(msk_path):
            msk_rgb, stats = _mask_to_rgb(Image.open(msk_path))
            ax_msk.imshow(msk_rgb)
            ax_msk.set_title(stats, fontsize=7)
        else:
            ax_msk.text(0.5, 0.5, "mask\nmissing", ha="center", va="center",
                        transform=ax_msk.transAxes, fontsize=9, color="red")
        ax_msk.axis("off")

    for i in range(n * 2, len(axes_flat)):
        axes_flat[i].axis("off")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=np.array(c) / 255, label=CLASS_NAMES[k])
              for k, c in MASK_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=len(MASK_COLORS), fontsize=9)
    fig.suptitle(f"TileGen sample  |  tile_{tile_size}/{shift}  |  {n}/{len(tile_files)} tiles shown",
                 fontsize=11, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if save_path is None:
        save_path = os.path.join(output_root, f"tilegen_sample_{tile_size}_{shift}.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved sample grid ({n} tiles) -> {save_path}")
    return save_path


def _run_single_pair(image_path, mask_path, tile_size, overlap_pcts, output_root, visualize):
    """Mode 1: tile a single image–mask pair."""
    print(f"[TileGen] Single-pair mode")
    print(f"  image: {image_path}")
    print(f"  mask:  {mask_path}")
    print(f"  tile_size: {tile_size}  output: {output_root}")
    TileGenerator(
        image_path, mask_path, tile_size,
        overlap_pcts=overlap_pcts,
        output_root=output_root,
    ).generate_tiles(disable_tqdm=False)
    if visualize:
        visualize_sample_tiles(output_root, tile_size, shift="no_shift")


def _run_directory(input_dir, tile_sizes, overlap_pcts, output_root, n_workers, visualize, dry_run):
    """Mode 2: discover all pairs in a directory and tile them."""
    combinations = find_image_mask_combinations(input_dir)
    pairs = [(img, msk) for img, msk in combinations if msk is not None]
    unpaired = [img for img, msk in combinations if msk is None]
    print(f"[TileGen] Directory mode: {input_dir}")
    print(f"  Found {len(pairs)} image–mask pair(s)")
    if unpaired:
        print(f"  ({len(unpaired)} image(s) without masks — will be skipped)")
    print(f"  Tile sizes: {tile_sizes}  output: {output_root}")
    print("  Pairs:")
    for img_p, msk_p in pairs:
        print(f"    {os.path.basename(img_p)} -> {os.path.basename(msk_p)}")
    if dry_run:
        os.makedirs(output_root, exist_ok=True)
        print("  Dry run complete (no tiles generated).")
        return
    if not pairs:
        print("  Nothing to do.")
        return
    run_tile_gen_for_combinations(
        input_dir, tile_sizes=tile_sizes, overlap_pcts=overlap_pcts,
        output_root=output_root, n_workers=n_workers,
    )
    if visualize:
        for ts in tile_sizes:
            visualize_sample_tiles(output_root, ts, shift="no_shift")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate aligned image–mask tile pairs with background filtering.",
        epilog=(
            "Run modes:\n"
            "  Mode 1 (single pair):  --image IMG.tif --mask MSK.png\n"
            "  Mode 2 (directory):    --dir /path/to/data\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_argument_group("input (pick one)")
    mode.add_argument("--image", type=str, default=None,
                      help="Path to a single .tif image (requires --mask)")
    mode.add_argument("--mask", type=str, default=None,
                      help="Path to the corresponding .png mask (requires --image)")
    mode.add_argument("--dir", type=str, default=None,
                      help="Directory to scan for .tif/.png pairs")

    parser.add_argument("--output", type=str, required=True,
                        help="Output root directory (e.g. path/to/tiles)")
    parser.add_argument("--tile-size", type=int, nargs="+", default=[448],
                        help="Tile size(s) to generate (default: 448)")
    parser.add_argument("--workers", type=int, default=DEFAULT_N_WORKERS,
                        help=f"Parallel workers for directory mode (default: {DEFAULT_N_WORKERS})")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a sample visualization grid after tiling")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show discovered pairs without generating tiles")
    parser.add_argument("--inspect", action="store_true",
                        help="Scan output dir for existing tiles and report")

    args = parser.parse_args()

    if args.inspect:
        inspect_tile_output(args.output, tile_sizes=tuple(args.tile_size))
        raise SystemExit(0)

    if args.image and args.mask:
        if not os.path.isfile(args.image):
            parser.error(f"Image file not found: {args.image}")
        if not os.path.isfile(args.mask):
            parser.error(f"Mask file not found: {args.mask}")
        _run_single_pair(
            os.path.abspath(args.image),
            os.path.abspath(args.mask),
            tile_size=args.tile_size[0],
            overlap_pcts=TileGenerator.DEFAULT_OVERLAP_PCTS,
            output_root=os.path.abspath(args.output),
            visualize=args.visualize,
        )
    elif args.dir:
        if not os.path.isdir(args.dir):
            parser.error(f"Directory not found: {args.dir}")
        _run_directory(
            os.path.abspath(args.dir),
            tile_sizes=tuple(args.tile_size),
            overlap_pcts=TileGenerator.DEFAULT_OVERLAP_PCTS,
            output_root=os.path.abspath(args.output),
            n_workers=args.workers,
            visualize=args.visualize,
            dry_run=args.dry_run,
        )
    else:
        parser.error("Provide either --image IMG --mask MSK  or  --dir DIR")
