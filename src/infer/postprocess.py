"""Spatial-constraint post-processing for predicted class masks."""

import cv2
import numpy as np
from scipy import ndimage

from src.constants import (
    CLASS_ID_ABLATION as ABL,
    CLASS_ID_BACKGROUND as BG,
    CLASS_ID_COAGULATION as COAG,
    CLASS_ID_TISSUE as TISSUE,
)


def post_process_mask(mask: np.ndarray) -> np.ndarray:
    """Apply spatial-constraint post-processing to a predicted mask.

    Passes enforcing the concentric lesion model
    (tissue -> coagulation -> ablation):

      1. Reassign ONLY background/tissue regions that are fully enclosed by
         lesion (true holes within a red/blue ring) to the nearest lesion
         class by distance.  Open white/tissue areas that connect to the
         outside of the lesion are left untouched.
      2. Merge fragmented ablation per lesion — within each coag
         contour, multiple ablation blobs are unified via convex
         hull (constrained to the coag interior).
      3. Fill remaining non-ablation holes inside ablation zones.
      4. Enforce ablation containment — any ablation pixel adjacent
         to tissue or background becomes coagulation.

    Operates on a copy and returns the corrected mask.
    """
    mask = mask.copy()
    h, w = mask.shape

    # ---- Pass 1: reassign only TRULY ENCLOSED holes inside lesions ----
    # A "hole" is a connected region of non-lesion pixels that is entirely
    # surrounded by lesion (coag/ablation).  We find these via hole-filling:
    # binary_fill_holes fills enclosed background; the difference with the
    # original lesion mask is exactly the set of enclosed holes.  Open white
    # areas inside the tissue (which connect to the sample exterior) are NOT
    # enclosed and are therefore preserved.
    lesion_binary = ((mask == COAG) | (mask == ABL))
    lesion_filled = ndimage.binary_fill_holes(lesion_binary).astype(np.uint8)

    holes = (lesion_filled == 1) & ~lesion_binary
    n_holes = int(np.sum(holes))

    if n_holes > 0:
        coag_src = ((mask == COAG) & ~holes).astype(np.uint8)
        abl_src = ((mask == ABL) & ~holes).astype(np.uint8)
        has_coag, has_abl = np.any(coag_src), np.any(abl_src)

        if has_coag and has_abl:
            dist_to_coag = cv2.distanceTransform(
                (1 - coag_src).astype(np.uint8), cv2.DIST_L2, 5)
            dist_to_abl = cv2.distanceTransform(
                (1 - abl_src).astype(np.uint8), cv2.DIST_L2, 5)
            mask[holes] = np.where(
                dist_to_abl[holes] <= dist_to_coag[holes], ABL, COAG
            ).astype(np.uint8)
        elif has_abl:
            mask[holes] = ABL
        else:
            mask[holes] = COAG

        print(f"Post-processing: reassigned {n_holes} enclosed hole pixels "
              "to nearest lesion class")

    # ---- Pass 2: merge fragmented ablation per lesion ----
    # Each coag contour represents one lesion. If multiple ablation
    # blobs exist inside the same coag contour they are fragments of
    # a single ablation zone.  Merge them via convex hull, constrained
    # to within the coag contour so we don't spill into tissue.
    coag_binary = (mask == COAG).astype(np.uint8)
    coag_close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    coag_closed = cv2.morphologyEx(coag_binary, cv2.MORPH_CLOSE, coag_close_k)
    coag_contours, _ = cv2.findContours(
        coag_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_merged = 0
    for cc in coag_contours:
        # Build a filled mask for this coag contour
        coag_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(coag_roi, [cc], -1, 1, cv2.FILLED)

        # Find ablation blobs inside this coag contour
        abl_in_coag = ((mask == ABL) & (coag_roi == 1)).astype(np.uint8)
        n_abl, abl_ccs = cv2.connectedComponents(abl_in_coag)
        if n_abl <= 2:
            # 0 or 1 ablation blob (label 0 is background) — nothing to merge
            continue

        # Multiple ablation fragments: compute convex hull of all
        # ablation pixels inside this coag contour and fill.
        abl_points = np.column_stack(np.where(abl_in_coag > 0))
        if len(abl_points) < 3:
            continue
        hull_pts = abl_points[:, ::-1]  # (row,col) -> (x,y) for cv2
        hull = cv2.convexHull(hull_pts)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 1, cv2.FILLED)

        # Constrain to within both the coag contour and the overall
        # lesion boundary, so the hull cannot leak into tissue.
        to_fill = (hull_mask == 1) & (mask != ABL) & (lesion_filled == 1)
        cnt = int(np.sum(to_fill))
        if cnt > 0:
            mask[to_fill] = ABL
            n_merged += cnt

    if n_merged > 0:
        print(f"Post-processing: merged {n_merged} pixels bridging fragmented ablation zones")

    # ---- Pass 3: fill remaining non-ablation holes inside ablation ----
    abl_binary = (mask == ABL).astype(np.uint8)
    if np.any(abl_binary):
        abl_close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        abl_closed = cv2.morphologyEx(abl_binary, cv2.MORPH_CLOSE, abl_close_k)

        abl_contours, _ = cv2.findContours(
            abl_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        abl_filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(abl_filled, abl_contours, -1, 1, cv2.FILLED)

        interior = (abl_filled == 1) & (mask != ABL) & (lesion_filled == 1)
        n_interior = int(np.sum(interior))
        if n_interior > 0:
            mask[interior] = ABL
            print(f"Post-processing: filled {n_interior} non-ablation pixels inside ablation zones")

    # ---- Pass 4: enforce ablation containment (single pass) ----
    if np.any(mask == ABL):
        neighbor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        non_lesion = ((mask == BG) | (mask == TISSUE)).astype(np.uint8)
        exposed = (mask == ABL) & (cv2.dilate(non_lesion, neighbor_k) > 0)
        n_exposed = int(np.sum(exposed))
        if n_exposed > 0:
            mask[exposed] = COAG
            print(f"Post-processing: converted {n_exposed} exposed ablation pixels to coagulation")

    return mask
