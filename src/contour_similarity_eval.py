import cv2
import numpy as np

# Contour similarity metric (core)
def edges_from_image(img_rgb, canny1=80, canny2=160, blur_ksize=5):
    # Original image structural outline (for reference)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(gray, canny1, canny2)  # 0/255

def boundary_from_mask(mask01_or_255):
    # Convert the segmentation results to the same scale as the original one and get the boundary
    m = mask01_or_255.copy()
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, kernel, 1)
    bd = cv2.subtract(m, er)
    return bd

def boundary_f1(edge_pre, edge_post, tol=2):
    # Align edge_post and edge_pre spatially, allowing for a buffer error of tol=2 pixels
    a = (edge_pre > 0).astype(np.uint8)   # reference
    b = (edge_post > 0).astype(np.uint8)  # predicted
    if a.sum() == 0 or b.sum() == 0:
        return 0.0, 0.0, 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
    a_d = cv2.dilate(a, k, 1)
    b_d = cv2.dilate(b, k, 1)

    prec = (b & a_d).sum() / (b.sum() + 1e-9)
    rec  = (a & b_d).sum() / (a.sum() + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)

# Unify arbitrary segmentation output as a binary mask
def ensure_binary_mask(x, thr=127):
    # Standardise different outputs into a binary mask of uint8 (0/255).
    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(x, thr, 255, cv2.THRESH_BINARY)
    return m

def pick_best_mask_by_contour_similarity(img_rgb, candidates, tol=2):
    """
    From a set of candidate masks, automatically select the one whose edges most closely the original image.
    (K-means outputs not a single target mask, but multiple clusters/multiple regions.)
    """
    edge_pre = edges_from_image(img_rgb)
    best = None
    best_score = -1.0
    for m in candidates:
        bd = boundary_from_mask(m)
        _, _, f1 = boundary_f1(edge_pre, bd, tol=tol)
        if f1 > best_score:
            best_score = f1
            best = m
    return best, best_score

# The 'adapter' for the four methods
def mask_from_contour_method(masked_from_file):
    # ContourSegmentation.apply has returned mask(0/255)
    return ensure_binary_mask(masked_from_file)

def mask_from_otsu(thresh_from_file):
    # OtsuSegmentation.apply has returned to thresh(0/255)
    return ensure_binary_mask(thresh_from_file)

def mask_from_kmeans(img_rgb, segmented_rgb):
    """
    Convert the colour-coded K-means clustering results into an evaluable binary mask.
    Automatically determine which cluster is the target using “contour similarity”.
    """
    seg = segmented_rgb.reshape(-1, 3)
    colors, inv = np.unique(seg, axis=0, return_inverse=True)
    h, w, _ = segmented_rgb.shape
    candidates = []
    for k in range(colors.shape[0]):
        m = (inv.reshape(h, w) == k).astype(np.uint8) * 255
        # 过滤太小/太大区域，避免全图或噪声簇
        area = m.sum() / 255.0
        if area < 0.01 * h * w or area > 0.95 * h * w:
            continue
        candidates.append(m)
    if not candidates:
        # 兜底：直接灰度Otsu当mask
        return ensure_binary_mask(segmented_rgb)
    best, _ = pick_best_mask_by_contour_similarity(img_rgb, candidates, tol=2)
    return best
    
def mask_from_watershed(img_rgb, markers):
    # Merge all foregrounds from watershed as the result
    return ((markers > 1).astype(np.uint8) * 255)

# Unify the raw outputs from the four segmentation methods, then score them using "contour similarity" metrics.
def evaluate_methods(img_rgb, outputs_dict, tol=2):
    """
    outputs_dict：the raw outputs from the four segmentation methods
      {
        "contour": masked,
        "otsu": thresh,
        "kmeans": segmented_rgb,
        "watershed": markers
      }
    """
    edge_pre = edges_from_image(img_rgb)

    masks = {}
    masks["contour"] = mask_from_contour_method(outputs_dict["contour"])
    masks["otsu"] = mask_from_otsu(outputs_dict["otsu"])
    masks["kmeans"] = mask_from_kmeans(img_rgb, outputs_dict["kmeans"])
    masks["watershed"] = mask_from_watershed(img_rgb, outputs_dict["watershed"])

    scores = {}
    for name, m in masks.items():
        bd = boundary_from_mask(m)
        p, r, f1 = boundary_f1(edge_pre, bd, tol=tol)
        scores[name] = {"bf_precision": p, "bf_recall": r, "bf_f1": f1}
    return masks, scores
