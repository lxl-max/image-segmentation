import cv2
import numpy as np
from typing import Dict, Tuple


# ============================
# 基础工具：掩膜 / 边界 / 统一格式
# ============================

def ensure_binary_mask(x: np.ndarray, thr: int = 127) -> np.ndarray:
    """
    把各种形式的输出统一成 uint8 的二值掩膜 (0 / 255).
    """
    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(x, thr, 255, cv2.THRESH_BINARY)
    return m


def boundary_from_mask(mask01_or_255: np.ndarray) -> np.ndarray:
    """
    把分割掩膜转换为边界图 (0/255)，近似 BSDS 的 segmentation->boundary.
    """
    m = mask01_or_255.copy()
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, kernel, 1)
    bd = cv2.subtract(m, er)
    return bd  # 0 / 255


# ============================
# 四种方法输出 -> 掩膜 / “概率图”
# ============================

def mask_from_contour_method(masked_from_file: np.ndarray) -> np.ndarray:
    # ContourDetection.apply 已经返回目标掩膜(0/255 或 0/1)
    return ensure_binary_mask(masked_from_file)


def mask_from_otsu(thresh_from_file: np.ndarray) -> np.ndarray:
    # OtsuThresholding.apply 返回 thresh(0/255)
    return ensure_binary_mask(thresh_from_file)


def _pick_best_mask_by_contour_similarity(
    img_rgb: np.ndarray, candidates, tol: int = 2
) -> Tuple[np.ndarray, float]:
    """
    用“和原图 Canny 轮廓最接近”原则，从多个候选簇里选一个 mask。
    这里只用于从 kmeans 的多个 cluster 里选出一个目标区域。
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if tol > 0:
        gray = cv2.GaussianBlur(gray, (2 * tol + 1, 2 * tol + 1), 0)
    edge_ref = cv2.Canny(gray, 80, 160)  # 0/255

    def boundary_f1(edge_pre, edge_post, tol_pix=2):
        a = (edge_pre > 0).astype(np.uint8)   # reference
        b = (edge_post > 0).astype(np.uint8)  # predicted
        if a.sum() == 0 or b.sum() == 0:
            return 0.0, 0.0, 0.0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_pix + 1, 2 * tol_pix + 1))
        a_d = cv2.dilate(a, k, 1)
        b_d = cv2.dilate(b, k, 1)
        prec = (b & a_d).sum() / (b.sum() + 1e-9)
        rec = (a & b_d).sum() / (a.sum() + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        return float(prec), float(rec), float(f1)

    best = None
    best_score = -1.0
    for m in candidates:
        bd = boundary_from_mask(m)
        _, _, f1 = boundary_f1(edge_ref, bd, tol_pix=tol)
        if f1 > best_score:
            best_score = f1
            best = m
    return best, best_score


def mask_from_kmeans(img_rgb: np.ndarray, segmented_rgb: np.ndarray) -> np.ndarray:
    """
    把 K-means 的彩色聚类结果转成单一目标 binary mask.

    策略：
      - 对分割后的彩色图 segmented_rgb 计算每个 cluster 的平均亮度
      - 认为“最暗的那个 cluster”是背景
      - 其它所有 cluster 都视为前景，并集成一个前景掩膜
    """
    # segmented_rgb 是 KMeans_Clustering 的输出（每一类一个固定颜色）
    h, w, _ = segmented_rgb.shape
    seg = segmented_rgb.reshape(-1, 3)

    # 找到所有不同颜色（对应各个 cluster）
    colors, inv = np.unique(seg, axis=0, return_inverse=True)  # colors[K,3], inv[N]

    # 计算每个 cluster 的平均亮度
    # 用简单的灰度：Y = 0.299 R + 0.587 G + 0.114 B
    brightness = (0.299 * colors[:, 0] +
                  0.587 * colors[:, 1] +
                  0.114 * colors[:, 2])

    # 认为最暗的那个 cluster 是背景
    bg_idx = int(np.argmin(brightness))

    # 前景 = 非背景的所有 cluster
    inv_2d = inv.reshape(h, w)
    fg01 = (inv_2d != bg_idx).astype(np.uint8)   # 0/1

    # 可选的小清理：把特别小的孤立块去掉（避免噪声）——先忽略也可以
    # 这里先不做形态学，保证你能清楚看到 kmeans 的完整前景

    # 返回 0/255 掩膜
    return (fg01 * 255).astype(np.uint8)


def mask_from_watershed(markers: np.ndarray) -> np.ndarray:
    """
    WatershedSegmentation.apply 返回 markers:
      -1: 分水岭线
       1: 背景
      >1: 前景
    把所有前景合并为一个目标。
    """
    return ((markers > 1).astype(np.uint8) * 255)


# ============================
# BSDS 风格边界评测（单一 GT 掩膜）
# ============================

def _fmeasure(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    denom = P + R + 1e-9
    F = 2 * P * R / denom
    return F


def _boundary_pr_bsds(
    pb: np.ndarray,
    gt_mask: np.ndarray,
    nthresh: int = 30,
    max_dist_frac: float = 0.0075,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    现在改为“区域版” P/R/F:
      - 不再提取边界，也不再用 distanceTransform
      - 直接在整体前景区域上计算：
            TP = pred ∧ GT
            FP = pred ∧ ¬GT
            FN = ¬pred ∧ GT
        P = TP / (TP + FP)
        R = TP / (TP + FN)

    pb: 预测的前景“概率图”，范围 [0,1] 或 [0,255]
    gt_mask: GT 掩膜（0/255 或 0/1）
    """
    # 归一化为 [0,1]
    pb = pb.astype(np.float32)
    if pb.max() > 1.0:
        pb = pb / 255.0

    # GT 前景 (0/1)
    gt_fg = (ensure_binary_mask(gt_mask) > 0).astype(np.uint8)

    # 保持和原来一样的阈值扫描风格
    if nthresh <= 1:
        thresh = np.array([0.5], dtype=np.float32)
    else:
        thresh = np.linspace(
            1.0 / (nthresh + 1),
            1.0 - 1.0 / (nthresh + 1),
            nthresh,
        ).astype(np.float32)

    R_list, P_list, F_list = [], [], []

    for t in thresh:
        # 按阈值得到预测前景 (0/1)
        pred_fg = (pb >= t).astype(np.uint8)

        # 极端情况和原来保持一致
        if pred_fg.sum() == 0 and gt_fg.sum() == 0:
            R_list.append(1.0)
            P_list.append(1.0)
            F_list.append(1.0)
            continue
        if pred_fg.sum() == 0 or gt_fg.sum() == 0:
            R_list.append(0.0)
            P_list.append(0.0)
            F_list.append(0.0)
            continue

        # ========= 区域版 TP / FP / FN =========
        pred_bool = pred_fg > 0
        gt_bool = gt_fg > 0

        TP = np.logical_and(pred_bool, gt_bool).sum()
        FP = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
        FN = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()

        P = TP / (TP + FP + 1e-9)
        R = TP / (TP + FN + 1e-9)
        F = 2 * P * R / (P + R + 1e-9)

        R_list.append(float(R))
        P_list.append(float(P))
        F_list.append(float(F))

    return (
        thresh,
        np.array(R_list, dtype=np.float32),
        np.array(P_list, dtype=np.float32),
        np.array(F_list, dtype=np.float32),
    )

def _maxF_from_pr(thresh, R, P, F):
    idx = int(np.argmax(F))
    return float(thresh[idx]), float(R[idx]), float(P[idx]), float(F[idx])


def evaluate_methods_bsds(
    img_rgb: np.ndarray,
    outputs_dict: Dict[str, np.ndarray],
    gt_mask: np.ndarray,
    nthresh: int = 30,
    max_dist_frac: float = 0.0075,
):
    """
    使用 BSDS 边界评测，对四种方法进行比较。

    outputs_dict:
      {
        "contour": contour_mask_or_pb,
        "otsu":    otsu_mask,
        "kmeans":  segmented_rgb,
        "watershed": markers
      }

    gt_mask: GT 掩膜 (0/1 或 0/255)，前景=目标区域。
    返回:
      scores: {method: {Fmax, P, R, T}}
      pr_curves: {method: np.ndarray [N,4] -> [thresh, R, P, F]}
      masks: 实际用于评测的二值掩膜（可选可视化）
    """
    gt_mask_bin = ensure_binary_mask(gt_mask)

    masks: Dict[str, np.ndarray] = {}
    pbs: Dict[str, np.ndarray] = {}

    # 1) 先把四种方法统一成二值掩膜
    masks["contour"] = mask_from_contour_method(outputs_dict["contour"])
    masks["otsu"] = mask_from_otsu(outputs_dict["otsu"])
    masks["kmeans"] = mask_from_kmeans(img_rgb, outputs_dict["kmeans"])
    masks["watershed"] = mask_from_watershed(outputs_dict["watershed"])

    # 2) 掩膜 -> 前景 “概率图”(pb)
    #    这里直接用掩膜/255 作为 pb（0 或 1）。
    for name, m in masks.items():
        pbs[name] = ensure_binary_mask(m).astype(np.float32) / 255.0
        # 如果某个方法本身输出的是概率图（而不是掩膜），
        # 你可以单独对它跳过 ensure_binary_mask，直接赋值给 pbs[name]。

    scores: Dict[str, Dict[str, float]] = {}
    pr_curves: Dict[str, np.ndarray] = {}

    # 3) 对每个方法做 BSDS 风格评测
    for name, pb in pbs.items():
        thresh, R, P, F = _boundary_pr_bsds(
            pb,
            gt_mask_bin,
            nthresh=nthresh,
            max_dist_frac=max_dist_frac,
        )
        T, bestR, bestP, bestF = _maxF_from_pr(thresh, R, P, F)
        scores[name] = {
            "Fmax": bestF,
            "P": bestP,
            "R": bestR,
            "T": T,
        }
        pr_curves[name] = np.stack([thresh, R, P, F], axis=1)

    return masks, scores, pr_curves
