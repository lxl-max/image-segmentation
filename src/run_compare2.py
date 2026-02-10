import cv2
from contour_similarity_eval2 import evaluate_methods_bsds

from Contour_Detection import ContourDetection
from KMeans_Clustering import KMeansClustering
from Otsu_Thresholding import OtsuThresholding
from Watershed_Segmentation_with_markers import WatershedSegmentation


def main(image_path: str, gt_path: str):
    # 读原图
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 读 GT 掩膜（0/1 或 0/255 都可以）
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(gt_path)

    # 1) 跑四种分割方法，得到原始输出
    contour_seg = ContourDetection()
    kmeans_seg = KMeansClustering(k=3, attempts=10, max_iter=10, epsilon=1.0)
    otsu_seg = OtsuThresholding()
    watershed_seg = WatershedSegmentation()

    contour_mask, _ = contour_seg.apply(img_rgb)
    kmeans_img, _ = kmeans_seg.apply(img_rgb)
    otsu_mask, _ = otsu_seg.apply(img_rgb)
    watershed_img, markers, _ = watershed_seg.apply(img_rgb)

    outputs = {
        "contour": contour_mask,
        "kmeans": kmeans_img,
        "otsu": otsu_mask,
        "watershed": markers,   # 注意：这里传 markers，而不是带红边的显示图
    }

    # 2) 用 BSDS 边界评测
    masks, scores, pr_curves = evaluate_methods_bsds(
        img_rgb,
        outputs,
        gt_mask=gt,
        nthresh=30,
        max_dist_frac=0.0075,
    )

    # 3) 打印结果（和你之前看到的类似）
    for name, s in scores.items():
        print(
            f"{name:10s} BSDS-Fmax={s['Fmax']:.4f}  "
            f"P={s['P']:.4f}  R={s['R']:.4f}  T={s['T']:.3f}"
        )

    # 如果你之后想画 PR 曲线，可以在这里用 pr_curves 画图，
    # 例如：每个方法一条曲线，横轴 R，纵轴 P。
    # ==== 显示分割结果 + GT 掩膜 ====
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    methods = ["contour", "kmeans", "otsu", "watershed"]
    for i, m in enumerate(methods, 1):
        plt.subplot(2, 3, i)
        plt.imshow(masks[m], cmap="gray")
        plt.title(f"{m} ")
        plt.axis("off")

    # 显示 GT 掩膜
    plt.subplot(2, 3, 5)
    plt.imshow(gt, cmap="gray")
    plt.title("GT mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ==== 画 PR 曲线（类似你贴的 BSDS 图） ====
    def plot_iso_f():
        import numpy as np
        R = np.linspace(0.001, 1, 200)
        F_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for F in F_list:
            P = (F * R) / (2 * R - F + 1e-9)
            P[(2 * R - F) <= 0] = np.nan
            plt.plot(R, P, "g--", linewidth=0.5)

    plt.figure(figsize=(8, 6))
    plot_iso_f()

    for m in methods:
        arr = pr_curves[m]   # shape (N,4) -> [thresh, R, P, F]
        R = arr[:, 1]
        P = arr[:, 2]
        F = arr[:, 3]
        Fmax = F.max()
        plt.plot(R, P, label=f"{m} (F={Fmax:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Boundary Detection PR Curves (BSDS-style)")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # 把下面两行改成你自己的路径即可
    main(
        r"C:\dragonfly\datasets\sample.png",
        r"C:\dragonfly\datasets\GT.png"
    )
