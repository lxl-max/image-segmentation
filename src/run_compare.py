import cv2
from contour_similarity_eval2 import evaluate_methods_bsds

from Contour_Detection import ContourDetection
from KMeans_Clustering import KMeansClustering
from Otsu_Thresholding import OtsuThresholding
from Watershed_Segmentation_with_markers import WatershedSegmentation


def main(image_path: str, gt_path: str):
    # Read the original image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Read GT mask (either 0/1 or 0/255 is acceptable)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(gt_path)

    # 1) Execute four partitioning methods to obtain the original output.
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
        "watershed": markers,
    }

    # 2) Calculate recall
    masks, scores = evaluate_methods_bsds(
        img_rgb,
        outputs,
        gt_mask=gt,
        nthresh=30,
        max_dist_frac=0.0075,
    )

    # 3) Print result
    for name, s in scores.items():
        print(
            f"{name:10s}  Rmax={s['Rmax']:.4f}  T={s['T']:.3f}"
        )

    # Display segmentation results + GT mask
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    methods = ["contour", "kmeans", "otsu", "watershed"]
    for i, m in enumerate(methods, 1):
        plt.subplot(2, 3, i)
        plt.imshow(masks[m], cmap="gray")
        plt.title(f"{m}")
        plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(gt, cmap="gray")
    plt.title("GT mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simply replace the following two lines with your own paths.
    main(
        r"input the image path",
        r"input the GT path"
    )
