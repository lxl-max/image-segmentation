import cv2

from contour_similarity_eval import evaluate_methods

from Contour_Detection import ContourDetection
from KMeans_Clustering import KMeansClustering
from Otsu_Thresholding import OtsuThresholding
from Watershed_Segmentation_with_markers import WatershedSegmentation

def main(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1) Run four methods to obtain the raw outputs
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
        "watershed": markers,   # test markers (not the image with the red border)
    }

    # 2) Unified mask + contour similarity score
    masks, scores = evaluate_methods(img_rgb, outputs, tol=2)

    # 3) Printed output
    for k, v in scores.items():
        print(f"{k:10s} BF-F1={v['bf_f1']:.4f}  P={v['bf_precision']:.4f}  R={v['bf_recall']:.4f}")

if __name__ == "__main__":
    main(r"input the image path")


