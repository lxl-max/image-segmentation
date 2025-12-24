import cv2

from Contour_Detection import ContourDetection
from KMeans_Clustering import KMeansClustering
from Otsu_Thresholding import OtsuThresholding
from Watershed_Segmentation_with_markers import WatershedSegmentation

from Contour_Detection import ms, select_rois, show_image

# main runner
def run_all(image_path: str) -> None:
    # Read image (BGR) and convert to RGB
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Select ROIs
    rois = select_rois(img_rgb, window_name="Select ROI(s) Once")

    # Show original image
    show_image(img_rgb, title="Original Image")

    # Initialise four methods
    contour = ContourDetection()
    kmeans = KMeansClustering(k=3, attempts=10, max_iter=10, epsilon=1.0)
    otsu = OtsuThresholding()
    watershed = WatershedSegmentation()

    # Whole-image processing
    contour_mask, t1 = contour.apply(img_rgb)
    show_image(contour_mask, title="Full - ContourDetection", footer=f"time: {ms(t1)}")

    kmeans_img, t2 = kmeans.apply(img_rgb)
    show_image(kmeans_img, title="Full - KMeansClustering", footer=f"time: {ms(t2)}")

    otsu_mask, t3 = otsu.apply(img_rgb)
    show_image(otsu_mask, title="Full - OtsuThresholding", footer=f"time: {ms(t3)}")

    ws_img, ws_markers, t4 = watershed.apply(img_rgb)
    show_image(ws_img, title="Full - Watershed (boundaries in red)", footer=f"time: {ms(t4)}")
    show_image(ws_markers, title="Full - Watershed Markers")

    # Each ROI processing
    for i, roi in enumerate(rois, start=1):
        show_image(roi, title=f"ROI {i} - Original")

        contour_mask, t1 = contour.apply(roi)
        show_image(contour_mask, title=f"ROI {i} - ContourDetection", footer=f"time: {ms(t1)}")

        kmeans_img, t2 = kmeans.apply(roi)
        show_image(kmeans_img, title=f"ROI {i} - KMeansClustering", footer=f"time: {ms(t2)}")

        otsu_mask, t3 = otsu.apply(roi)
        show_image(otsu_mask, title=f"ROI {i} - OtsuThresholding", footer=f"time: {ms(t3)}")

        ws_img, ws_markers, t4 = watershed.apply(roi)
        show_image(ws_img, title=f"ROI {i} - Watershed (boundaries in red)", footer=f"time: {ms(t4)}")
        show_image(ws_markers, title=f"ROI {i} - Watershed Markers")

if __name__ == "__main__":
    # Input the path of the figure here
    image_path = r"input the image path"
    run_all(image_path)