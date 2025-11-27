from typing import List, Tuple

import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Recording time: format seconds as milliseconds string
def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"

# Use OpenCV to interactively select multiple ROIs.
def select_rois(
    image: np.ndarray,
    window_name: str = "select"
) -> List[np.ndarray]:
    """
    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3).
    window_name : str
        Window name for ROI selection.

    Returns
    -------
    rois : list of np.ndarray
        Selected ROI images (RGB).
    """
    rects = cv2.selectROIs(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    rois: List[np.ndarray] = []

    for (x, y, w, h) in rects:
        if w > 0 and h > 0:
            roi = image[y:y + h, x:x + w]
            rois.append(roi)

    return rois

# Display an image with Matplotlib and optionally show text at the bottom.
def show_image(
    img: np.ndarray,
    title: str = "",
    footer: str | None = None
) -> None:

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

    if footer is not None:
        fig.text(
            0.5,
            0.02,
            footer,
            ha="center",
            va="bottom"
        )

    plt.show()

# Otsu thresholding segmentation.
class OtsuSegmentation(object):
    """
    Apply Otsu thresholding to an image (grayscale).
    """

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Parameters
        ----------
        image : np.ndarray
            Grayscale image.

        Returns
        -------
        thresh : np.ndarray
            Binary image after Otsu thresholding.
        elapsed : float
            Execution time in seconds (thresholding only).
        """

        # Ensure it is a greyscale image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        t0 = time.perf_counter()
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        elapsed = time.perf_counter() - t0

        return thresh, elapsed

# Load an image, interactively select ROIs, apply K-Means segmentation to the full image and to each ROI, and display the results.
def process_image(image_path: str) -> None:

    # Read image (BGR) and convert to RGB
    sample_image = cv2.imread(image_path)
    if sample_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    # Select ROIs
    rois = select_rois(img_rgb, window_name="select")

    # Show original image
    show_image(img_rgb, title="Initial Image")

    # Create Otsu segmenter instance
    segmenter = OtsuSegmentation()

    # Full image segmentation
    full_thresh, t_seg_full = segmenter.apply(img_rgb)
    show_image(
        full_thresh,
        title="Full image with Otsu Thresholding",
        footer=f"Execution times: {ms(t_seg_full)}",
    )

    # Show original ROIs
    for i, roi in enumerate(rois, start=1):
        show_image(roi, title=f"ROI {i}")

    # Segment each ROI and show
    for i, roi in enumerate(rois, start=1):
        roi_thresh, t_roi = segmenter.apply(roi)
        show_image(
            roi_thresh,
            title=f"ROI {i} with Otsu Thresholding",
            footer=f"Execution times: {ms(t_roi)}",
        )

if __name__ == "__main__":

    # Input the path of the figure here
    image_path = r"input the image path"
    process_image(image_path)
