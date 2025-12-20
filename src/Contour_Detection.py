import time
from typing import List, Tuple

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

# Contour detection segmentation.
class ContourDetection(object):
    """：
    1) Convert to greyscale
    2) Apply thresholding using mean(greyscale) with THRESH_BINARY_INV
    3) Apply Canny edge detection followed by dilation
    4) Identify the largest contour
    5) Fill the contour onto the empty mask
    """

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), dtype=uint8.

        Returns
        -------
        masked : np.ndarray
            Single-channel mask image, with contour regions set to 255 and all others to 0.
        elapsed : float
            Execution time in seconds.
        """
        # Convert to greyscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Thresholding and edge detection
        t0 = time.perf_counter()
        _, thresh = cv2.threshold(
            gray_img, np.mean(gray_img), 255, cv2.THRESH_BINARY_INV
        )
        edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

        # findContours
        contours = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[-2]

        # Select the largest area outline
        cnt = sorted(contours, key=cv2.contourArea)[-1]

        h, w = gray_img.shape
        mask = np.zeros((h, w), np.uint8)
        masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
        elapsed = time.perf_counter() - t0

        return masked, elapsed

# Load an image, interactively select ROIs, apply Contour Detection segmentation to the full image and to each ROI, and display the results.
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

    # Create contour segmenter instance
    segmenter = ContourDetection()

    # Full image segmentation
    masked_full, t_seg_full = segmenter.apply(img_rgb)
    show_image(
        masked_full,
        title="Full image with Contour Detection",
        footer=f"Execution time: {ms(t_seg_full)}",
    )

    # Show original ROIs
    for i, roi in enumerate(rois, start=1):
        show_image(roi, title=f"ROI {i}")

    # Segment each ROI and show
    for i, roi in enumerate(rois, start=1):
        masked_roi, t_roi = segmenter.apply(roi)
        show_image(
            masked_roi,
            title=f"ROI {i} with Contour Detection",
            footer=f"Execution time: {ms(t_roi)}",
        )

if __name__ == "__main__":

    # Input the path of the figure here
    image_path = r"input the image path"
    process_image(image_path)
