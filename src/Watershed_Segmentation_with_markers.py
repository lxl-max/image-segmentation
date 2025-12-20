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

# Watershed segmentation.
class WatershedSegmentation(object):
    """
    Apply watershed segmentation with foreground/background segmentation, morphological operations and distance transformation
    """

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), dtype=uint8.

        Returns
        -------
        segmented : np.ndarray
            Image with watershed boundaries marked in red.
        markers : np.ndarray
            Label image from watershed.
        elapsed : float
            Execution time in seconds.
        """
        # Convert to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        t0 = time.perf_counter()

        # OTSU Binarization
        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform,
            0.7 * dist_transform.max(),
            255,
            0
        )

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Watershed
        markers = cv2.watershed(image, markers)

        # Mark the boundaries on the original image
        segmented = image.copy()
        segmented[markers == -1] = [255, 0, 0]

        elapsed = time.perf_counter() - t0

        return segmented, markers, elapsed

# Load an image, interactively select ROIs, apply watershed segmentation to the full image and to each ROI, and display the results.
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

    # Create segmenter instance
    segmenter = WatershedSegmentation()

    # Full image segmentation
    seg_full, markers_full, t_seg_full = segmenter.apply(img_rgb)
    show_image(seg_full, title="Image with Markers")
    show_image(
        markers_full,
        title="Image with Watershed Segmentation",
        footer=f"Execution time: {ms(t_seg_full)}"
    )

    # Show original ROIs
    for i, roi in enumerate(rois, start=1):
        show_image(roi, title=f"ROI {i}")

    # Segment each ROI and show
    for i, roi in enumerate(rois, start=1):
        seg_roi, markers_roi, t_roi = segmenter.apply(roi)
        show_image(seg_roi, title=f"ROI {i} with Markers")
        show_image(
            markers_roi,
            title=f"ROI {i} with Watershed Segmentation",
            footer=f"Execution time: {ms(t_roi)}"
        )

if __name__ == "__main__":

    # Input the path of the figure here
    image_path = r"input the path of the figure"
    process_image(image_path)
