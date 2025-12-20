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

# K-Means color clustering segmentation.
class KMeansClustering(object):
    """
    Example
    -------
    segmenter = KMeansSegmentation(k=3, attempts=10)
    seg_img, elapsed = segmenter.apply(image)

    The input is assumed to be an RGB image (H, W, 3) as a NumPy array.
    """

    def __init__(
        self,
        k: int = 3,
        attempts: int = 10,
        max_iter: int = 10,
        epsilon: float = 1.0,
    ):
        """
        Parameters
        ----------
        k : int
            Number of clusters.
        attempts : int
            Number of times K-Means is executed using different initial labellings.
        max_iter : int
            Maximum number of iterations for K-Means.
        epsilon : float
            Required accuracy.
        """
        self.k = k
        self.attempts = attempts
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            max_iter,
            epsilon,
        )

    # Apply K-Means segmentation to an image.
    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), dtype=uint8.

        Returns
        -------
        segmented : np.ndarray
            Segmented RGB image (each pixel replaced by its cluster center color).
        elapsed : float
            Execution time in seconds.
        """
        # Reshape to 2D
        two_d_image = image.reshape((-1, 3))
        two_d_image = np.float32(two_d_image)

        # Same criteria and parameters pattern
        t0 = time.perf_counter()
        ret, label, center = cv2.kmeans(two_d_image,
            self.k,
            None,
            self.criteria,
            self.attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        elapsed = time.perf_counter() - t0

        center = np.uint8(center)
        res = center[label.flatten()]
        segmented = res.reshape(image.shape)

        return segmented, elapsed

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

    # Create segmenter instance
    segmenter = KMeansClustering(k=3, attempts=10, max_iter=10, epsilon=1.0)

    # Full image segmentation
    seg_img, t_seg_full = segmenter.apply(img_rgb)
    show_image(
        seg_img,
        title="Image with K-Means Clustering",
        footer=f"Execution time: {ms(t_seg_full)}",
    )

    # Show original ROIs
    for i, roi in enumerate(rois, start=1):
        show_image(roi, title=f"ROI {i}")

    # Segment each ROI and show
    for i, roi in enumerate(rois, start=1):
        seg_roi, t_roi = segmenter.apply(roi)
        show_image(
            seg_roi,
            title=f"ROI {i} with K-Means Clustering",
            footer=f"Execution time: {ms(t_roi)}",
        )

if __name__ == "__main__":

    # Input the path of the figure here
    image_path = r"input the image path"
    process_image(image_path)