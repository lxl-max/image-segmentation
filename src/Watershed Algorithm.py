import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

def ms(s):
    return f"{s*1000:.1f} ms"

# Read image
sample_image = cv2.imread('input the path of the figure')
img=cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# Select ROIs
rects = cv2.selectROIs("select", img, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("select")
rois = []
for i, (x, y, w, h) in enumerate(rects):
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        rois.append(roi)

# Show initial image
plt.imshow(img)
plt.title("Initial Image")
plt.axis("off")
plt.show()

# Show segment image
t0 = time.perf_counter()
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations = 3)
# sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv2.watershed(sample_image,markers)
sample_image[markers == -1] = [255,0,0]
t_seg_full = time.perf_counter() - t0
plt.imshow(sample_image)
plt.title("Image with Markers")
plt.axis("off")
plt.show()
fig, ax = plt.subplots()
plt.imshow(markers)
plt.title("Image with Watershed Algorithm")
fig.text(0.5, 0.02, f"Execution times: {ms(t_seg_full)}",
         ha='center', va='bottom')
plt.axis("off")
plt.show()

# Show ROIs
for i in range(len(rois)):
    plt.imshow(rois[i])
    plt.title(f"ROI {i+1}")
    plt.axis('off')
    plt.show()

# Show segment ROIs
for i in range(len(rois)):

    t0 = time.perf_counter()
    gray_roi = cv2.cvtColor(rois[i], cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(rois[i], markers)
    rois[i][markers == -1] = [255, 0, 0]
    t_seg_full = time.perf_counter() - t0
    plt.imshow(rois[i])
    plt.title(f"ROI {i+1} with Markers")
    plt.axis("off")
    plt.show()
    fig, ax = plt.subplots()
    plt.imshow(markers)
    plt.title(f"ROI {i+1} with Watershed Algorithm")
    fig.text(0.5, 0.02, f"Execution times: {ms(t_seg_full)}",
             ha='center', va='bottom')
    plt.axis("off")
    plt.show()
