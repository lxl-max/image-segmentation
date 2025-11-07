import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Recording time
def ms(s):
    return f"{s*1000:.1f} ms"

# Read image
sample_image = cv2.imread('input the path of the figure')
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

# Select ROI
rects = cv2.selectROIs("select", img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

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
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
t0 = time.perf_counter()
_, thresh = cv2.threshold(gray_img, np.mean(gray_img), 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
h, w = gray_img.shape
mask = np.zeros((h, w), np.uint8)
masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
t_seg_full = time.perf_counter() - t0

fig, ax = plt.subplots()
plt.imshow(masked, cmap = 'gray')
plt.title(f"Full image with Contour Detection")
fig.text(0.5, 0.02, f"Execution times: {ms(t_seg_full)}",
         ha='center', va='bottom')
plt.axis('off')
plt.show()

# Show ROIs
for i in range(len(rois)):
    plt.imshow(rois[i])
    plt.title(f"ROI {i+1}")
    plt.axis('off')
    plt.show()

# Show segment ROIs
for i in range(len(rois)):

    gray_roi = cv2.cvtColor(rois[i], cv2.COLOR_RGB2GRAY)
    t0 = time.perf_counter()
    _, thresh = cv2.threshold(gray_roi, np.mean(gray_roi), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    h, w = gray_roi.shape
    mask = np.zeros((h, w), np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
    t_seg_full = time.perf_counter() - t0

    fig, ax = plt.subplots()
    plt.imshow(masked,cmap = 'gray')
    plt.title(f"ROI {i+1} with Contour Detection")
    fig.text(0.5, 0.02, f"Execution times: {ms(t_seg_full)}",
             ha='center', va='bottom')
    plt.axis('off')
    plt.show()
