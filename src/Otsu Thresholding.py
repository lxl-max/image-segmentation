import time
import matplotlib.pyplot as plt
import cv2

#Recording time
def ms(s):
    return f"{s*1000:.1f} ms"

# Read image
sample_image = cv2.imread('input the path of the figure')
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

plt.axis('off')
plt.imshow(img)

# Select ROI
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
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
t_seg_full = time.perf_counter() - t0

fig, ax = plt.subplots()
plt.imshow(thresh, cmap = 'gray')
plt.title(f"Full image with Otsu Thresholding")
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

    t0 = time.perf_counter()
    gray_roi = cv2.cvtColor(rois[i], cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_seg_full = time.perf_counter() - t0

    fig, ax = plt.subplots()
    plt.imshow(thresh, cmap='gray')
    plt.title(f"ROI {i+1} with Otsu Thresholding")
    fig.text(0.5, 0.02, f"Execution times: {ms(t_seg_full)}",
             ha='center', va='bottom')
    plt.axis('off')
    plt.show()
