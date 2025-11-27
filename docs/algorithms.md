# Classical Segmentation Algorithms
The project includes several classical segmentation methods, such as:

- Otsu Thresholding
- K-Means–based Segmentation
- Contour Detection
- Watershed Segmentation

---

## 1. Otsu Thresholding

> The algorithm looks at all pixel intensities and tries to find one “best” brightness threshold that separates foreground and background.

- Pixels **brighter** than the threshold → foreground
- Pixels **darker** than the threshold → background


---

## 2. K-Means Segmentation

> Treat each pixel as a point (e.g. in color space), and automatically group these points into **K clusters**. Each cluster corresponds to a region in the image.

You choose the number of clusters **K** (e.g. 2, 3, 4…).

---

## 3. Contour Detection


> First detect “edges” where the image intensity changes quickly,  
> then connect these edges into contours.

Often built on edge detectors (like Canny) plus contour finding.


---

## 4. Watershed Segmentation


> Imagine the grayscale image as a landscape:  
> bright areas are “mountains”, dark areas are “valleys”.  
> If you slowly “flood” this landscape with water, the lines where water from different valleys meet become the segmentation boundaries.


---

