# Classical Segmentation Algorithms
The project includes several classical segmentation methods, such as:

- Otsu Thresholding
- K-Means–based Segmentation
- Contour Detection
- Watershed Segmentation

---

## 1. Otsu Thresholding

Informal idea:

> The algorithm looks at all pixel intensities and tries to find one “best” brightness threshold that separates foreground and background.

- Pixels **brighter** than the threshold → foreground
- Pixels **darker** than the threshold → background

**Strengths**

- Very simple and fast
- Works well when the image histogram has two clear peaks  
  (e.g. dark background + bright object)

**Weaknesses**

- Struggles when:
  - foreground and background intensities overlap a lot
  - lighting is very uneven across the image

**Good for**

- Simple black-and-white objects
- Images with clear contrast (e.g. documents, certain biological images)

---

## 2. K-Means Segmentation

Intuition:

> Treat each pixel as a point (e.g. in color space), and automatically group these points into **K clusters**. Each cluster corresponds to a region in the image.

You choose the number of clusters **K** (e.g. 2, 3, 4…).

**Strengths**

- Can handle **multiple regions**, not just foreground vs background
- Works well for color images where different regions have different colors

**Weaknesses**

- You must choose K manually
- Sensitive to initialization and noise
- Does not directly use spatial information (each pixel is clustered independently)

**Good for**

- Color images with a few dominant regions
- Quick, unsupervised segmentation for exploration

---

## 3. Contour Detection

High-level idea:

> First detect “edges” where the image intensity changes quickly,  
> then connect these edges into contours.

Often built on edge detectors (like Canny) plus contour finding.

**Strengths**

- Focuses on **object boundaries**
- Useful when you care about shapes more than filled regions

**Weaknesses**

- Very sensitive to noise
- May produce many small or broken contours if the image is complex

**Good for**

- Shape analysis
- Measuring object boundaries, perimeters, etc.

---

## 4. Watershed Segmentation

Mental model:

> Imagine the grayscale image as a landscape:  
> bright areas are “mountains”, dark areas are “valleys”.  
> If you slowly “flood” this landscape with water, the lines where water from different valleys meet become the segmentation boundaries.

**Strengths**

- Good at separating **touching or overlapping objects**  
  (e.g. clustered cells or particles)
- Can produce detailed boundaries

**Weaknesses**

- Very sensitive to noise and over-segmentation
- Usually needs good preprocessing (smoothing, markers, etc.)

**Good for**

- Microscopy images with touching objects
- Cases where simple thresholding merges objects together

---

