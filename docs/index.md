# Image Segmentation

Welcome to the **ROI Selection and Classical Image Segmentation** project 

This project is a small, interactive demo for classic image segmentation methods.

In short, it lets you:

> Load an image → draw one or more Regions of Interest (ROIs) with the mouse →  
> apply several classical segmentation algorithms → visually compare their results.

### Who is this for?

- Students who are starting to learn about image segmentation
- Anyone who wants to **see what classical methods actually do** on real images
- People who prefer a **simple, runnable demo** rather than only reading theory

### Who is this *not* for?

- Large-scale production systems
- Deep learning–based segmentation (this project focuses on classical methods)


---

## What’s in this repository?

- `data/` – Example images (you can replace or add your own)
- `src/` – Main source code
  - ROI selection
  - Four classical image segmentation algorithms
  - Visualization and comparison of results
- `docs/` – This documentation

If you just want to **play with the demo**, you only need to:

1. Install the Python dependencies  
2. Run the entry script  
3. Use your mouse to draw a ROI, press **Space** to confirm, **Esc** to exit  

---

## How to understand these algorithms?

The project includes several classical segmentation methods, such as:

- Otsu Thresholding
- K-Means–based Segmentation
- Contour Detection
- Watershed Segmentation

---
### 1. Otsu Thresholding

> We extract the brightness values of every pixel in an image, then plot a histogram. and tries to find one “best” brightness threshold that separates foreground and background.

- Pixels **brighter** than the threshold → foreground
- Pixels **darker** than the threshold → background

**Strengths**

- Very simple and fast
- Works well when the sample consists of a single object and background

**Weaknesses**

- Works poor when the sample consists of two or more objects because of relying on brightness only
- Only suitable for gray scale image
- Easily affected by lighting

---

### 2. K-Means Clustering

> Treat each pixel as a point (e.g. in color space), and automatically group these points into **K clusters**. Each cluster corresponds to a region in the image.

You choose the number of clusters **K** (e.g. 2, 3, 4…). 

**Strengths**

- Can handle sample with multiple objects
- Works well for color images where different regions have different colors

**Weaknesses**

- You must choose K manually
- Only focus on a single pixel, regardless of its neighbours, may lead to some errors

---

### 3. Contour Detection

> First detect “edges” where the image intensity changes quickly, then connect these edges into contours.

Often built on edge detectors (like Canny) plus contour finding.

**Strengths**

- Focuses on **object boundaries**
- Useful when you care about shapes more than internal features of the sample

**Weaknesses**

- Very sensitive to noise
- Unable to process complex images, may produce many small or broken contours

---

### 4. Watershed Segmentation


> Imagine the grayscale image as a landscape: bright areas are “mountains”, dark areas are “valleys”.  
> If you slowly “flood” this landscape with water, the lines where water from different valleys meet become the segmentation boundaries.

**Strengths**

- Good at separating objects that stick together (e.g. cells, particles, and coins pressed together)
- Can produce detailed boundaries

**Weaknesses**

- May draw boundaries everywhere because of noise.
- Sometimes even there are no edges, it can draw a line between two sections to separate the two areas.

---




