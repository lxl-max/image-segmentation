# Image Segmentation

Welcome to the **ROI Selection and Classical Image Segmentation** project 

This project is a small, interactive demo for classic image segmentation methods.

In short, it lets you:

> Load an image → draw one or more Regions of Interest (ROIs) with the mouse →  
> apply several classical segmentation algorithms → visually compare their results.

### Who is this for?

1. Students who are starting to learn about image segmentation
2. Anyone who wants to **see what classical methods actually do** on real images
3. People who prefer a **simple, runnable demo** rather than only reading theory

---

## What’s in this repository?

1. `data/` – Example images (you can replace or add your own)
2. `src/` – Main source code
  - ROI selection
  - Four classical image segmentation algorithms
  - Visualization and comparison of results
3. `docs/` – This documentation

If you just want to **play with the demo**, you only need to:

1. Install the Python dependencies  
2. Run the entry script  
3. Use your mouse to draw a ROI, press **Space** to confirm, **Esc** to exit  

---

## How do these algorithms work?

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

## Advanced about these algorithms

  ### 1. Otsu Thresholding

The goal is to maximise between-class variance. 

$$
\sigma_B^2 = P_1 (m_1 - m_G)^2 + P_2 (m_2 - m_G)^2
$$

where $\sigma_B^2$ is the between-class variance, $m_1 $, $m_2$, $m_G$ represent the mean of class 1, the mean of class 2, and the global mean respectively, $P_1$, $P_2$ represent the probabilities of class 1 and class2 occurring respectively. 

$$
m_1(k) = \frac{1}{P_1(k)} \sum_{i=0}^{k} i \cdot p_i,\quad
m_2(k) = \frac{1}{P_2(k)} \sum_{i=k+1}^{255} i \cdot p_i,\quad
m_G(k) = \sum_{i=0}^{255} i \cdot p_i
$$

$$
P_1(k) = \sum_{i=0}^{k} p_i,\quad
P_2(k) = \sum_{i=k+1}^{255} p_i = 1 - P_1(k)
$$

where $p_i = \frac{n_i}{M \cdot N}$ is the proportion of pixels with intensity $i$, $n_i$ is the number of these pixels and $M* N$ represent the total number of pixels. 

Also, $k$ is the threshold between Class 1 and Class 2, which is what we wish to obtain. 

---

### 2. K-Means Clustering

K-means algorithm is an iterative procedure that successively refines the means until convergence is achieved. The criterion is: 

$$
\arg\min_{C} \left( \sum_{i=1}^{k} \sum_{z \in C_i} \lVert z - m_i \rVert^{2} \right)
$$

where $z$ is the set of vector observations, $m_i$ is the mean vector of the samples in set $C_i$, $\lVert arg \rVert$ is the vector norm of the argument, and $C$ is the cluster sets. Typically, the Euclidean norm is used, so the term $\lVert z - m_i \rVert$ is the familiar Euclidean distance from a sample in $C_i$ to mean $m_i$. This equation says we are interested in finding the sets $C$ such that the sum of the distances from each point in a set to the mean of that set is minimum. 

1. Set an initial set of means $m_i$.
2. Assign each sample to the cluster set whose mean is the closest:

$$
z \to C_i \quad \text{if} \quad \lVert z - m_i \rVert^2 < \lVert z - m_j \rVert^2
$$

3. Update the means(cluster centers): 

$$
m_i = \frac{1}{|C_i|} \sum_{z \in C_i} z
$$

where $|C_i|$ is the number of samples in cluster in set $C_i$. 

4. Compute the Euclidean norms of the differences between the mean vectors in the current and previous steps. Compute the residual error, $E$, as the sum of the $k$ norms. Stop if $E\leqslant T$ , where $T$ a specified, nonnegative threshold. Else, go back to Step 2. 

---

### 3. Contour Detection

We find contours, which are enclosed large outline, from Canndy Edge Detection. 

#### Canndy Edge Detection

1. Smoothing input image with a Gaussian filter: $f_s(x, y) = G(x, y) \star f(x, y)$, where $f_s(x, y)$ is the smoothed image, $f(x, y)$ is the input image and $G(x, y)$ is the Gaussian function: $G(x, y) = e^{-\frac{x^2 + y^2}{2\sigma^2}}$. 
2. Compute the gradient magnitude and angle images, 

$$
M_s(x, y) = \left\lVert \nabla f_s(x, y) \right\rVert
           = \sqrt{g_x^2(x, y) + g_y^2(x, y)}
$$

$$
\alpha(x, y) = \tan^{-1} \left[ \frac{g_y(x, y)}{g_x(x, y)} \right]
$$

with $g_x(x, y) = \frac{\partial f_s(x, y)}{\partial x}$ and $g_y(x, y) = \frac{\partial f_s(x, y)}{\partial y}$ and $g(x, y)$ are nonzero pixels. 


3. Apply nonmaxima suppression: if the gradient magnitude at a point is not the maximum in that direction, set it to zero.
4. Use double thresholding and connectivity analysis to detect and link edges: 

$$
T_H < g(x, y)
$$

$$
T_L < g(x, y) < T_H
$$

Pixels satisfying the first equation are called strong edge pixels and those satisfying to second equation are called weak edge pixels.   

Upon completion of Canndy detection process, we perform a **dilate** operation. This thickens the edges and makes the contours more consistent, thereby more convenient for contour detection. 

#### Find Contour

1. Locate the foreground connected components, get $C_1$, $C_2$, ..., $C_n$, each is an independent region.
2. Perform contour tracking for each $C_i$: starting from a boundary pixel, walk around the perimeter of the neighbourhood, record the pixels in sequence. The result is an ordered, connected outline.​
3. Pick the largest one. 

---

### 4. Watershed Segmentation

Watershed Segmentation is based on regional expansion and conflict resolution algorithm. It needs discrete, clean foreground and background structure. So we can use **Otsu** as the starting point. 

1. Otsu operation. 
2. Remove noise by first eroding and then dilating.
3. Sure background by dilating. 
4. Sure foreground by distancetransform, calculate at each foreground pixel x:

$$
d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
$$

where $x_1$ and $y_1$ belongs to the foreground, $x_2$ and $y_2$ belongs to the background. Identify the maximum distance from the background, retaining only pixels over 0.7 times this maximum distance as the sure foreground. 

6. Finding unknown region by subtracting the foreground from the background. 
7. Marker labeling: assign an integer label to each connected region.
8. Watershed process: begin with the label of unknown is 0, if their neighbours have multiple different kinds of labels, then set this pixel to -1. All of these pixels form the watershed line.

---

### Assessment

#### 1. Precision

$$
P = \frac{\lvert B \cap A^t \rvert}{\lvert B \rvert}
$$

$A$ is reference boundary (original Canny edge), $B$ is predicted boundary (segmentation boundary), $A^t$ is boundary band obtained by dilating A by tol and tol is a buferr error allowed. Precision represent among the boundary pixels $B$ drawn by the segmentation method, the number of pixels lie near the reference boundary. 

#### 2. Recall

$$
P = \frac{\lvert A \cap B^t \rvert}{\lvert A \rvert}
$$

Recall represent among the boundary pixels $A$ obtained by reference method, the number of pixels lie near the predicated boundary.  

#### 3. BF-F1

$$
F_1 = \frac{2PR}{P + R}
$$

BF-F1 is the harmonic mean of P and R, it used to measure the overall performance of boundary alignment.  

---




