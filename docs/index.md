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

## Where should I start?

Recommended reading order:

1.  [Quick Start](quickstart.md)  
   Get the environment ready and run the first example in a few minutes.

2.  [UI & Controls](ui-guide.md)  
   Learn how to select ROIs, confirm them, and exit the program.

3.  [Algorithms](algorithms.md)  
   Intuition only: what each method is roughly doing and when it works well.

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

