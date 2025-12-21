# Classical Image Segmentation

This repository contains a collection of image segmentation algorithms implemented in Python using OpenCV, and NumPy.


## What the project does
Implement four classical segmentation algorithums: Otsu Thresholding, K-Means Clustering, Contour Detection, and Watershed Segmentation. 
Using these algorithums, user can select a Region of Interest (ROI) or multiple ROIs of a sample image by drawing a rectangle with the mouse on the image. 

Users may choose to run a single segmentation algorithm, or utilise [`run_segmentations`](https://github.com/lxl-max/image-segmentation/blob/main/src/run_segmentations.py) to execute all segmentation algorithms at once.

## Segmentation Algorithms

The following segmentation algorithms are included:

1. **Otsu Thresholding**: Implemented using OpenCV.
2. **K-means Clustering**: Implemented using OpenCV.
3. **Contour Detection**: Implemented using OpenCV.
4. **Watershed Segmentation**: Implemented using OpenCV.

Each algorithm is implemented in a separate Python file for better organization.

## Usage

To use these segmentation algorithms, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/lxl-max/image-segmentation/tree/main/src
```
2. Install the required dependencies:
   
```bash
  pip install -r requirements.txt
```
3. Run, you can

4. Choose ROIs, enter "space" to confirm, enter "Esc" to exit

5. Get results

## Example screenshoots or figures 
Using K-means clustering with k=3 as an example: 

<img width="270" height="534" alt="image" src="https://github.com/user-attachments/assets/dbaa2be1-a462-4dea-85ee-0c93eb5c3e79" />
<img width="342" height="554" alt="image" src="https://github.com/user-attachments/assets/942e70c9-3020-4cc1-8829-f03d465aabd7" />
<img width="476" height="509" alt="image" src="https://github.com/user-attachments/assets/5d370978-32c2-41db-ae3d-b8886f5e92a6" />
<img width="470" height="555" alt="image" src="https://github.com/user-attachments/assets/aaa78ed2-b152-40bf-9798-6b98d4c10d3a" />
<img width="413" height="504" alt="image" src="https://github.com/user-attachments/assets/e5b42d5d-6fc4-43e4-a893-6a59ab79029b" />
<img width="405" height="554" alt="image" src="https://github.com/user-attachments/assets/e514e616-bf14-46a5-9242-7224e3aa77e4" />


## References
[OpenCV](https://opencv.org/) and [Examples](https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/)




