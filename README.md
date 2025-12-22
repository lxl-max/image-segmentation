# Classical Image Segmentation

This repository contains a collection of image segmentation algorithms implemented in Python using OpenCV, and NumPy.


## What the project does
Implement four classical segmentation algorithums: [`Otsu Thresholding`](https://github.com/lxl-max/image-segmentation/blob/main/src/Otsu_Thresholding.py), [`K-Means Clustering`](https://github.com/lxl-max/image-segmentation/blob/main/src/KMeans_Clustering.py), [`Contour Detection`](https://github.com/lxl-max/image-segmentation/blob/main/src/Otsu_Thresholding.py), and [`Watershed Segmentation`](https://github.com/lxl-max/image-segmentation/blob/main/src/Watershed_Segmentation_with_markers.py). 
Using these algorithums, user can select a Region of Interest (ROI) or multiple ROIs of a sample image by drawing a rectangle with the mouse on the image. 

User may choose to run a single segmentation algorithm, or utilise [`run_segmentations`](https://github.com/lxl-max/image-segmentation/blob/main/src/run_segmentations.py) to execute all segmentation algorithms at once. 

User also can operatre [`run_compare`](https://github.com/lxl-max/image-segmentation/blob/main/src/run_compare.py) to evaluate the segmentation capabilities of different segmentation methods. 

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
3. Run

4. Choose ROIs, enter "space" to confirm, enter "Esc" to exit

5. Get results

## Results 

<img width="1538" height="546" alt="1" src="https://github.com/user-attachments/assets/435f6504-0bcf-4e96-bb23-5df50fe42490" />
<img width="1823" height="471" alt="2" src="https://github.com/user-attachments/assets/0f32221b-b417-46d8-85f5-e0c668baa831" />
<img width="1838" height="467" alt="3" src="https://github.com/user-attachments/assets/eaae0040-dba9-44f0-ac2a-28ee327f5415" />


## References
[OpenCV](https://opencv.org/) and [Examples](https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/)




