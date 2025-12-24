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

<img width="1827" height="645" alt="image" src="https://github.com/user-attachments/assets/7ea4e8ad-d86c-4c9f-8819-26751e5409de" />
<img width="1822" height="521" alt="image" src="https://github.com/user-attachments/assets/8d55f693-82ed-446d-a239-4be8dfc90548" />
<img width="1838" height="490" alt="image" src="https://github.com/user-attachments/assets/8da1c7ae-75b6-4fad-9efc-d44c4534afbf" />



## References
[OpenCV](https://opencv.org/) and [Examples](https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/)




