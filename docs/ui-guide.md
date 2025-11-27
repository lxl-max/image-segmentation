# UI & Controls

This page explains **how to interact with the program** once it is running.

If you are not sure how to start the program, see [Quick Start](quickstart.md) first.

---

## 1. Overall interaction flow

1. Run the Python script from the terminal.
2. Type the path to an image (for example: `data/sample1.png`) and press **Enter**.
3. An image window opens (using OpenCV).
4. You **draw one or more ROIs** (rectangles) using the mouse.
5. You press **Space** to confirm your selection(s).
6. The program runs several segmentation algorithms and shows result windows.
7. You press **Esc** to close all windows and exit the program.

---

## 2. Mouse controls

These actions happen in the **image window**:

- **Left mouse button + drag**

  - Press and hold the left mouse button.
  - Drag to create a rectangular ROI.
  - Release the mouse button to finish that ROI.

- **Multiple ROIs**

  - After finishing one ROI, you can click and drag again to create another one.
  - Each rectangle marks one region of interest to be segmented.

The exact appearance of the rectangle may vary (e.g. color, thickness),
but you should see a box following your mouse while you drag.

---

## 3. Keyboard controls

All key presses below refer to the image / result windows.

- **Space**

  - Confirms all currently drawn ROIs.
  - Starts the segmentation step using the four classical algorithms.
  - After you press Space, one or more new windows will appear showing the results.

- **Esc**

  - Exits the program.
  - Closes all OpenCV windows that were opened by this script.

> Tip:  
> If you accidentally drew a wrong ROI and do not see a "reset" option in the UI,
> you can press **Esc** to quit and simply run the script again.

---

## 4. Typical windows you will see

Exact window titles depend on how the script is written, but you will generally see:

1. **Original image window**

   - Shows the original image you loaded.
   - You interact with this window to draw ROIs.

2. **Result windows for each algorithm**

   For example (names are illustrative):

   - `Otsu thresholding`
   - `K-Means`
   - `Contours`
   - `Watershed`

   Each window visualizes how that specific algorithm segmented your selected ROI.

---

## 5. Suggested way to explore

To get a good feel for the UI and algorithms:

1. Start with a **simple image** from `data/` (clear object on uniform background).
2. Draw a **tight ROI** around the main object.
3. Press **Space**, then carefully compare:
   - Which algorithm gives the cleanest object boundary?
   - Which algorithm leaves a lot of noise?
4. Try a **more complex image** (shadows, overlapping objects, texture, etc.).
5. Repeat with different ROIs and see how the results change.

---

## 6. Common issues

- **Windows freeze or become unresponsive**

  - This can happen if the terminal is closed while the windows are open.
  - Always close the windows by pressing **Esc**, not by killing the Python process abruptly.

- **Keyboard does nothing**

  - Make sure the **image window is active (focused)** when you press keys.
  - Click once on the window and then press `Space` or `Esc` again.

- **Nothing happens after pressing Space**

  - If no ROI was drawn yet, the script may have nothing to segment.
  - Try drawing at least one rectangle and press Space again.

If you understand these controls, you can comfortably use the project even without
knowing any image-processing theory. For explanations of the algorithms themselves,
see: [Algorithms](algorithms.md).
