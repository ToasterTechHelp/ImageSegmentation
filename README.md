# Image Segmentation with Otsu's Thresholding

This project is a Python implementation of **Otsu's Automatic Thresholding** algorithm for image segmentation, developed as part of _CAP 5415 – Computer Vision_ (Programming Assignment 5).

## Overview

The program implements both manual and automatic thresholding methods:

1. **Grayscale loading** – load an image and represent it as a NumPy matrix.
2. **Histogram computation** – calculate the intensity distribution of pixel values.
3. **Class probability calculation** – compute cumulative probabilities for each potential threshold.
4. **Mean intensity calculation** – determine the mean intensity for foreground and background classes.
5. **Between-class variance** – compute σ²_b to measure separation between classes.
6. **Threshold optimization** – find the threshold that maximizes between-class variance.
7. **Binarization** – apply the threshold to segment the image into binary output.

> Implemented from scratch following the Otsu's method algorithm to understand how automatic thresholding works mathematically instead of relying on built-in OpenCV functions.

## Implementation Details

The implementation follows the standard **Otsu's method definition and pseudocode**:

- **Definition**: The method uses the histogram of the given image as input and aims at providing the best threshold by maximizing the **between-class variance** σ²_b.

- **Formula**: σ²_b = P₁P₂(μ₁ - μ₂)²

  - Where P₁, P₂ are class probabilities
  - μ₁, μ₂ are mean intensities of foreground and background classes

- **Algorithm**: Brute-force search testing each potential threshold value from 0 to G_max (255), computing class probabilities c_I(u) and class means μᵢ(u) for each threshold, then selecting the threshold that maximizes σ²_b.

**Note**: More detailed pseudo-code can be found in the [project report](./docs/report.pdf) (see `/docs`).

---

## Installation

**Requirements**

- Python 3.12+
- Poetry (recommended)

**Install dependencies (Poetry):**

```bash
poetry install
```

**Or install with pip:**

```bash
pip install -r requirements.txt
```

---

## Usage

**Run the main program:**

```bash
poetry run python -m src.pa5.main
```

**Interactive Steps:**

1. Place input images in `images/`.
2. Run the program and you will be prompted to:
   - Enter the image name (with extension, e.g., `238011.jpg`)
   - Select thresholding mode:
     - **Mode 1** – Manual thresholding (you specify the threshold value)
     - **Mode 2** – Otsu's automatic thresholding (algorithm computes optimal threshold)
3. For Mode 2, you can specify the threshold increment (1 recommended for accuracy).
4. Outputs will be saved to:
   - `output/images/` – Binarized images
   - `output/plots/` – Histogram visualizations

---

## Results and Findings

- **Otsu's method** works best on images with distinct foreground and backgrounds.
- The algorithm automatically finds the optimal threshold by maximizing the between-class variance σ²_b.
- **Manual thresholding** is useful for images where you have specific details in mind you'd like to keep.

Example results and analysis are included in the [project report](./docs/report.pdf) (see `/docs`).

---

## Thoughts

Re-implementing Otsu's method gave me a deeper understanding of how segmentation via binarization works. Computing class probabilities, means, and between-class variance for every possible threshold value showed me the elegance and simplicity of this algorithm, making it clear why it is the preferred segmentation method in certain scenarios. The algorithm's ability to automatically find optimal thresholds without user intervention makes it a fundamental technique in image segmentation, and understanding its mathematics helps appreciate why it remains widely used in computer vision applications.
