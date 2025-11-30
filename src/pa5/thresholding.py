import numpy as np


def binarization(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Apply simple binary thresholding.

    Args:
        image: Grayscale image as a NumPy array
        threshold: Threshold for binarization

    Returns:
        Binarized image with threshold applied
    """
    return np.where(image >= threshold, 255, 0).astype(np.uint8)


def otsu_thresholding(
    histogram_array: np.ndarray, threshold_increment: int, threshold_max: int = 255
) -> int:
    """
    Compute optimal threshold using Otsu's method.

    Args:
        histogram_array: Histogram of pixel intensities as np.ndarray
        threshold_increment: Step size for testing potential thresholds
        threshold_max: Maximum intensity value to consider (default: 255 for 8-bit images)

    Returns:
        Optimal threshold value that maximizes between-class variance
    """
    threshold = 0
    BCV_max = 0
    ideal_threshold = 0

    # Brute force attempt, iterating by threshold_increment
    while threshold < threshold_max:
        P1, P2, M1, M2 = compute_prob(histogram_array, threshold, threshold_max)
        BCV = compute_BCV(P1, P2, M1, M2)
        # Update new ideal_threshold
        if BCV > BCV_max:
            BCV_max = BCV
            ideal_threshold = threshold

        threshold += threshold_increment

    return ideal_threshold


def compute_prob(
    histogram_array: np.ndarray, threshold: int, threshold_max: int
) -> tuple[float, float, float, float]:
    """
    Compute class probabilities and means for a given threshold.

    Args:
        histogram_array: Histogram of pixel intensities as np.ndarray
        threshold: Current threshold value separating the two classes
        threshold_max: Maximum intensity value to consider (default: 255 for 8-bit images)

    Returns:
        Tuple of (P1, P2, M1, M2) where:
            - P1: Probability of class 1 (pixels <= threshold)
            - P2: Probability of class 2 (pixels > threshold)
            - M1: Mean intensity of class 1
            - M2: Mean intensity of class 2
    """
    P1, P2, M1, M2 = 0, 0, 0, 0
    total_pixels = np.sum(histogram_array)  # Count pixels in image

    # Calculate class probabality of pixels within threshold
    for i in range(0, threshold + 1):
        P1 += histogram_array[i] / total_pixels

    # Calculate class probablity of pixels outside threshold
    P2 = 1 - P1

    # Calculate mean of foreground classes
    if P1 > 0:
        for i in range(0, threshold + 1):
            M1 += (i * (histogram_array[i] / total_pixels)) / P1

    # Calculate mean of background classes
    if P2 > 0:
        for i in range(threshold + 1, threshold_max + 1):
            M2 += (i * (histogram_array[i] / total_pixels)) / P2

    return P1, P2, M1, M2


def compute_BCV(P1: float, P2: float, M1: float, M2: float) -> float:
    """
    Compute between-class variance for Otsu's method.

    Args:
        P1: Probability of class 1
        P2: Probability of class 2
        M1: Mean intensity of class 1
        M2: Mean intensity of class 2

    Returns:
        Between-class variance
    """
    return P1 * P2 * (M1 - M2) ** 2
