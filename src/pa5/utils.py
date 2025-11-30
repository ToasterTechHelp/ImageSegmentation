from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_histogram(image: np.ndarray, plot_save_dir: Path, image_name: str) -> None:
    """
    Plot and save a histogram of pixel intensities for a grayscale image.

    Args:
        image: Grayscale image as a NumPy array
        plot_save_dir: Directory path where the histogram plot will be saved
        image_name: Name identifier for the image
    """
    # Create save path
    plot_save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    save_path = plot_save_dir / ("histogram_plot_" + image_name + ".png")

    # Turn image into histogram plot
    plt.hist(image.flatten(), bins=256, range=(0, 256))
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Image Histogram")

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def save_image(
    image: np.ndarray, save_dir: Path, image_name: str, mode: str, threshold_value: str
) -> None:
    """
    Save a NumPy array as an image file with a descriptive filename.

    Args:
        image: Image as NumPy array
        save_dir: Directory path where image will be saved
        image_name: Base name identifier for the image
        mode: The image processing mode
    """
    save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    filename = f"{image_name}_{mode}_{threshold_value}.png"
    save_path = save_dir / filename
    cv2.imwrite(str(save_path), image)
