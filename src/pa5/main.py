from pathlib import Path
import numpy as np
import cv2

from src.pa5.utils import plot_histogram, save_image
from src.pa5.thresholding import binarization, otsu_thresholding


def main():
    # Prompt user for image name
    image_name = input("Enter image name (with extension): ").strip()
    test_image_dir = Path("images")
    image_path = test_image_dir / image_name

    # Validate image exists
    if not image_path.exists():
        print(f"Error: Image '{image_path}' not found!")
        return

    # Create save file path object
    image_save_dir = Path("output") / "images"
    plot_save_dir = Path("output") / "plots"

    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    print(f"Loaded image: {image_path}")

    # Prompt user for thresholding mode
    print("\nSelect thresholding mode:")
    print("  1 - Manual thresholding")
    print("  2 - Otsu's thresholding")

    while True:
        mode = input("Enter mode (1 or 2): ").strip()
        if mode in ["1", "2"]:
            break
        print("Invalid input. Please enter 1 or 2.")

    # Convert image to histogram array
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    plot_histogram(image, plot_save_dir, image_name)

    # Apply thresholding based on selected mode
    if mode == "1":  # Manual thresholding
        # Prompt user for threshold value
        while True:
            try:
                threshold = int(input("Enter threshold value (0-255): "))
                if 0 <= threshold <= 255:
                    break
                print("Threshold must be between 0 and 255.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        print(f"Selected threshold: {threshold}")

    else:  # Otsu's thresholding
        # Prompt user for threshold increment value
        while True:
            try:
                threshold_increment = int(
                    input("Enter a threshold increment value (1 recommended): ")
                )
                if 1 <= threshold_increment <= 255:
                    break
                print("Threshold increment must be between 1 and 255.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        threshold = otsu_thresholding(histogram, threshold_increment)
        print(f"Ideal threshold computed: {threshold}")

    # Binarize and save image
    binarized_image = binarization(image, threshold)
    save_image(
        binarized_image,
        image_save_dir,
        image_name.split(".")[0],
        "manual" if mode == "1" else "otsu",
        threshold,
    )
    print(f"\nImage saved to: {image_save_dir}")
    print(f"Histogram saved to: {plot_save_dir}")


if __name__ == "__main__":
    main()
