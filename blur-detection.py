from imutils import paths
import argparse
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
import gradio as gr
import sys

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def variance_of_sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.var(gradient_magnitude)

def edge_frequency_sharpness(image):
    # Edge Detection
    edges = cv2.Canny(image, 100, 200)
    # Frequency Analysis
    f_transform = np.fft.fft2(edges)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20*np.log(np.abs(f_shift))
    # Sharpness measure based on high-frequency content
    sharpness_measure = np.sum(magnitude_spectrum > np.mean(magnitude_spectrum))
    return sharpness_measure

def assess_sharpness(image_path, threshold, method):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'laplacian':
        fm = variance_of_laplacian(gray)
    elif method == 'sobel':
        fm = variance_of_sobel(gray)
    elif method == 'edge_frequency':
        fm = edge_frequency_sharpness(gray)
    else:
        raise ValueError("Unsupported method. Choose 'laplacian', 'sobel', or 'edge_frequency'.")
    
    # Calculate sharpness as a percentage of the threshold
    sharpness_percentage = (fm / threshold) * 100
    # Optionally, you can categorize the sharpness level based on percentage
    # For example, < 100% might still be considered blurry to some degree
    status = f"{sharpness_percentage:.2f}% Sharpness"  # Display sharpness level as a percentage
    
    return image_path, status

# No changes are needed in the gradio_interface function itself,
# but ensure the output label in the Gradio interface correctly reflects the new percentage-based output.
# For example, the output label could be "Sharpness Level" instead of just "Status".

# Similarly, for the PyQt interface, ensure that the labels are adjusted to display the sharpness level appropriately.


def gradio_interface(threshold, method):
    def wrapper(image_path):
        return assess_sharpness(image_path.name, threshold, method)
    gr.Interface(fn=wrapper, inputs="file", outputs=["text", "text"]).launch()

def show_test_results_qt(image_results):
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle('測試結果')
    layout = QVBoxLayout()

    for imagePath, status in image_results.items():
        label = QLabel(f"{imagePath} - {status}")
        layout.addWidget(label)

    window.setLayout(layout)
    window.show()
    sys.exit(app.exec_())

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False, help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0, help="focus measures that fall below this value will be considered 'blurry'")
ap.add_argument("--gui", choices=['qt', 'gradio'], required=True, help="Select the GUI mode: 'qt' for PyQt or 'gradio' for Gradio web interface")
ap.add_argument("--method", choices=['laplacian', 'sobel', 'edge_frequency'], required=True, help="Select the method for sharpness assessment: 'laplacian', 'sobel', or 'edge_frequency'")
args = vars(ap.parse_args())

if args["gui"] == "qt":
    image_results = {}
    for imagePath in paths.list_images(args["images"]):
        _, status = assess_sharpness(imagePath, args["threshold"], args["method"])
        image_results[imagePath] = status
    show_test_results_qt(image_results)
elif args["gui"] == "gradio":
    gradio_interface(args["threshold"], args["method"])
