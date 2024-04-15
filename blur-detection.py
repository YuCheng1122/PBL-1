import argparse
import cv2
import numpy as np
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox, QFileDialog
from PyQt5.QtGui import QPixmap
import gradio as gr

# Constants for normalization
MAX_LAPLACIAN = 1000
MAX_SOBEL = 5000
MAX_EDGE_FREQUENCY = 50000
MAX_TENENGRAD = 50000000
MAX_HISTOGRAM = 500
MAX_FDA = 100
# Helper Functions for Sharpness Calculations
def variance_of_laplacian(image):
    return (cv2.Laplacian(image, cv2.CV_64F).var() / MAX_LAPLACIAN) * 100

def variance_of_sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return (np.var(gradient_magnitude) / MAX_SOBEL) * 100

def edge_frequency_sharpness(image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    f_transform = np.fft.fft2(edges)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-10)  # Avoid log(0)
    threshold = np.max(magnitude_spectrum) * 0.1
    return (np.sum(magnitude_spectrum > threshold) / MAX_EDGE_FREQUENCY) * 100

def tenengrad(image, ksize=3):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude = sobelx**2 + sobely**2
    return (np.sum(gradient_magnitude) / MAX_TENENGRAD) * 100

def histogram_spread(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return (np.std(hist) / MAX_HISTOGRAM) * 100

def assess_sharpness(image, method, **kwargs):
    method = method.lower()  # Normalize the input to lower case
    if method == 'laplacian':
        sharpness = variance_of_laplacian(image)
    elif method == 'sobel':
        sharpness = variance_of_sobel(image)
    elif method == 'edge_frequency':
        sharpness = edge_frequency_sharpness(image, **kwargs)
    elif method == 'tenengrad':
        sharpness = tenengrad(image, **kwargs)
    elif method == 'histogram':
        sharpness = histogram_spread(image)
    elif method == 'frequency domain assessment':
        sharpness = frequency_domain_assessment(image)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    category = categorize_sharpness(sharpness)
    return sharpness, category


def categorize_sharpness(sharpness_percentage):
    if sharpness_percentage < 10:
        return "Low Sharpness"
    elif sharpness_percentage < 30:
        return "Medium Sharpness"
    else:
        return "High Sharpness"

def run_gui():
    class SharpnessApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.initUI()
        
        def initUI(self):
            self.setWindowTitle('Image Sharpness Assessment Tool')
            self.setGeometry(100, 100, 800, 600)
            layout = QVBoxLayout()

            self.label = QLabel('Load an image and select a method for sharpness assessment or perform autofocus.')
            layout.addWidget(self.label)

            self.method_combo = QComboBox()
            self.method_combo.addItems(['laplacian', 'sobel', 'edge_frequency', 'tenengrad', 'histogram', 'Frequency Domain Assessment'])
            layout.addWidget(self.method_combo)

            btn_load = QPushButton('Load Image')
            btn_load.clicked.connect(self.load_image)
            layout.addWidget(btn_load)

            btn_evaluate = QPushButton('Evaluate Sharpness')
            btn_evaluate.clicked.connect(self.evaluate_sharpness)
            layout.addWidget(btn_evaluate)

            btn_autofocus = QPushButton('Perform Autofocus')
            btn_autofocus.clicked.connect(self.perform_autofocus)
            layout.addWidget(btn_autofocus)

            container = QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)
            self.image_path = None
            self.image = None

        def load_image(self):
            fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.png *.jpeg *.bmp)")
            if fname:
                self.image_path = fname
                pixmap = QPixmap(fname)
                self.label.setPixmap(pixmap.scaled(400, 400))

        def evaluate_sharpness(self):
            if self.image_path:
                image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                method = self.method_combo.currentText()
                sharpness, category = assess_sharpness(image, method)
                self.label.setText(f"Sharpness: {sharpness:.2f}%\nCategory: {category}")
            else:
                self.label.setText("Load an image first.")

        def perform_autofocus(self):
            if self.image_path:
                result = simulate_autofocus(self.image_path, self.method_combo.currentText())
                if "error" in result:
                    self.label.setText(result["error"])
                else:
                    self.label.setText(f"Best focus: {result['best_focus']}, Sharpness: {result['sharpness']:.2f}%")
            else:
                self.label.setText("Load an image first.")

    app = QApplication(sys.argv)
    ex = SharpnessApp()
    ex.show()
    sys.exit(app.exec_())


def evaluate_sharpness_web(image, method):
    sharpness, category = assess_sharpness(image, method)
    return f"Sharpness: {sharpness:.2f}%", f"Category: {category}"

def run_gradio():
    def evaluate_sharpness_web(image, method):
        sharpness, category = assess_sharpness(image, method)
        return f"Sharpness: {sharpness:.2f}%", f"Category: {category}"

    def autofocus_web(image, method):
        result = simulate_autofocus(image, method)
        if "error" in result:
            return result["error"]
        else:
            return f"Best focus: {result['best_focus']}, Sharpness: {result['sharpness']:.2f}%"

    with gr.Blocks() as demo:
        with gr.Row():
            image_input = gr.Image(label="Upload Image (supports JPG, PNG, BMP)", type="numpy")
            method_dropdown = gr.Dropdown(choices=['laplacian', 'sobel', 'edge_frequency', 'tenengrad', 'histogram', 'frequency domain assessment'], label="Select Method")
            evaluate_button = gr.Button("Evaluate Sharpness")
            autofocus_button = gr.Button("Perform Autofocus")
        sharpness_output = gr.Textbox(label="Sharpness Results")
        autofocus_output = gr.Textbox(label="Autofocus Results")

        evaluate_button.click(
            fn=evaluate_sharpness_web,
            inputs=[image_input, method_dropdown],
            outputs=sharpness_output
        )

        autofocus_button.click(
            fn=autofocus_web,
            inputs=[image_input, method_dropdown],
            outputs=autofocus_output
        )

    demo.launch()


def simulate_autofocus(image_input, method, focus_range=(5, 30), focus_steps=10):
    """
    Simulates autofocus by artificially adjusting the sharpness of the image using Gaussian blur.
    
    Parameters:
    - image_input: Either a path to the image file or a NumPy array of the image.
    - method: Sharpness assessment method to use.
    - focus_range: Tuple representing the range of Gaussian blur (simulate less to more blur).
    - focus_steps: Number of blur levels to test.
    
    Returns:
    - A dictionary with the optimal blur level and the corresponding sharpness score.
    """
    if isinstance(image_input, str):  # Check if input is a filepath
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"error": "Image could not be loaded"}
    elif isinstance(image_input, np.ndarray):  # Check if input is already a NumPy array
        image = image_input
    else:
        return {"error": "Invalid image input"}

    best_focus = None
    max_sharpness = 0

    for sigma in np.linspace(focus_range[0], focus_range[1], num=focus_steps):
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        sharpness, _ = assess_sharpness(blurred_image, method)
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_focus = sigma

    return {"best_focus": best_focus, "sharpness": max_sharpness}

def frequency_domain_assessment(image):
    """
    Assess the sharpness of an image based on the frequency domain.
    
    Parameters:
    - image: Image array in grayscale.
    
    Returns:
    - Sharpness score based on frequency domain analysis.
    """
    # Convert image to float32 for Fourier transformation
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Adding 1 to avoid log(0)
    
    # Calculate a threshold to consider only high frequency components
    mean_val = np.mean(magnitude_spectrum)
    high_freq_threshold = magnitude_spectrum > (mean_val + 1.5 * np.std(magnitude_spectrum))  # using 1.5*STD as threshold

    # Calculate sharpness score as the sum of high frequency components
    sharpness_score = np.sum(high_freq_threshold/MAX_FDA)
    return sharpness_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch GUI application")
    parser.add_argument("--web", action="store_true", help="Launch web application")
    parser.add_argument("--test-function", help="Function to test sharpness")
    parser.add_argument("--test-image", help="Path to the image file to test")
    parser.add_argument("--autofocus", action="store_true", help="Perform autofocus simulation")
    args = parser.parse_args()

    if args.gui:
        run_gui()
    elif args.web:
        run_gradio()
    elif args.autofocus and args.test_function and args.test_image:
        result = simulate_autofocus(args.test_image, args.test_function)
        if "error" in result:
            print(result["error"])
        else:
            print(f"Best focus (lower is better): {result['best_focus']}, Sharpness: {result['sharpness']:.2f}%")
    elif args.test_function and args.test_image:
        if os.path.exists(args.test_image):
            image = cv2.imread(args.test_image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read the image at {args.test_image}")
                sys.exit(1)
            sharpness, category = assess_sharpness(image, args.test_function)
            print(f"Raw sharpness of {args.test_function} on {args.test_image}: {sharpness:.2f}%")
            print(f"Clarity: {category}")
        else:
            print(f"Image not found at {args.test_image}")
            sys.exit(1)
    else:
        print("Specify an interface to launch (--gui or --web), perform autofocus (--autofocus), or provide command line arguments for sharpness testing.")

if __name__ == "__main__":
    main()
