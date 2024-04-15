# Blur Detection for MVA PBL# HW

This repository contains the Python code for the blur detection component of the Machine Vision Application (MVA) Project-Based Learning (PBL) homework assignment. The `blur-detection.py` script utilizes various image processing techniques to detect and flag blurry images.

## Getting Started

To get started with this project, clone this repository to your local machine. You will need Python installed on your system.

### Prerequisites

Ensure you have Python 3.8 or newer installed. This project relies on several external libraries, which can be installed using `pip`.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
```
2. Pip install packages:
```bash
pip install requirements.txt
```
## Sharpness Assessment Methods
The script assesses image sharpness using the following methods:

- Laplacian (拉普拉斯變換): Calculates the sum of the second derivatives. A higher variance indicates a clearer image.
- Sobel (索貝爾算子): Detects horizontal and vertical edges by convolution kernels. Higher gradient values suggest a clearer image.
- Edge Frequency (邊緣頻率): Works with Canny, Sobel, Laplacian to identify image edges, calculating edge density and distribution. Higher frequencies indicate a clearer image.
- Tenengrad (梯度變化): Uses the Sobel operator to calculate image gradients. The sum of squared gradients serves as a sharpness indicator. Larger variations typically imply a sharper image.
- Histogram Spread (直方圖擴散): Analyzes the image's brightness histogram, calculating its standard deviation. A wider histogram distribution signifies richer image details and higher sharpness.
- Frequency Domain Analysis (頻域分析): Transforms the image to the frequency domain using Fast Fourier Transform (FFT) and analyzes high-frequency components. More high-frequency components usually mean more image detail and higher sharpness.

## Running the Application
The script can be executed in two different modes: Gradio for a web-based interface and PyQt for a graphical user interface (GUI).

1. To run the Gradio interface:
``` bash
python blur-detection.py --web
```

1. To run the PyQt GUI:
``` bash
python blur-detection.py --gui
```
## Usage
The user can select an image and choose a sharpness assessment method from the application interface. Additionally, there is an option to perform autofocus simulation to determine the best simulated focus for an image.

For detailed usage and more options, refer to the inline comments and documentation within the blur-detection.py script.
