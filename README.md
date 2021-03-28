# Polymer Membrane Pore Analysis
A program used to analyze pore size distributions of polymer membrane SEM images. 

## Overview
Both Tesseract-OCR Engine and Otsu thresholding are iplimented to clean the SEM images to help identify pores in the membrane. The cleaned images are analyzed with my algorithm which outputs an average pore size vs. film depth plot. The resultant plots show quantitatively how asymmetrical the pore Sizes become further down the polymer membrane film.

## Dependencies
* **python>=3.6**
* **pytesseract**: If you don't have tesseract executable in your PATH, include the following: pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
* **cv2**
* **numpy**
* **matplotlib**

## Feedback
For questions and comments, feel free to contact [James Von Raj](mailto:jvraj@calpoly.edu).

## License
MIT
