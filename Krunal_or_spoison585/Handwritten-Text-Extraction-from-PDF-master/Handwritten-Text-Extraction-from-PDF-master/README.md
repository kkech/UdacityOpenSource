# Handwritten-Text-Extraction-from-PDF



Given a scanned query document, you have to predict( extract) the (a) Date (b) Bank a/c number. The document template is same across all images and it as above.

Libraries and Dependencies Used:
-python 3
-Pip3
-numpy 
-pandas as pd
-matplotlib
-math
-os
-sys
-PIL
-cv2
-wand
-imagemagick
-pydbgen
-pytesseract
-pillow
-tesseract
-ghostscript

Tutorials and references:

1. https://medium.com/@winston.smith.spb/python-ocr-for-pdf-or-compare-textract-pytesseract-and-pyocr-acb19122f38c

2. https://www.youtube.com/watch?v=_5ml_Y9hqG8 (How to read text from an image using python and tesser OCR with pytesseract)

3. https://www.ghostscript.com/download/gsdnld.html (You need to install Ghostscript in order to rasterize vector files (PDF, EPS, PS, etc.) with ImageMagick.)

4. https://www.youtube.com/watch?v=YM8j9dzuKsk (Tesseract installation on windows)

5. https://stackoverflow.com/questions/32466112/imagemagick-convert-pdf-to-jpeg-failedtoexecutecommand-gswin32c-exe-pdfdel


how it works:

converting the pdf to image via imagemagick,wand and ghostscript

resize the image using fx,fy values

cleaning the image by using threshold, contrast, grayscale, bgrtorgb, gaussian blur etc

Change the values of threshold acc. to the image preferences also erode and dilate

extracting the text from image via pytesseract
