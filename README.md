# Live-Video-OCR
This python script captures a frame from a live video stream, detects text in the frame and prints the output text on the screen which it later saves on the hard drive.

This script uses pytesseract
And the EAST TEXT DETECTOR is used to train the algorithm
Download the pre-trained model at https://drive.google.com/open?id=1yHEuc6AK0JI0yzR4Qcru0Z_6GVGHkwHV

You can improve text accuracy by increasing or decreasing the padding line 147
Also being familiar with the East text detector engine will help to select the right configuration on line 179.
