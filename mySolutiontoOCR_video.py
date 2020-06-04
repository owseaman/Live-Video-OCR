"""
ocr script
This script detects and outputs the detected text in a video frame.
Can work for live video stream with little tweak.
"""

import os
import imutils
import cv2
import numpy as np
import pytesseract

# user packages
from east_text_detector import east_detector

# The next line is needed in windows only,
# Put the path to tesseract.exe here
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# load the input image and grab the image dimensions
cam = cv2.VideoCapture(0)

#cv2.namedWindow("Press Space bar to Capture, Esc to Exit")

img_counter = 0 # In case multiple images need to be read

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Press Space bar to Capture, Esc to Exit", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        # You may comment the next line out if it works for you
        newFrame = cv2.imwrite(img_name, frame) #Sometimes not saving to disk throws back error, and sometimes it works, so I write
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows() # Or cv2.destroyWindow(frame)
input_image = cv2.imread(img_name)

gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

# detect paper edges
edge_image = cv2.Canny(gray_image, threshold1=100, threshold2=100)
cv2.imshow('gray 1', gray_image) # Debugging

def order_points(pts): # Copied from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts): # Copied from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

# I tried each dimensions and I got the same results
new_corners = np.float32([[0, 0], [540, 0], [540, 720], [0, 720]])
#new_corners = np.float32([[0, 0], [540, 0], [0, 720], [540, 720]])
#new_corners = np.float32([[0, 0], [0, 720], [540, 0], [540, 720]])
#new_corners = np.float32([[0, 0], [540, 0], [0, 720], [540, 720]])

print(input_image.shape)


warped_image = four_point_transform(input_image, new_corners )
#ratio = input_image.shape[0]

cv2.imshow('warped image', warped_image)

# detect text bounding boxes
boxes, confidences = east_detector(warped_image)
(H, W) = warped_image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (640, 640)
rW = W / float(newW)
rH = H / float(newH)

# padding for bounding boxes
padding = 0.05 # Increase/Decrease this to depending on your font size. This worked well for my Test Images

results = []
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # use pytessaract to get text
    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    dX = int((endX - startX) * padding)
    dY = int((endY - startY) * padding)
    # apply padding to each side of the bounding box, respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(540, endX + (dX * 2))
    endY = min(720, endY + (dY * 2))

    # extract the actual padded ROI
    roi = warped_image[startY:endY, startX:endX]

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = "-l eng --oem 1 --psm 7"
    text = pytesseract.image_to_string(roi, config=config)
    # add the bounding box coordinates and OCR'd text to the list
    # of results
    results.append(((startX, startY, endX, endY), text))

    # draw the bounding box on the image
    # cv2.rectangle(warped_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# sort the results bounding box coordinates from top to bottom
line_quantize = 40
image_height = 720
results = sorted(
    results, key=lambda r: r[0][1] * image_height // line_quantize + r[0][0]
)

# loop over the results
all_text = ""
for ((startX, startY, endX, endY), text) in results:
    # display the text OCR'd by Tesseract

    # print("OCR TEXT")
    # print("========")
    # print("{}\n".format(text))

    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV, then draw the text and a bounding box surrounding
    # the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    all_text += " " + text
    # output = warped_image.copy()
    cv2.rectangle(warped_image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(
        warped_image,
        text,
        (startX, startY),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )

print(all_text)

# write output to file # a append, r read, w write
output_file = open("mySolutiontoOCR_video.txt", "a")
output_file.write(all_text)
output_file.close()

# show the output image
cv2.imshow("Text Detection", warped_image)
cv2.waitKey(0)

# close all windows
cv2.destroyAllWindows()
