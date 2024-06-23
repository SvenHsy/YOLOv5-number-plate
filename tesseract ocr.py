import cv2
from PIL import Image

import pytesseract


def my_function(image):
    # import the necessary packages
    import numpy as np
    # import argparse
    import cv2

    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow('C:/Users/master/Desktop/reserch/yolov5-mask-42-master/runs/detect/exp35/crops/license plate/P1000385.jpg', image)
    cv2.imshow('C:/Users/master/Desktop/reserch/yolov5-mask-42-master/runs/detect/exp35/crops/license plate/P1000385.jpg', rotated)
    return rotated
    cv2.waitKey(0)

text = pytesseract.image_to_string('C:/Users/master/Desktop/reserch/yolov5-mask-42-master/runs/detect/exp35/crops/license plate/P1000385.jpg', lang="jpn")
print(text)

import numpy as np
import pytesseract
from pytesseract import Output
import cv2

try:
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
except ImportError:
    import Image

img = cv2.imread('C:/Users/master/Desktop/reserch/yolov5-mask-42-master/runs/detect/exp35/crops/license plate/P1000385.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

width_list = []
for c in cnts:
    _, _, w, _ = cv2.boundingRect(c)
    width_list.append(w)
wm = np.median(width_list)

tess_text = pytesseract.image_to_data(img, output_type=Output.DICT, lang='jpn')
for i in range(len(tess_text['text'])):
    word_len = len(tess_text['text'][i])
    if word_len > 1:
        world_w = int(wm * word_len)
        (x, y, w, h) = (tess_text['left'][i], tess_text['top'][i], tess_text['width'][i], tess_text['height'][i])
        cv2.rectangle(img, (x, y), (x + world_w, y + h), (255, 0, 0), 1)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font="simsun.ttc", size=18, encoding="utf-8")
        draw.text((x, y - 20), tess_text['text'][i], (255, 0, 0), font=font)
        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

cv2.imshow("TextBoundingBoxes", img)
cv2.waitKey(0)



