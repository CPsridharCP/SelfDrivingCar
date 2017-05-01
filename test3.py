# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

print "Hello"
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")

args = vars(ap.parse_args())

if not args.get("video", False):
	camera = cv2.VideoCapture(0)

else:
	camera = cv2.VideoCapture(args["video"])

while True:

	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	image = imutils.resize(frame, width=600)
	image = cv2.GaussianBlur(image,(11,11),0)
	
	#hsv = image
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	#hsv = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
	hsv = cv2.GaussianBlur(hsv,(11,11),0) #7,7 works fine

	height = np.size(hsv,0)
	width = np.size(hsv,1)

	#hsv = cv2.rectangle(hsv, (0, 0), (width, 5*height/16), (255,0,0), 3)
	hsv = hsv[5*height/16:height, 0:width]

	height = np.size(hsv,0)
	width = np.size(hsv,1)

	#hsv = cv2.circle(hsv, (1*width/32, 1*height/32), 25, (255,0,0), 3)
	#hsv = cv2.circle(hsv, (31*width/32, 1*height/32), 25, (255,0,0), 3)
	#hsv = cv2.circle(hsv, (width/2, height/2), 6*width/32, (255,0,0), 3)

    # OUR REFERENCE PATH
    # centre rectangle
	#path = hsv[height/2-(2*width/32):height/2+(2*width/32), width/2-(4*width/32):width/2+(4*width/32)]
	# Base width full rectangle
	#path = hsv[5*height/8:height,0:width]
	# centre full vertical
	path = hsv[1*height/8:height,width/2-(2*width/8):width/2+(2*width/8)]

	#hPath = np.size(path,0)
	#wPath = np.size(path,1)

	h = []
	s = []
	v = []

	for i in path:
		for j in i:
			h.append(j[0])
			s.append(j[1])
			v.append(j[2])

	hmin = min(h)
	hmax = max(h)
	smin = min(s)
	smax = max(s)
	vmin = min(v)
	vmax = max(v)

	rangeMin = np.array([hmin, smin, vmin])
	rangeMax = np.array([hmax, smax, vmax])
    
	#print "rangeMin : ",rangeMin
	#print "rangeMax : ",rangeMax

	mask = cv2.inRange(hsv, rangeMin, rangeMax)
	#mask = cv2.circle(mask, (width/2, 3*height/8), 5, (0,0,0), 2)

	cv2.imshow("Mask", mask)
	cv2.imshow("image",image)

	kernel = np.ones((25,25),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 1)
	cv2.imshow("Erosion",erosion)
	opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
	cv2.imshow("opening",opening)
	kernel = np.ones((15,15),np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("closing",closing)

	hMask = np.size(closing,0)
	wMask = np.size(closing,1)
	
	leftTop  = closing[0][2*wMask/16]
	rightTop = closing[0][14*wMask/16-1]
	if leftTop == 0 and rightTop == 0:
		message = "10000011"
	elif leftTop == 255 and rightTop == 255:
		message = "10000011"
	elif leftTop == 255 and rightTop == 0:
		message = "10012011"
	elif leftTop == 0 and rightTop == 255:
		message = "10012010"

	print message

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()