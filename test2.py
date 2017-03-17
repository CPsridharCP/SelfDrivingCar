import cv2
import argparse
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
cv2.imshow("Original Image",image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("un-blurred", hsv)
hsv = cv2.GaussianBlur(hsv,(7,7),0)
cv2.imshow("blurred", hsv)
print "shapee of blurred:: ", hsv.shape
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height = np.size(hsv,0)
width = np.size(hsv,1)

#hsv = cv2.circle(hsv, (1*width/8, 2*height/8), 45, (255,0,0), 3)
#hsv = cv2.circle(hsv, (7*width/8, 2*height/8), 45, (255,0,0), 3)
#hsv = cv2.circle(hsv, (width/2, 5*height/8), 150, (255,0,0), 3)

cropped = hsv[5*height/8-75:5*height/8+75, width/2-75:width/2+75]
side = hsv[2*height/8-22:2*height/8+22, 1*width/8-22:1*width/8+22]
#(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cropped)
height = np.size(cropped,0)
width = np.size(cropped,1)

#cv2.imshow("sides",side)

print width
print height

h = []
s = []
v = []
hs = []
ss = []
vs = []

a = [([1,2,0],[1,2,0]),([1,2,0],[1,2,0])]
for i in cropped:
	for j in i:
		h.append(j[0])
		s.append(j[1])
		v.append(j[2])

for i in side:
	for j in i:
		hs.append(j[0])
		ss.append(j[1])
		vs.append(j[2])


hmin = min(h)
hmax = max(h)
smin = min(s)
smax = max(s)
vmin = min(v)
vmax = max(v)
hsmin = min(hs)
hsmax = max(hs)
ssmin = min(ss)
ssmax = max(ss)
vsmin = min(vs)
vsmax = max(vs)

rangeMin = np.array([hmin, smin, vmin])
rangeMax = np.array([hmax, smax, vmax])
rangeMins = np.array([hsmin, ssmin, vsmin])
rangeMaxs = np.array([hsmax, ssmax, vsmax])

print "hsv shape:: ", hsv.shape
mask = cv2.inRange(hsv, rangeMin, rangeMax)
print "mask shape:: ", mask.shape
mask2 = cv2.inRange(hsv, rangeMins, rangeMaxs)
print "mask2 shape:: ", mask2.shape
final_mask = mask - mask2
print "final_mask shape:: ", final_mask.shape

#opening = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
#opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (101,101))
#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (501,501))

cv2.imshow("mask",mask)
#cv2.imshow("mask2",mask2)
#cv2.imshow("final_mask",final_mask)

#cv2.imshow("closing",closing)

kernel = np.ones((7,7),np.uint8)
erosion = cv2.erode(mask,kernel,iterations = 1)
cv2.imshow("erosion",erosion)
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening",opening)
kernel = np.ones((25,25),np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing",closing)

### CLARIFY WHY YOU GET WHAT YOU GET  WHEN USING FINAL MASK
output = cv2.bitwise_and(hsv, hsv, mask = closing)
print "output shape:: ", output.shape
print "gray shape:: ", gray.shape

g_final = cv2.bitwise_and(gray, gray, mask = mask)
#cv2.imshow("grey final out", g_final)

cv2.imshow("mask with h_min_max s_min_max v_min_max", output)
cv2.waitKey(0)



















