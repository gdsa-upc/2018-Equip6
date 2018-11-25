import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cv2


cv2.__version__

# Initialize ORB detector
orb = cv2.ORB_create()

# Read the image from OpenCV
img = cv2.imread('castell.png',0)

# Detect the keypoints
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img_with_keypoints = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0, outImage=np.array([]))
plt.imshow(img_with_keypoints),plt.show()

cv2.__version__

# Initialize ORB detector
orb = cv2.ORB_create()

# Read the image from OpenCV
img1 = cv2.imread('castell2.jpg',0)

# Detect the keypoints
kp1 = orb.detect(img1,None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)

# draw only keypoints location,not size and orientation
img_with_keypoints = cv2.drawKeypoints(img1,kp1,color=(0,255,0), flags=0, outImage=np.array([]))
plt.imshow(img_with_keypoints),plt.show()

#Comprovem les coincidencies entre les dues fotografies

# Compute the matches between the two images
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches,key=lambda val: val.distance)

# Show the matches
img_out = cv2.drawMatches(img, kp, img1, kp1, matches[:20], None)

plt.imshow(img_out)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
