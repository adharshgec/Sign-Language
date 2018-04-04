import cv2, os, random
import numpy as np

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

gestures = os.listdir('gestures/')
gestures.sort(key = int)
start_index = 0
end_index = 5
image_height, image_width = get_image_size()

if len(gestures)%5 != 0:
	rows = int(len(gestures)/5)+1
else:
	rows = int(len(gestures)/5)

all_gestures_img = None
for i in range(rows):
	col_img = None
	for j in range(start_index, end_index):
		img_path = "gestures/%s/%d.jpg" % (j, random.randint(1, 1200))
		img = cv2.imread(img_path, 0)
		if np.any(img == None):
			img = np.zeros((image_height, image_width), dtype = np.uint8)
		if np.any(col_img == None):
			col_img = img
		else:
			col_img = np.hstack((col_img, img))

	start_index += 5
	end_index += 5
	if np.any(all_gestures_img == None):
		all_gestures_img = col_img
	else:
		all_gestures_img = np.vstack((all_gestures_img, col_img))


cv2.imshow("gestures", all_gestures_img)
cv2.waitKey(0)