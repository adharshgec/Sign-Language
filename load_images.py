import cv2, os
import numpy as np
import random
from sklearn.utils import shuffle
import pickle

def pickle_image_labels():
	gest_folder = "gestures"
	image_labels = []
	images = []
	labels = []
	for g_id in os.listdir(gest_folder):
		for i in range(1200):
			img = cv2.imread(gest_folder+"/"+g_id+"/"+str(i+1)+".jpg", 0)
			if np.any(img == None):
				continue
			image_labels.append((np.array(img, dtype=np.float32), int(g_id)))
	return image_labels

def split_image_labels(image_labels):
	images = []
	labels = []
	for (image, label) in image_labels:
		images.append(image)
		labels.append(label)
	return images, labels

image_labels = pickle_image_labels()
image_labels = shuffle(shuffle(shuffle(image_labels)))
images, labels = split_image_labels(image_labels)
print("Length of image_labels", len(image_labels))

train_images = images[:int(5/6*len(images))]
print("Length of train_images", len(train_images))
with open("train_images", "wb") as f:
	pickle.dump(train_images, f)
del train_images

train_labels = labels[:int(5/6*len(labels))]
print("Length of train_labels", len(train_labels))
with open("train_labels", "wb") as f:
	pickle.dump(train_labels, f)
del train_labels

test_images = images[int(5/6*len(images)):]
print("Length of test_images", len(test_images))
with open("test_images", "wb") as f:
	pickle.dump(test_images, f)
del test_images

test_labels = labels[int(5/6*len(labels)):]
print("Length of test_labels", len(test_labels))
with open("test_labels", "wb") as f:
	pickle.dump(test_labels, f)
del test_labels