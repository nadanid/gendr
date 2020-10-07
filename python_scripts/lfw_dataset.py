# Arranged by Nadani Dixon; source: How to Develop a Face Recognition System Using FaceNet in Keras by Jason Brownlee
# 1. Steps over each subdirectory for a given dataset (eg. train or val)
# 2. Extract faces from an image and and returns an array containing the height, width and number of channels
# 3. Prepares a dataset with the name as the output label for each detected face 

# Generally: all of the photos in the ‘train‘ dataset are loaded, then faces are extracted, resulting in samples with square face input and a class label string as output. Then the ‘val‘ dataset is loaded, providing samples that can be used as a test dataset. Both datasets are then saved to a compressed NumPy array file called ‘lfw-dataset.npz‘



# face detection for the Labelled Faces in the Wild  Dataset 
import os
from os.path import isdir 
from os import listdir 
from PIL import Image 
from numpy import array 
from numpy import asarray 
from numpy import savez_compressed 
from matplotlib import pyplot 
from mtcnn.mtcnn import MTCNN



# FUNCTIONS
# extract a single face from a given photograph and returns a array containing the width, height and number of channels in the image  
def extract_face(filename, required_size=(160, 160)): 
	# load image from file 
	image = Image.open(filename)
	# convert to RGB, if needed 
	image = image.convert('RGB') 
	# convert to array 
	pixels = asarray(image) 

	# create the detector, using default weights 
	detector = MTCNN() 
	# detect faces in the image 
	# result is a list of bounding boxes where each bounding box defines a
	# ... lower-left corner of the bounding box, as well as the width and height 
	results = detector.detect_faces(pixels)   

	# extract the bounding box from the first face 
	x1, y1, width, height = results[0]['box'] 
	x1, y1 = abs(x1), abs(y1) 
	x2, y2 = x1 + width, y1 + height 

	# extract the face 
	face = pixels[y1:y2, x1:x2] 

	# resize pixels to the expected model size 
	image = Image.fromarray(face) 
	image = image.resize((160, 160))
	face_array = asarray(image) 
	
	return face_array 



# load images and extract faces for all images in a directory 
# each face has one label, the name of the person in the photograph
def load_faces(directory): 
	faces = list() 
	# enumerate files 
	for filename in listdir(directory): 
		if filename != ".DS_Store": 
			# path 
			path = directory + filename 
			# get face 
			face = extract_face(path) 
			# store 
			faces.append(face) 
	return faces 



# loads a dataset that contains one subdir for each class that in turn contain images and returns the x and y elements of the dataset as NumPy arrays 
# for example, takes a directory name such as 'lfw-dataset/train/' and detects faces for each subdirectory, assigning labels to each detected face


def load_dataset(directory):
        X, y = list(), list()
        # enumerate folders, on per class
        for subdir in listdir(directory):
                # path
                path = directory + subdir + '/'
                # skip any files that might be in the dir
                if not isdir(path):
                        continue
                # load all faces in the subdirectory
                faces = load_faces(path)
                # create labels
                labels = [subdir for _ in range(len(faces))]
                # summarize progress
                print('>loaded %d examples for class: %s' % (len(faces), subdir))
                # store
                X.extend(faces)
                y.extend(labels)
        return asarray(X), asarray(y)

# MAIN
# load train dataset
trainX, trainy = load_dataset('lfw-19-v2-m80-f20/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('lfw-19-v2-m80-f20/val/')
# save arrays to one file in compressed format
savez_compressed('lfw-dataset-19-v2-m80-f20.npz', trainX, trainy, testX, testy)
