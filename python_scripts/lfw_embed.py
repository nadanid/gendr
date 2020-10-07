# Arranged by Nadani Dixon; source: How to Develop a Face Recognition System Using FaceNet in Keras by Jason Brownlee

# pre-process a face to calculate a face embedding for each face in both train and test datasets using facenet to be stored as used as imput to the classifer model 
import os
from numpy import load 
from numpy import expand_dims 
from numpy import asarray 
from numpy import savez_compressed
from keras.models import load_model

# FUNCTIONS 
# returns a face embedding for a single image of a face 
def get_embedding(model, face_pixels): 
	# scale pixel values because FaceNet model expects that the pixel values are standardized 
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global) 
	mean, std = face_pixels.mean(), face_pixels.std() 
	face_pixels = (face_pixels - mean) / std 
	# transform face into one sample by expanding the dimensions  
	samples = expand_dims(face_pixels, axis=0) 
	# make prediction to get embedding 
	yhat = model.predict(samples) 
	return yhat [0]


# MAIN 
# load the face dataset 
data = load('lfw-dataset-19-v2-m80-f20.npz') 
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'] 
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape) 
# load the facenet model 
dir = os.getcwd()
model = load_model(dir+'/model/facenet_keras.h5')
print('Loaded model') 

# convert each face in the train set to an embedding 
newTrainX = list() 
for face_pixels in trainX: 
	embedding = get_embedding(model, face_pixels) 
	newTrainX.append(embedding) 
newTrainX = asarray(newTrainX) 
print(newTrainX.shape) 

#  convert each face in the test set to an embedding
newTestX = list() 
for face_pixels in testX: 
        embedding = get_embedding(model, face_pixels) 
        newTestX.append(embedding) 
newTestX = asarray(newTestX) 
print(newTestX.shape)

# save arrays to one file in compressed format 
savez_compressed('lfw-dataset-19-v2-m80-f20-embeddings.npz', newTrainX, trainy, newTestX, testy)  
