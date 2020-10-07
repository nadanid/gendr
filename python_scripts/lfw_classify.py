import csv 
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot

# load faces
data = load('lfw-dataset-19-v2-m80-f20.npz')
testX_faces = data['arr_2']

# load embeddings dataset
data = load('lfw-dataset-19-v2-m80-f20-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']


# normalize input vectors because the vectors are often compared to each other using a distance metric
# vector normalization means scaling the values until the length or magnitude of the vectors is 1 or unit length
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
# string target variables for each celebrity converted to integers
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
# SVM is used to normalize face embedding inputs. This is because the method is very effective at separating the face embedding vectors.
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy) # the fit model is used to make a prediction for each example in the train and test datasets and then calculates the classification accuracy.

# test model on a random example from the test dataset
# random example selected, face embedding retrieved, face pixels, expected class prediction and the corresponding class name
with open('lfw-dataset-19-v2-m80-f20.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Expected", "Predicted", "Accuracy"])

        for selection in range(testX.shape[0]):
                face_pixels = testX_faces[selection]
                face_emb = testX[selection]
                face_class = testy[selection]
                face_name = out_encoder.inverse_transform([face_class])

                # prediction for the face using the face embedding as input prediciting both class integer and probability of the prediction
                samples = expand_dims(face_emb, axis=0)
                yhat_class = model.predict(samples)
                yhat_prob = model.predict_proba(samples)

                # get name for the predicted class integer, and the probablity for this prediction
                class_index = yhat_class[0]
                class_probability = yhat_prob[0, class_index] * 100
                predict_names = out_encoder.inverse_transform(yhat_class)
                print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
                print('Expected: %s' % face_name[0])

                # plot for fun
                # pyplot.imshow(face_pixels)
                # title = '%s (%.3f)' % (predict_names[0], class_probability)
                # pyplot.title(title)
                # pyplot.show()

                writer.writerow([face_name[0], predict_names[0], class_probability])

csv_file.close()
