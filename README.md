# gendr
Investigates the relationship between male-to-female ratio of Faces in the Wild dataset compared to the performance of a face classifier that utilizes FaceNet

Gender Imbalance & Face Recognition: Does the ratio of male-to-female subjects in the Labeled Faces in the Wild (LFW) dataset affect the performance of a face classification system that uses FaceNet?

## Hypothesis
The ratio of male-to-female subjects in the Labeled Faces in the Wild dataset impacts the performance of a facial recognition system utilizing FaceNet. If there are more male subjects to female subjects in the dataset, there will be a higher probability for the recognition of male individuals than female individuals. If there are more female subjects to male subjects in the dataset, there will be a higher probability for the recognition of female individuals than male individuals.

## Method 
   1. Labeled LFW Dataset with gender (male/female) labels 
   2. Modified LFW Dataset with varying male-to-female ratios producing 7 versions
   3. Classified the images of each version of the dataset 
   4. Compared average probabilities of male and female subjects across dataset versions
   
## LFW Dataset 
LFW is a database of face photographs designed for studying unconstrained face recognition. The dataset was created and maintained by researchers at the University of Massachusetts, Amherst. It is comprised of 13,233 images of 5,749 people collected from the web with 1,680 of the individuals pictured having two or more distinct photographs in the dataset. The deep-funneled version of the dataset was downloaded as this version was reported to produce superior results for face verification algorithms compared to the other image types. 

Manually verified LFW gender labels were sourced and used from ‘AFIF4: Deep Gender Classification based on AdaBoost-based Fusion of Isolated Facial Features and Foggy Faces’ by Mahmoud Afifi and Abdelrahman Abdelhamed. Male subjects were labeled with ‘M’ and female subjects ‘F’. (tables 1 and 2 below). Of the 13,233 images in the dataset, only 2966 images are of female subjects. 

To analyze the performance of the face recognition system, the dataset was modified as follows: 
Only subjects with 19 images or more were used for the experiment to allow for 16 images of that person in the training set and 3 images of the same person in the validation set. 
There are only 14 female subjects in the dataset that had 19 or more images and so only 14 male subjects with 19 or more images were used as well. 

## Face Classification System 
To compose the face classification system, the MTCNN model was used for face detection, the FaceNet model was used to create a face embedding for each detected face and then a Linear Support Vector Machine (SVM) classifier model was used to predict the identity of a given face. 

## Results
FaceNet boasts a 99.63% accuracy on the Labeled Faces in the Wild dataset and evaluating the classification model on the tutorial train and test dataset of 5 celebrities showed a perfect classification accuracy. The results of all 7 Gender Imbalance & Facial Recognition datasets showed a 100% accuracy in identifying the individual with insignificant differences in probabilities despite gender make-up. 

## Limitations 
A major limitation to this study was the small dataset that was used. Each dataset version contained 266 images. This made it difficult to gain higher probabilities as the training set was limited to only 16 images. Furthermore, more subjects are needed to be able to make a conclusive relationship between the gender make-up and the performance of the face classification system. 

## References 
Florian Schroff, Dmitry Kalenichenko, James Philbin. 2015. FaceNet: A Unified Embedding for Face Recognition and Clustering. https://arxiv.org/abs/1503.03832

Jason Brownlee. 2019. How to Develop a Face Recognition System Using FaceNet in Keras. https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

Jessica Li. 2018. Labeled Faces in the Wild (LFW) Dataset. https://www.kaggle.com/jessicali9530/lfw-dataset

Mahmoud Afifi and Abdelrahman Abdelhamed. 2017. AFIF4: Deep Gender Classification based on AdaBoost-based Fusion of Isolated Facial Features and Foggy Faces. https://arxiv.org/abs/1706.04277

Hiroki Taniai. 2018. keras-facenet. https://github.com/nyoki-mtl/keras-facenet

## Technologies 
Python, TensorFlow 2.0, Keras, Numpy, PIL, Matplotlib, skikit-learn

## Acknowledgement 
Huge thanks to Jason Grant, Assistant Professor of Computer Science at Middlebury College for advising this research project. 




