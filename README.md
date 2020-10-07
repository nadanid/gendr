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
