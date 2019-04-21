# Decoding-Image-Classifier
Decoding The Neural Networks
Introduction:
	The Key Barrier of the Artificial Intelligence or Deep Learning model in healthcare is the “Black Box” problem. The Neural Network is indeed the state of the art machine learning algorithm that models the complex relationships between numerous factors to make precise and accurate predictions or recommendations. Despite being very accurate the problem to interpret the results makes it difficult to integrate Artificial Intelligence in healthcare. In the course of the independent study, I have tried exploring different methods to address the black box issue of Artificial Intelligence.

Image Classification Task:
	An Image classifier was build using the pre-trained Convolutional Neural Network called “VGG-16” on the Imagenet, the database of 22,000 image types.1 The Pre-trained weights of the model were tuned to the dataset with 4000 images of cats and 4000 images of dogs to build an image classifier predicting cats and dogs. The data for training the model was retrieved from Kaggle:https://www.kaggle.com/tongpython/cat-and-dog 

Local Interpretable Model Agnostic Explanation(LIME):
	LIME is a model specific approach that interprets the black box by perturbing inputs and measuring the change in the prediction. The LIME model uses a set of explanation which tries to understand the relationship between the feature and the output by approximating the model locally with a linear model which are more interpretable.2 For the analysis, a LIME model was tested using the image with the dog and cat as shown in the figure.

The image was initially split into superpixels which are built by approximating the image regions locally. The superpixels were used by the LIME model to measure the interaction between a different feature to identify the pro-dog and the pro-cat features considered by the model in the given image.

Permutation-Based Approach:
	The Permutation-Based Approach is a model specific interpretation approach that measures the model’s prediction error for every permutation of the input feature and measures the relative importance by estimating the impact of the feature on the performance error.3 For the analysis, a test image was permuted about 1000 times by hiding certain regions of the pixel and the prediction probability was measured for every permutation based on the contribution to the prediction error the relative importance of the feature with respect to cat and dog was computed.

Saliency Maps:
	Saliency refers to the most important features of the image in the context of visual processing. The saliency maps are used to differentiate the visually alluring features from each other.4 In the context of the analysis the saliency maps were used in identifying the pro-cat and pro-dog pixels in the test image learned by the model to make a prediction. The saliency maps were constructed by differentiating the test image with respect to the class(cats or dogs). The gradients of the with respect to the class was computed by computing weights after one complete backpropagation.


Conclusion:
	The methods explored during the class shows that the experimented methods can also be further expanded to cross-sectional and other data formats to identify the key variables in making the decision. This indeed proves that black box models can also be decoded and can be used in the policy front and other sectors which relies on causal inference in making the decision. 
References:
1)	Karen Simonyan and Andrew Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” arXiv (2014), https://arxiv.org/abs/1409.1556.
2)	Hulstaert, L., & Hulstaert, L. (2018, July 11). Understanding model predictions with LIME. Retrieved from https://towardsdatascience.com/understanding-model-predictions-with-lime-a582fdff3a3b
3)	Molnar, C. (2019, April 12). Interpretable Machine Learning. Retrieved from https://christophm.github.io/interpretable-ml-book/feature-importance.html

4) 	Zeiler M.D., Fergus R. (2014) Visualizing and Understanding Convolutional Networks. In: Fleet D., Pajdla T., Schiele B., Tuytelaars T. (eds) Computer Vision – ECCV 2014. ECCV 2014. Lecture Notes in Computer Science, vol 8689. Springer, Cham
