# Bee-Wasp Image classification

### Keywords
Image classification, Convolutional Neural Networks (CNN), Bee-wasp, ResNet, MobileNet

## Introduction
Computer Vision, the study of visual data, has proliferated in the past decade. This can be attributed to the development of various deep learning techniques in the machine learning field. Few decades ago, we could see and manipulate image data, but it was not possible to get inference from large sizes of image data due to insufficient resources and algorithms that just didn’t perform well on large and complex datasets. This has changed today with neural networks and powerful local machines that can achieve much better accuracies on ginormous datasets.
One of the fundamental tasks for any labelled data is the classification problem. The classic dataset in computer vision field for classification is the MNIST dataset. Now the question arises, if a model has already performed well on a dataset for classification, why do we need to test it on other datasets (considering we are using the new dataset just for analyzing a models’ performance)? Well, the answer is simple, not every dataset is the same. There was a time when algorithms struggled on the MNIST dataset for number classification. But todays algorithms achieve almost perfect results on its testing data. This does not mean we have found the perfect algorithm for image classification.
Technology has allowed us to collect even more complex and larger datasets, where the deep learning algorithms do achieve respectable results but do not perform exceptionally well. This particularly teaches us a lot about the how the deep neural networks might be interpreting the data and where it might be struggling. That is why in this project, I will try some deep learning techniques on a dataset to help classify between bees and wasps. We will look at what is particularly unique in this dataset, the challenges and issues faced as well as the approaches tried on this dataset.

## Problem Statement
Today with smartphones, it has become extremely easy to take good quality pictures of what’s happening around us. Apart from memories, photos may also be taken due to general curiosity. Services like Google lens assist in such tasks by identifying the object in the picture. It may some grocery item or a tourist place like a monument or even something from nature like insects or flowers. 
This project focuses on the insects, specifically the bees and wasps. A lot of times people are unable to identify or rather differentiate between a bee and a wasp that they might encounter in their neighborhood. Identifying one or the other is quite important because one is a pollinator while the other is a predator. Both have a set of similarities and differences. Bees, especially the honeybees, die after they sting once. But wasps can sting multiple times when they feel threatened. Not only that, but wasps are much more aggressive than bees. Both are usually brightly colored in black and yellow. Also, bees are generally fatter than wasps, which are long and thin.
The goal of this project is to not only confidently identify whether an insect is bee or wasp, but also to do it efficiently such that person can take a photo on his/her smartphone and locally classify the image. That is, the approach should have the right balance between performance and simplicity. The model should also be able to perform well on images that are not cropped to regions where these insects might be present.

## Technical Approach
As seen in the assignments, machine learning methods like k-means and KNN don’t work well for large datasets in classifying the images. Since todays deep neural networks are known to perform well on such datasets, I will be using them for the classification task. Specifically, I have trained three models on the data, the state-of-the-art ResNet model, the MobileNet model as well as a custom CNN model.
The idea is to use the concept of filters in the Convolutional Neural Networks (CNNs) to extract useful information from the images, that is, attributes specific to the bees and wasps. We will start with the complex deep neural network models and then move towards simpler models and analyze how the metrics (accuracies and loss) are affected.
We will also look at how the dataset was collected keeping in mind certain biases that our model might pick up. Also, some interesting features from the dataset will be investigated. Now, of all the three models tried, ResNet, specifically the ResNet50, is the most complex model. This model has around 48 convolutional layers. It can learn better from such a deep network using residual connections. Basically, these connections allow the gradient to flow without vanishing or exploding gradient problems. The ResNet model performs well on a lot of difficult image tasks. But the disadvantage is its complexity, which means more use of resources than others. Therefore, the other model tested is the famous MobileNet model which has a mere 3.5 million trainable parameters. This allows the model to be run on a lot of portable devices, useful for our application. To take things further, we will also try an even simpler network built from two convolutional layers and compare its performance.

## Experimental Setup
In this section we will look at the dataset and the three models in greater detail. In this data, each image pixel value has been scaled between 0 and 1 before training it on the model for normalization. Also, due to the size of the dataset, it is not possible to load the complete dataset onto the memory. Hence, a data-frame is used to assist the model in training. The model will retrieve the path of the requested image from the data-frame. Then this path will be used to read the image from the storage in real-time. This slows down the training but allows for training on local machine with limited main memory. The same method is used for the validation set.
Each model has an optimal learning rate set to 0.0001 and uses a soft-max activation function on the final layer for this classification task. They are trained for a maximum of 50 epochs and an early-stopping callback method is defined to stop the training when the validation loss does not improve over a set number of epochs. When this method is executed, only the best weights found are restored and used. Also, I have used binary cross-entropy to compute the losses.

### Data
 
Figure 1: Dataset size
The dataset consists of four categories, bees, wasps, other insects, and flowers. The bee, wasp and insect images are collected from Kaggle dataset by Callum Robertson and George Ray.1 The authors themselves have collected the data from multiple other Kaggle datasets and Fliker image data. Finally, the flower dataset is collected from VGG at the University of Oxford.2 This leads to a total size of over 11,000 images as shown in figure 1, with an 80-10-10 split for training-validation-testing respectively. The reasons for adding two more categories to the bee-wasp classification task are as follows:
1.	Foreground bias: After training on the bee-wasp dataset, suppose we show the model an image of a fly. Looking at the similar structure and shape, the model might classify it as a bee or a wasp. But this is misleading and that is why it was necessary to add images of tons of other insect species so that the model not only learns the general structure of the insect but also peculiar details of wasps and bees.
2.	Background bias: If the images are closely observed, there are a lot of images with the background of flower, especially for the bee category. We don’t want the model to associate a flower with bees. This may lead to incorrect predictions with just flowers in the image classified as bees. So, a fourth category of flowers is added for more accurate model learning.
 
Figure 2-a: Bees

 
Figure 2-b: Wasps

 
Figure 2-c: Other insects

 
Figure 2-d: Flowers
Figure 2 shows randomly sampled images from the dataset for each label. The bee and wasp categories also contain a lot of low-quality images, which will make our model more robust to zoomed in images taken from smartphones.

### Model
The three models used in this project are discussed in greater detail below. We will look at its architecture and complexity. All models have a pooling layer (either max pooling or average pooling layer) for reducing the dimensions and dropout for reducing over-fitting. Even though pre-trained models have been used in some layers, they are still trained further with the other layers for this dataset.3







1.	https://www.kaggle.com/jerzydziewierz/bee-vs-wasp
2.	https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
3.	https://keras.io/api/applications/
ResNet50 V2
 
Figure 3-a: with ResNet50 V2
In the course, we have seen how increasing the model complexity does not always help achieving better results. This is because the gradient calculated grows exponentially small or large as we go towards the start of the model, which is known as the vanishing or exploding gradient problem. ResNet solved this problem by building skip connections between the layers so that the gradient can flow easily without causing the aforementioned problem. Such a model will tell us what a highly complex model achieves on our current dataset. This also gives us a benchmark for the simpler models discussed later. Figure 3-a shows the general model structure. This model has around 25 million parameters.
MobileNet V2
 
Figure 3-b: with MobileNet V2
Since, we also want to try the simpler models (complexity-wise) on the data, we will try another state-of-the-art model, MobileNet, with just around 3 million parameters. This kind of model is made to achieve good results and perform well on embedded devices.
Custom CNN model
 
Figure 3-c: model with 2 convolutional layers
Finally, to reduce the complexity further, I made a model without any pre-trained layers. It consists of only two 2D convolutional layers. Each convolutional layer is followed by a max-pooling layer to reduce dimensionality and a dropout layer to avoid overfitting. The dropout layers can drop up to 20% of the neurons while training for robustness of the model.
Finally, we have a flatten layer, since the goal is image classification, followed by a dense layer for prediction between the four classes. This approach has just over a million trainable parameters making it significantly less complex than the other three models. For visualization it has only 4% of the parameters in the ResNet model and 33% of the parameters in the MobileNet model.

## Results
In figure 4, the blue line is for the ResNet model, green line is the MobileNet model, and red is the custom CNN model. The dotted lines are the validation accuracies for the same models.
 
Figure 4-a: Training and validation accuracy
From the figures, we can observe that both the pre-trained models have achieved an almost perfect accuracy on the training set and over 90% accuracy on the validation set. The MobileNet architecture has performed as good if not better than the other. This goes to show that it is not important to have a more complex model to get better results on all datasets. MobileNet has the right balance between performance and efficiency among all three.
The third CNN model has also performed well with over 90% of training accuracy and 80% of validation accuracy. This is a respectable result for a model that is a fraction of the size of the other models. Another observation is the number of epochs required for training for each model. Overall, higher the complexity, lesser number of epochs is required for the model to train. So, if one can parallelize the model and requires faster training, they can use such a complex model.
 
Figure 4-b: Training and validation loss
Both the ResNet and MobileNet models have achieved similar results on test data. Around 92%, which is also close to the validation results. ResNet only has a slightly lower loss than MobileNet, but again, MobileNet offers the right balance, atleast for this dataset. Finally, the third models’ around 80% test accuracy is also respectable (though it has much higher loss).
 
Figure 4-c: Testing set metrics

## Conclusions and Future Work
This project presented the uniqueness of the bee-wasp dataset. It showed us the challenges faced, and why it was necessary to expand the dataset into more categories (other insects and flowers). We looked at some of the deep learning approaches after ruling out the naïve machine learning algorithms. We also analyzed the results over changing model complexity. Even though the ResNet model performed the best, MobileNet is the most suitable for this application of classifying insects.
As a future scope, it is possible to analyze the third model in greater detail and try to modify it. The goal will remain the same, improve the performance without increasing the complexity too much. Also, more low-quality images could be collected in the future to study specifically the impact of image quality on the model performance. This project could also be hosted online where people can upload bee and wasp photos to expand the dataset size further.
