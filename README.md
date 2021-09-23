# Feasibility study: automatic categorization of products

This project aims at assessing the feasibility of an automatic categorization for a catalog of products of an e-commerce
site. We assess its feasibility by clustering the products and see can find the natural partition of categories in the 
clustering. We will measure how the partitions match using ARI.

We have two types of information per product to apply a clustering algorithm on: a text description and an image. 
We first try clustering the products using their vectorized text descriptions. In a second phase, we combine the vectorized 
descriptions with vectorized images to see if it improves the quality of the clustering (ARI).

We will compare the partitions (compute ARI score) of clustered vectors with two category levels:
* cat_0: the highest-level category of the catalog: 7 categories with 150 products each
* cat: the lowest-level category containing at least 20 products. 27 categories with between 21 and 127 products
<p align="center">
  <img width="600" src="https://github.com/vpvinc/p6_products_classification/blob/assets/cat_0_dis.PNG?raw=true" />
</p>
<p align="center">
  <img width="600" src="https://github.com/vpvinc/p6_products_classification/blob/assets/cat_dis.PNG?raw=true" />
</p>

## Table of content

1. Structure of the project
2. Vectorization of product descriptions
3. Evaluation of clustering these vectors using ARI
4. Vectorization of images
5. Evaluation of clustering enriched vectors using ARI

## 1. Structure of the project

**This project articulates around 3 files:**

- P6: (comments in French) notebook containing the whole script of the project:
  * Data import and cleaning
  * test of different vectorization techniques
  * dimensionality reduction
  * application of clustering algorithm
  * graphic representation of clustering scores (ARI)
- environment.yml: file to set up dependencies with conda
- requirements.txt: file to set up dependencies with pip

## 2. Vectorization of product descriptions

Several vectorization techniques are considered from most basic to recent and complex. A graph is displayed to illustrate 
the technique
1) **Bag of words (BOW)**

The Simplest vectorization technique. Each doc (description) is turned into a vector as long as the number of unique 
tokens all documents considered (4298 for this project)

<p align="center">
  <img width="300" src="https://github.com/vpvinc/p6_products_classification/blob/assets/bow.png?raw=true" />
</p>

2) **TF-IDF**

TF-IDF works like BOW but also take into account the frequency of a token in all the documents. Indeed, a token that is 
present in all documents will not help distinguish descriptions, the weight of too-frequent tokens are consequently decreased.
TF-IDF is known to perform well for topic-based classification

<p align="center">
  <img width="500" src="https://github.com/vpvinc/p6_products_classification/blob/assets/tfidf.PNG?raw=true" />
</p>

source [blog.octo.com](https://blog.octo.com/apprentissage-distribue-avec-spark/)

3) **embedding using Word2Vec(skipgram)**

Skipgram is a neural network that takes a one-hot vector representing a word as input, and predict the nearby words as output.

Unlike BOW and TF-IDF, skipgram enables to obtain dense vectors and take into account the semantic relationships between tokens.

First, we use the skipgram network to compute a dense vector for each token with the connection weights.
Once we have a dense vector for each token, we multiply the vector by the inverse document frequency (IDF) of the token
to make the vector more informative.
At last, to obtain the vector of the doc, we average the vectors of the tokens in the doc.

<p align="center">
  <img width="500" src="https://github.com/vpvinc/p6_products_classification/blob/assets/skipgram.png?raw=true" />
</p>

source [McCormick, C. (2016, April 19). Word2Vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

4) **Universal Sentence Encoder (Deep Averaging Network) from Tensforflow**

Universal Sentence Encoder is an encoder that converts a doc into a 512-length dense vector. The weights used to encode
the doc are taken from an optimized Deep Averaging Network trained on a variety of NLP tasks with a very large corpus. 

paper: [Daniel Cer & al, 2018](https://arxiv.org/pdf/1803.11175.pdf)

<p align="center">
  <img width="500" src="https://github.com/vpvinc/p6_products_classification/blob/assets/DAN.png?raw=true" />
</p>

source [www.weak-learner.com](https://www.weak-learner.com/blog/2019/07/31/deep-averaging-networks/)

## 3. Evaluation of clustering these vectors using ARI
We now cluster the vectorized data using K-mean with K equal to the cardinality of the respective category levels: K=7 for cat_0
and K=27 for cat.
Then, we calculate the ARI score between the two partitions: on one hand the partition by original categories (cat_0 or cat)
and on the other hand the partition by K-mean. 

As said in introduction, the idea is to assess whether we can find a natural partition by category in the descriptions.

Here below are the ARI-scores for the different vectorization techniques between the partition by K-mean and original
categories:

<p align="center">
  <img width="800" src="https://github.com/vpvinc/p6_products_classification/blob/assets/ARI_text.png?raw=true" />
</p>

The two vectorization that enable us to find the partition that is closest to the original categories are respectively USE
for cat_0 and tf-idf for cat

## 4. Vectorization of images

We have obtained an ARI between 0.35 and 0.4 with descriptions, which is not a perfect match (1) but encouraging. Let's see 
if we can drive it up by enriching these vectors with the vectorized images.

1) **Bag Of Visual Words and ORB**

Oriented FAST and Rotated BRIEF (ORB) was developed at OpenCV labs by Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary R. Bradski in 2011

It is a feature detection extraction technique that detects and extracts features as vectors from an image. Such a vector is 
called a descriptor. As a reminder, a feature is a local area of an image that helps distinguish between one image and 
another. In ORB context, a feature is defined as local (part of the image), repeatable (invariant to transformations) 
and distinctive (non-ambiguous with respect to other parts of the image).

Steps to vectorize the images:
1) detection and extraction of features for each image using ORB
2) clustering of all the descriptors (of all the images) using MiniBatchKMeans with K equal to the square root of the total
number of descriptors
3) Make a histogram per image counting the number of descriptors in each cluster. We end up with a table where images are
the rows, the clusters found in 2) are the columns and the values are the number of descriptors per cluster per image

visualisation of these three steps:

<p align="center">
  <img width="811" src="https://github.com/vpvinc/p6_products_classification/blob/assets/bovw.PNG?raw=true" />
</p>

2) **CNN: transfer-learning from VGG16**

Convolutional Neural Networks (CNN) have outclassed techniques such as ORB and SIFT in recent years. However, neural networks
do need large amounts of data to be trained. This is where transfer learning is very helpful. We can use VGG16, a network trained
on a dataset of over 14 million images belonging to 1000 classes without having to train it. We assume that this model is
capable of distinguishing our images as is. 

Architecture of VGG16:

    #     ....
    #     block5_conv4 (Conv2D)        (None, 15, 15, 512)       2359808
    #     _________________________________________________________________
    #     block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
    #     _________________________________________________________________
    #     flatten (Flatten)            (None, 25088)             0
    #     _________________________________________________________________
    #     fc1 (Dense)                  (None, 4096)              102764544
    #     _________________________________________________________________
    #     fc2 (Dense)                  (None, 4096)              16781312
    #     _________________________________________________________________
    #     predictions (Dense)          (None, 1000)              4097000

We will use the model up to the second fully connected layer (fc2). Therefore, by feeding the model with a 224 × 224 RGB image, 
we obtain a 4096-length vector, such that similar images will have similar vectors and conversely.

source  [Karen Simonyan, Andrew Zisserman, 2014](https://arxiv.org/pdf/1409.1556.pdf)

## 5. Evaluation of clustering enriched vectors using ARI

From the vectors of the images we found in 4., let's keep only the principal components that explain up to 99% of the variation. Let's 
now concatenate these components to the best textual vectors we found in 3. Does the ARI get better ?

<p align="center">
  <img width="811" src="https://github.com/vpvinc/p6_products_classification/blob/assets/ARI_text_cnn.png?raw=true" />
</p>

We can see from the above graph that the components from the ORB vectorization make the ARI worse both for tf_idf and USE
whereas the components from the CNN vectorization improve the ARI when combined with USE for cat_0. However, CNN components 
make the ARI worse when combined with TF-IDF for cat.

## Conclusion

It appears we have enough information in the descriptions and the images to consider an automatic categorization, all the 
more so if we consider a supervised classification using this dataset as training data. The vectorization technique will 
depend on the level of category that we want to predict: tf-idf for cat, USE + CNN components for cat_0.

