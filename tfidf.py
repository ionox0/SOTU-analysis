
# coding: utf-8

# In[38]:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[17]:

f = open('State of the Union Addresses 1970-2016.txt')
lines = f.readlines()
bigline = " ".join(lines)
stars = bigline.split('***')
splits = [s.split('\r\n') for s in stars[1:]]
tups = [(s[2], s[3], s[4], "".join(s[5:])) for s in splits]
speech_df = pd.DataFrame(tups)


# In[69]:

X = vectorizer.fit_transform(speech_df[3])
print(speech_df)


# In[63]:

vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(speech_df[3])
# X = pd.DataFrame(data=X)


# In[68]:

# print(vectorizer.get_stop_words())
# print(vectorizer.get_feature_names())
print(vectorizer.inverse_transform(vectorizer))


# In[71]:

from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np



labels = speech_df[1]
true_k = np.unique(speech_df[1]).shape[0]

vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                             min_df=2, stop_words='english',
                             use_idf=opts.use_idf)

X = vectorizer.fit_transform(dataset.data)
km = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1, verbose=opts.verbose)
 
print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()


# In[ ]:



