#Naive Bayes Classifier written for Machine Learning Project, Spring 2021
#Author: Adam Rankin
#Teammates: Austin Fuller and Carter Poythress
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

def naive_bayes_train(train_data, train_labels):
    labels = np.unique(train_labels)
    n, d = train_data.shape
    
    priors = dict()
    gaussian_distribution = dict()

    #
    # Iterate through labels and calculate the gaussian distribution and variance
    # as well as the prior probabilities
    #
    for c in labels:
        x = train_data[train_labels == c]
        gaussian_distribution[c] = {
            'mean': x.mean(axis=0),
            'variance': x.var(axis=0) + 1e-2,
        }
        priors[c] = float(len(train_labels[train_labels==c])) / len(train_labels)
    
    return {'priors': priors, 'gaussian_dist': gaussian_distribution}


def naive_bayes_predict(test_data, model):
    n, d = test_data.shape
    t = len(model['gaussian_dist'])
    predictions = np.zeros((n, t))
    
    #
    # Calculate gaussian probability distribution for each data point with the model means and variance to make a prediction
    #
    for c, g in model['gaussian_dist'].items():
        mean, variance = g['mean'], g['variance']
        predictions[:,c] = mvn.logpdf(test_data, mean=mean, cov=variance) + np.log(model['priors'][c])
    return np.argmax(predictions, axis=1)