import numpy as np 
from kernel_svm import kernel_svm_train, kernel_svm_predict

class SvmMultipleLabels:

    def __init__(self, params={'kernel': 'linear', 'C': 1}):
        self.params = params
        self.classifiers = []


    def fit(self, X, y):
        unique_labels = np.unique(y)
        for unique_label in unique_labels:
            y_ = np.where(y == unique_label, 1, -1)

            new_classifier = kernel_svm_train(X, y_, self.params)

            self.classifiers.append(new_classifier)


    def predict(self, X):
        predictions = []
        for classifier in self.classifiers:
            _, scores = kernel_svm_predict(X, classifier)
            predictions.append(scores)
        predictions = np.array(predictions)
        predictions = np.argmax(predictions, axis=0)
        return predictions