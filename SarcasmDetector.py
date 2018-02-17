# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:27:05 2017
@author: hammadkhan
"""
#Loading Libraries.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
eng_stopwords = set(stopwords.words("english"))

## Reading all the datasets ##
train_df = pd.read_csv("E:/Seventh Semester/IR&TM/Project/Sarcasm+Detection/Data/reddit_training.csv")
test_df = pd.read_csv("E:/Seventh Semester/IR&TM/Project/Sarcasm+Detection/Data/reddit_test.csv")

#Tfidf features
tfidf_vect = TfidfVectorizer(ngram_range = (1,7),norm='l2',smooth_idf = False , analyzer='char', max_df= 0.30,min_df = 12, stop_words = 'english')
tfidf_features = tfidf_vect.fit_transform(train_df['body'].values.tolist())

#Splitting the dataset into Training set and Testing set.
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(tfidf_features, train_df.sarcasm_tag, train_size=0.7,test_size = 0.3, random_state = 10)

#Ensemble Mean Accuracy =  0.978723404255
clf1 = MultinomialNB(alpha = 0.05)# LogisticRegression(random_state = 10)
clf2 = RandomForestClassifier(n_estimators = 100,criterion = 'entropy', random_state = 0)
clf3 = GradientBoostingClassifier(n_estimators = 100, random_state = 10 )
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
                        voting='hard')
#Ensembling predictions
eclf.fit(X_train_count, y_train_count)
predictions_count = eclf.predict(X_test_count)

#cross validation with kfold = 10
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = eclf, X = X_train_count, y = y_train_count, cv = 10)
print ('Ensemble Max Accuracy',accuracies.max())


### Function to create confusion matrix ###
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Calculating Confusion Matrix
cnf_matrix = confusion_matrix(y_test_count, predictions_count)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix, classes=['NO', 'YES'],
                      title='Confusion matrix')
plt.show()


