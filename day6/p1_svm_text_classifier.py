import csv
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

data = np.array(list(csv.reader(open('spam.csv'))))
act_target = data[:,0]
text = data[:, 1]

spam_t = np.where(act_target == 'spam', 1, -1)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(text)
print(count_vect.get_feature_names())

tfid_transformer = TfidfTransformer()
X_train_tfidf = tfid_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)