from __future__ import print_function
import sklearn
import numpy as np
import sklearn.ensemble
import sklearn.metrics

from aix360.algorithms.lime import LimeTextExplainer

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.'))[-2:] for x in newsgroups_train.target_names]

class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

print(','.join(class_names))

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.fit(newsgroups_test.data)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=.01)

nb.fit(train_vectors, newsgroups_train.target)

pred=nb.predict(test_vectors)

sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')

#lets do the explanation
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(vectorizer, nb)

print(pipe.predict_proba([newsgroups_train.data[0]]).round(3)

lime_explainer = LimeTextExplainer(class_names = class_names)

print(type(lime_explainer))

idx = 1340

explanation = lime_explainer.explain_instance(newsgroups_test.data[idx], pipe.predict_proba, num_features = 6, labels = [0, 17])

print(f'document id: {idx}')
print(f'predicted class = {class_names[nb.predict(test_vectors[idx]).reshape(1, -1)[0,0]]}')

print(f'true class: {class_names[newsgroups_test.target[idx]]}')

print(f'explanation for class {class_names[0]}')
print('\n'.join(map(str, explanation.as_list(label=0))))

print(f'explanation for class {class_names[17]}')
print('\n'.join(map(str, explanation.as_list(label=17))))

explanation.show_in_notebook(text=False)
