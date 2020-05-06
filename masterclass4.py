import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

bank_data = pd.read_csv('', sep=';')

bank_data2= bank_data.copy()

bank_data2.info()

bank_data2['y new'] = bank_data2.y.map(dict(yes=1, no=0));

bank_data2.head()

bank_data2.describe()

#median is greater than mean -> there is a skew in the data
bank_data2.age.median()

plt.style.use('seaborn-whitegrid')
bank_data2.hist(bins=20, figsize=(14,10))
plt.show()

labels = 'Did not open term', 'opened term'

fig, ax = plot.subplots(1,2, figsize=(16,8))

bank_data2['y'].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, labels=labels, fontsize=12, startangle=135)
plt.subtitle('Information on term subscriptions', fontsize=20)
df = bank_data2.groupby(['education', 'y']).size().groupby(level=0).apply(lambda x: x/bank_data2.shape[0]).unstack().plot(kind='bar', ax=ax[1], stacked=True)
ax[1].set(ylabel='Percentage of term openers by level of education')
ax[1].set(xlable='Education level')
ax[1].legend(['Did not open', 'open'])

fig, ax = plt.subplots(1, 2, figsize=(16,8))
plt.subtitle('Information on Term Subscription 2', fontsize=20)
df = bank_data2.groupby(['age', 'y']).size().groupby(level=0, squeeze=True).apply(lambda x: x/bank_rate2.shape[0]).unstack().plot(kind='bar', ax=ax[0], stacked=True)

ax[0].set(ylabel='Percentage of term openers by age')
ax[0].set(xlabel='Age')
ax[0].locator_params(axis='x', nbins=60)
ax[0].legend(['Did not open', 'Open'])

dfl = bank_data2.groupby(['marital', 'y']).size().groupby(level=0).apply(lambda x: x/bank_data2.shape[0]).unstack().plot(kind='bar', ax=ax[1])

ax[1].set(ylabel='Percentage of term openers by marriage status')
ax[1].set(xlabel='Marriage status')
ax[1].legend(['Did not open', 'Open'])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

dependent_var = bank_data2['y new']
encoded_df = bank_data2.copy()

encoded_df = encoded_df.drop(['y', 'y new'], axis = 1)

encoded_df = encoded_df.apply(encoder.fit_transform)

encoded_df.head()

encoded_df.describe()

#lets make a correlation matrix
encoded_df = pd.concat([encoded_df, dependent_var], axis=1)

plt.figure(figsize=(12,10))
cor = encoded_df2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cn.Reds)
plt.show()

cor_target = abs(cor["y new"])

relevant_features = cor_target[cor_target > 0.2]
print(relevant_features, '\n')

feature_cors = encoded_df[['duration', 'pdays']].corr()
feature_cors1 = encoded_df[['euribor3m', 'nr.employed']].corr()
feature_cors2 = encoded_df[['job', 'marital']].corr()
feature_cors3 = encoded_df[['marital', 'age']].corr()
print(feature_cors, '\n', feature_cors1, '\n', feature_cors2, '\n', feature_cors3)

#I chose to drop: age, housing, loan, default, day of the week
encoded_df = encoded_df.drop(['age', 'housing', 'loan', 'default', 'day_of_week'], axis = 1)

encoded_df.describe()

print(bank_data2.shape)

train_len = int(.8 * (bank_data2.shape[0]))

train_x, train_y = encoded_df[:train_len], dependent_var[:train_len]
test_x, test_y = encoded_df[train_len:], dependent_var[train_len:]

from sklearn.linear_model import LogistricRegression

log_reg = LogistricRegression(solver='lbfgs', penalty='12', max_iter=10000)

log_reg_trained = log_reg.fit(train_x, train_y)

print(f'training accuracy is: {log_reg.score(train_x, train_y)}')

pred = log_reg.predict(test_x)

print(f'test accuracy is {sk.metrics.accuracy_score(test_y, pred)}')

log_reg.coef

from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()

tree_model = dec_tree.fit(train_x, train_y)

print(f'training accuracy for decision tree is: {dec_tree.score(train_x, train_y)}')

pred1 = dec_tree.predict(test_x)

print(f'training accuracy for decision tree is: {sk.metrics.accuracy_score(test_y, pred1)}')

from aix360.algorithms.lime import LimeTabularExplainer

class_names = [0, 1]
log_lime_explainer = LimeTabularExplainer(train_x.values, class_names=class_names, feature_names=train_x.columns)

print(f'the predicted class is: {log_reg_trained.predict_proba([train_x.values[0]])}')
print(f'the true class is: {train_y.loc[0]}')

idx = 1220
exp_log = log_lime_explainer.explain_instance(train_x.values[idx], log_reg_trained.predict_proba, num_features=6, labels=class_names)

print('Explanation for class %s' %class_names[0])
print('\n'.join(map(str, exp_log.as_list(label=0))))

print('Explanation for class %s' %class_names[1])
print('\n'.join(map(str, exp_log.as_list(label=1))))

log_reg_exp.show_in_notebook()

exp_log.show_in_notebook()

from aix360.metrics import faithfulness_metric, monotonicity_metric

predicted_class = log_reg.predict(test_x.values[0].reshape(1, -1))[0]

le = exp_log.local_exp[predicted_class]

m = exp_log.as_map()

x = test_x.values[0]

coefs = np.zeros(x.shape[0])

for v in le:
	coefs[v[0]] = v[1]

base = np.zeros(x.shape[0])

print('Faithfulness:', faithfulness_metric(log_reg, x, coefs, base))
print('Monotonicity:', monotonicity_metric(log_reg, x, coefs, base))



from lale.lib.sklearn import PCA
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import KNeighborsClassifier as Kn
from lale.lib.sklearn import DecisionTreeClassifier as tree
from lale.lib.sklearn import SVC
from lale.lib.sklearn import RandomForestClassifier as rf
from lale.lib.sklearn import XGBClassifier

from lale.lib.sklearn import NoOp, ConcatFeatures

import lale
import lale.helpers
import lale.operators
from lale.lib.lale import Hyperopt
lale.wrap_import_operators()

import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

explainable_pipe = lale.operators.make_union(PCA, NoOp) >> (LogistricRegression | Kn | tree)

explainable_train = explainable_pipe.auto_configure(train_x, train_y, optimizer= Hyperopt, cv= 3, max_evals= 3, scoring= 'accuracy')

explainable_train.visualize()

explainable_train.pretty_print(show_imports= False, ipython_display= True)

print(f'the accuracy of this pipeline is {sklearn.metrics.accuracy_score(test_y, explainable_train.predict(test_x))}')

import aix360
from aix360.algorithms.lime import LimeTabularExplainer

print(f'the predicted class is: {explainable_train.predict([train_x.values[0]])}')

print(f'the true class is {train_y.loc[0]}')

idx = 10

explainable_exp = limeexplainer.explain_instance(train_x.values[idx], explainable_train.predict_proba, num_features= 5, labels= [0, 1])


print('Explanation for class %s' %class_names[0])
print('\n'.join(map(str, explainable_exp.as_list(label=0))))

print('Explanation for class %s' %class_names[1])
print('\n'.join(map(str, explainable_exp.as_list(label=1))))
