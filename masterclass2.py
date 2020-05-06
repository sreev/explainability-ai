import pandas as pd
import sklearn as sk
import lale

cal_housing = sk.datasets.fetch_california_housing()
x = pd.DataFrame(call_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

pd.concat([x.head(), pd.DataFrame(y).head(1)], axis = 1)

from sklearn.preprocessing import Normalizer
from sklearn.tree import DecitionTreeRegressor as Tree
from lale.lib.lale import Hyperopt
import lale.helpers
lale.wrap_imported_operators()

tree_plan = Normalizer >> Tree

tree_plan.visualize()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2, random_state=0)

tree_trained = tree_plan.auto_configure(train_x, train_y, optimizer=Hyperopt, cv = 3, max_evals=10, scoring ='r2')

#lets look at our hyperparameters
tree_trained.pretty_print(ipython_display=True, show_imports=False)

tree_trained.visualize()

import sklearn.metrics
predicted = tree_trained.predict(test_x)
print(f'R2 score {sklearn.metrics.r2_score(test_y, predicted):.2f}')
