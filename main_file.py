# -*- coding: utf-8 -*-

### Import bibliotek ###
# manimpulacja danymi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as p
import seaborn as sns
import shap

# import warnings
# warnings.simplefilter("ignore")

# nasze funkcje
import our_functions


dataset = pd.read_csv('startup data.csv')

"""Usuwanie zbędnych kolumn"""

dataset=dataset.drop_duplicates(subset=['name'])
dataset = dataset.drop(['Unnamed: 0', 'Unnamed: 6','latitude', 'longitude', 'state_code', 'state_code.1', 'zip_code', 'object_id'], axis = 1)

"""Wyliczenie czasu trwania projektów w latach"""

time_columns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
for column in time_columns:
  dataset[column] = pd.to_datetime(dataset[column])

dataset['closed_date'] = dataset['closed_at']
dataset['closed_date'] = dataset['closed_date'].fillna('2013-12-31')

dataset['age'] = dataset['closed_date'] - dataset['founded_at']

dataset["age"] = round(dataset['age']/np.timedelta64(1,'Y'))

"""Usuwanie wierszy z wartościami 0"""

time_columns = ['age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 'age_last_milestone_year', 'age']
for column in time_columns:
  dataset = dataset.drop(dataset[dataset[column] < 0].index)

"""Wypełnianie wartości NaN"""

dataset['age_first_milestone_year'] = dataset['age_first_milestone_year'].fillna(0)
dataset['age_last_milestone_year'] = dataset['age_last_milestone_year'].fillna(0)

dataset_age_group = dataset[dataset['labels'] == 1].groupby(['age']).agg({'labels' : 'count'}).reset_index()
dataset_age_group.columns = ['age', 'total_succes']

dataset_age_group_total = dataset.groupby(['age']).agg({'labels' : 'count'}).reset_index()
dataset_age_group_total.columns = ['age', 'total']

dataset_age_group = dataset_age_group.merge(dataset_age_group_total, on='age')
dataset_age_group['succes_rate'] = round((dataset_age_group['total_succes']/dataset_age_group['total'])*100, 2)

"""Sprawdzanie zależności"""

# columns = ['age_first_funding_year',
#            'age_last_funding_year',
#            'age_first_milestone_year',
#            'age_last_milestone_year'
#            ]
#
# our_functions.scatter_all(columns, dataset)
#
# columns = ['relationships', 'funding_rounds', 'milestones']
#
# our_functions.scatter_all(columns, dataset)
#
# columns = ['age_first_funding_year',
#            'age_last_funding_year',
#            'age_first_milestone_year',
#            'age_last_milestone_year',
#            'relationships',
#            'funding_rounds',
#            'milestones'
#            ]
#
# our_functions.bar_single(columns, dataset)
#
# columns = ['is_software',
#            'is_web',
#            'is_mobile',
#            'is_enterprise',
#            'is_advertising',
#            'is_gamesvideo',
#            'is_ecommerce',
#            'is_biotech',
#            'is_consulting',
#            'is_othercategory'
#            ]
# our_functions.bar_categorical(columns, dataset)
#
# dataset.skew(axis=0, skipna=True)
#
#
# corr = dataset.corr(method='spearman')
# plt.subplots(figsize=(18, 18))
# sns.heatmap(corr, annot=True, linewidth=0.5, fmt='.1f')
#
# sns.boxplot(y=dataset["funding_total_usd"])
#
# dataset['category_code'].value_counts()
#
# dataset['category_code'].unique()
#
# dataset2 = dataset.copy()
# a = dataset['category_code'].unique()
# plt.figure(figsize=(25, 10))
# x = []
# y = []
# for i in a:
#   x.append(i)
#   l1 = dataset[dataset['category_code'] == i]['labels'].sum()
#   l2 = len(dataset[dataset['labels']==1])
#   y.append(l1/l2)
# plt.bar(x, y)
# plt.xticks(rotation=90)
# plt.legend()
# plt.show()
# #
# our_functions.bar_single(['category_code'], dataset)


X = dataset[['age',
             'age_first_funding_year',
             'age_last_funding_year',
             'age_first_milestone_year',
             'age_last_milestone_year',
             'relationships',
             'funding_rounds',
             'funding_total_usd',
             'milestones',
             'is_software',
             'is_web',
             'is_mobile',
             'is_enterprise',
             'is_advertising',
             'is_gamesvideo',
             'is_ecommerce',
             'is_biotech',
             'is_consulting',
             'is_othercategory',
             'has_VC',
             'has_angel',
             'has_roundA',
             'has_roundB',
             'has_roundC',
             'has_roundD',
             'avg_participants',
             'is_top500']
            ]
y = dataset['labels']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=68)

from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
parameters = {'n_estimators': [50, 100, 150,],
              'max_depth': [2, 4, 10, 12, 14],
              'min_samples_split': [8, 12, 20],
              'min_samples_leaf': [2, 4, 8],
              }
cv_stratify = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 3)
rfc = GridSearchCV(RandomForestClassifier(random_state = 2), parameters, cv = cv_stratify, scoring = 'f1')
rfc.fit(X_train, y_train)
print(rfc.best_params_)
print(rfc.best_score_)

explainer = shap.TreeExplainer(rfc.best_estimator_)
shap_values = explainer.shap_values(X_test)
# print(f'Shape: {shap_values.shape}')
# pd.DataFrame(shap_values).head(3)

print('RandomForest test')
y_pred = rfc.best_estimator_.predict(X_test)
print('f1-',f1_score(y_test, y_pred))
print('accuarcy-', accuracy_score(y_test, y_pred))
print('precision-', precision_score(y_test, y_pred))
print('recall-', recall_score(y_test, y_pred))

shap.summary_plot(shap_values[1], X_test, plot_type='bar')

shap.summary_plot(shap_values[1], X_test)

#shap.dependence_plot('age', shap_values[1], X_val)

explainer.expected_value

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test)

"""#GradienBoosting"""

from sklearn.ensemble import GradientBoostingClassifier
parameters = {'learning_rate' : [0.1, 0.2, 0.4],
              'n_estimators': [50, 100, 150],
              'max_depth': [2, 4, 8],
              'min_samples_split': [8, 12, 20],
              'min_samples_leaf': [2, 4],
              }
gbc = GridSearchCV(GradientBoostingClassifier(random_state = 2), parameters, cv = cv_stratify, scoring = 'f1')
gbc.fit(X_train, y_train)
print(gbc.best_params_)
print(gbc.best_score_)

y_pred = gbc.best_estimator_.predict(X_test)
print('Gradient Boosting test')
print('f1-', f1_score(y_test, y_pred))
print('accuarcy-', accuracy_score(y_test, y_pred))
print('precision-', precision_score(y_test, y_pred))
print('recall-', recall_score(y_test, y_pred))

explainer = shap.TreeExplainer(gbc.best_estimator_)
shap_values = explainer.shap_values(X_test)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_test)

shap.summary_plot(shap_values, X_test, plot_type='bar')

#shap.summary_plot(shap_values, X_val)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[5], X_test.iloc[5])

gbc.predict_proba(X_test)

gbc.predict(X_test)

p.dump([rfc.best_estimator_, gbc.best_estimator_], open("startup_prediction_saved.p", "wb"))