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

dataset.info()

dataset.describe()

dataset.columns

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

dataset_age_group

"""Sprawdzanie zależności"""

columns = ['age_first_funding_year',
           'age_last_funding_year',
           'age_first_milestone_year',
           'age_last_milestone_year'
           ]

our_functions.scatter_all(columns, dataset)

columns = ['relationships', 'funding_rounds', 'milestones']

our_functions.scatter_all(columns, dataset)

columns = ['age_first_funding_year',
           'age_last_funding_year',
           'age_first_milestone_year',
           'age_last_milestone_year',
           'relationships',
           'funding_rounds',
           'milestones'
           ]

our_functions.bar_single(columns, dataset)

columns = ['is_software',
           'is_web',
           'is_mobile',
           'is_enterprise',
           'is_advertising',
           'is_gamesvideo',
           'is_ecommerce',
           'is_biotech',
           'is_consulting',
           'is_othercategory'
           ]
our_functions.bar_categorical(columns, dataset)

dataset.skew(axis=0, skipna=True)


corr = dataset.corr(method='spearman')
plt.subplots(figsize=(18, 18))
sns.heatmap(corr, annot=True, linewidth=0.5, fmt='.1f')

sns.boxplot(y=dataset["funding_total_usd"])

dataset['category_code'].value_counts()

dataset['category_code'].unique()

dataset2 = dataset.copy()
a = dataset['category_code'].unique()
plt.figure(figsize=(25, 10))
x = []
y = []
for i in a:
  x.append(i)
  l1 = dataset[dataset['category_code'] == i]['labels'].sum()
  l2 = len(dataset[dataset['labels']==1])
  y.append(l1/l2)
plt.bar(x, y)
plt.xticks(rotation=90)
plt.legend()
plt.show()
#
our_functions.bar_single(['category_code'], dataset)

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
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=96)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
print('Decision Tree 1')
y_pred = clf.predict(X_val)
print('f1-',f1_score(y_val, y_pred))
print('accuarcy-', accuracy_score(y_val, y_pred))
print('precision-', precision_score(y_val, y_pred))
print('recall-', recall_score(y_val, y_pred))

data = {'age': [5],
        'age_first_funding_year': [1],
        'age_last_funding_year': [5],
        'age_first_milestone_year': [1],
        'age_last_milestone_year': [1],
        'relationships': [1],
        'funding_rounds': [0],
        'funding_total_usd': [0],
        'milestones': [1],
        'is_software': [1],
        'is_web': [0],
        'is_mobile': [0],
        'is_enterprise': [0],
        'is_advertising': [0],
        'is_gamesvideo': [0],
        'is_ecommerce': [0],
        'is_biotech': [0],
        'is_consulting': [0],
        'is_othercategory': [0],
        'has_VC': [1],
        'has_angel': [0],
        'has_roundA': [1],
        'has_roundB': [0],
        'has_roundC': [0],
        'has_roundD': [0],
        'avg_participants': [1],
        'is_top500': [0]
        }
y_pred = clf.predict(pd.DataFrame.from_dict(data))
print(y_pred)

from sklearn.model_selection import GridSearchCV
parameters = {'criterion': ['gini'],
              'splitter':['best'],
              'max_depth': [5],
              'min_samples_split': [2],
              'min_samples_leaf': [2],
              'min_weight_fraction_leaf': [0.0],
              'max_features': ['log2'],
              'random_state': [1],
              'max_leaf_nodes': [25],
              'min_impurity_decrease': [0.0]
              }
dtc = tree.DecisionTreeClassifier()
clf = GridSearchCV(dtc, parameters)
clf.fit(X_train, y_train)
clf.best_params_

y_pred = clf.predict(X_val)
print('Decision Tree 2')
print('f1-',f1_score(y_val, y_pred))
print('accuarcy-', accuracy_score(y_val, y_pred))
print('precision-', precision_score(y_val, y_pred))
print('recall-', recall_score(y_val, y_pred))

y_pred = clf.predict(pd.DataFrame.from_dict(data))
print(y_pred)

from sklearn.ensemble import RandomForestClassifier
parameters = {'n_estimators': [140],
              'criterion': ['gini'],
              'max_depth': [9],
              'min_samples_split': [2],
              'min_samples_leaf': [2],
              'min_weight_fraction_leaf': [0.0],
              'random_state': [1],
              'max_features': [None],
              'max_leaf_nodes': [20],
              }
dtc = RandomForestClassifier()
clf = GridSearchCV(dtc, parameters)
clf.fit(X_train, y_train)
clf.best_params_

y_pred = clf.predict(X_val)
print('Random Forest 1')
print('f1-',f1_score(y_val, y_pred))
print('accuarcy-', accuracy_score(y_val, y_pred))
print('precision-', precision_score(y_val, y_pred))
print('recall-', recall_score(y_val, y_pred))

data = {'age': [0],
        'age_first_funding_year': [0],
        'age_last_funding_year': [0],
        'age_first_milestone_year': [0],
        'age_last_milestone_year': [0],
        'relationships': [0],
        'funding_rounds': [0],
        'funding_total_usd': [0],
        'milestones': [0],
        'is_software': [0],
        'is_web': [0],
        'is_mobile': [0],
        'is_enterprise': [0],
        'is_advertising': [0],
        'is_gamesvideo': [0],
        'is_ecommerce': [0],
        'is_biotech': [0],
        'is_consulting': [0],
        'is_othercategory': [0],
        'has_VC': [0],
        'has_angel': [0],
        'has_roundA': [0],
        'has_roundB': [0],
        'has_roundC': [0],
        'has_roundD': [0],
        'avg_participants': [0],
        'is_top500': [0]
        }

rfc = RandomForestClassifier(criterion='gini',
                             max_depth=9,
                             max_features=None,
                             max_leaf_nodes=20,
                             min_samples_leaf=2,
                             min_samples_split=2,
                             min_weight_fraction_leaf=0.0,
                             n_estimators=140,
                             random_state=1)
rfc.fit(X_train, y_train)
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(X_val)
# print(f'Shape: {shap_values.hape}')
# pd.DataFrame(shap_values).head(3)

y_pred = rfc.predict(X_val)
print('Random Forest 2')
print('f1-', f1_score(y_val, y_pred))
print('accuarcy-', accuracy_score(y_val, y_pred))
print('precision-', precision_score(y_val, y_pred))
print('recall-', recall_score(y_val, y_pred))

y_pred_test = rfc.predict(X_test)
print('Random Forest 3')
print('f1-', f1_score(y_test, y_pred_test))
print('accuarcy-', accuracy_score(y_test, y_pred_test))
print('precision-', precision_score(y_test, y_pred_test))
print('recall-', recall_score(y_test, y_pred_test))

# shap.summary_plot(shap_values[1], X_val, plot_type='bar')

# shap.summary_plot(shap_values[1], X_val)

shap.dependence_plot('age', shap_values[1], X_val)

explainer.expected_value

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][7], X_val.iloc[7])

sum = 0
for x, y in zip(y_val, y_pred):
    if x == y:
        sum += 1
print(len(y_val), sum)

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_val)

"""#GradienBoosting"""

from sklearn.ensemble import GradientBoostingClassifier

parameters = {'loss': ['deviance'],
              'random_state': [1],
              'learning_rate': [0.1],
              'n_estimators': [100],
              'subsample': [0.8],
              'criterion': ['friedman_mse'],
              'min_samples_split': [2],
              'min_samples_leaf': [1],
              'min_weight_fraction_leaf': [0.0],
              'max_depth': [3],
              'max_features': ['auto'],
              'max_leaf_nodes': [8],

              }
gbc = GradientBoostingClassifier()
clf = GridSearchCV(gbc, parameters)
clf.fit(X_train, y_train)
clf.best_params_

y_pred = clf.predict(X_val)
print('Gradient Boosting 1')
print('f1-', f1_score(y_val, y_pred))
print('accuarcy-', accuracy_score(y_val, y_pred))
print('precision-', precision_score(y_val, y_pred))
print('recall-', recall_score(y_val, y_pred))

gbc = GradientBoostingClassifier(loss='deviance',
                                 random_state=1,
                                 learning_rate=0.1,
                                 n_estimators=100,
                                 subsample=0.8,
                                 criterion='friedman_mse',
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_depth=3,
                                 max_features='auto',
                                 max_leaf_nodes=8)
gbc.fit(X_train, y_train)
y_pred_test = gbc.predict(X_test)
print('Gradient Boosting 2')
print('f1-', f1_score(y_test, y_pred_test))
print('accuarcy-', accuracy_score(y_test, y_pred_test))
print('precision-', precision_score(y_test, y_pred_test))
print('recall-', recall_score(y_test, y_pred_test))

gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_val)

explainer = shap.TreeExplainer(gbc)
shap_values = explainer.shap_values(X_val)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_val)

shap.summary_plot(shap_values, X_val, plot_type='bar')

# shap.summary_plot(shap_values, X_val)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[5], X_val.iloc[5])

gbc.predict_proba(pd.DataFrame.from_dict(data))

gbc.classes_

gbc.predict_proba(X_val)

gbc.predict(X_val)

y_val


p.dump( gbc.fit(X_train, y_train), open( "startup_prediction_saved.p", "wb" ) )