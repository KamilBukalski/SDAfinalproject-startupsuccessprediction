# SDAfinalproject-startupsuccessprediction
In this project, we wanted to create a model that will predict the success rate of a startup based on multiple input parameters.

# Data
Source of the data used in this project is a dataset from Kaggle that can be found here:
https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

After initial data exploration and preparation we established a list of important parameters:

| Number | Name                     | Type    | Description                                                                                                                        |
|--------|--------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------|
| 1      | age                      | int     | Amount of years from founded year up to the year of closing (or 2013 as a maximum).                                                |
| 2      | age_first_funding_year   | float   | Age of the company when the it received first funding.                                                                             |
| 3      | age_last_funding_year    | float   | Age of the company when the it received last funding.                                                                              |
| 4      | age_first_milestone_year | float   | Age of the company in years when it reached first milestone.                                                                       |
| 5      | age_last_milestone_year  | float   | Age of the company in years when it reached last milestone.                                                                        |
| 6      | relationships            | int     | Amount of relations with others like accountants, investors, vendors, mentors etc.                                                 |
| 7      | funding_rounds           | int     | How many times was the funding money transferred.                                                                                  |
| 8      | funding_total_usd        | int     | Total amount of money provided to the company through funding (in US dollars)                                                      |
| 9      | milestones               | int     | How many milestones was set up for the startup.                                                                                    |
| 10     | is_software              | boolean | True/False parameter for the company being from 'software' category.                                                               |
| 11     | is_web                   | boolean | True/False parameter for the company being from 'web' category.                                                                    |
| 12     | is_mobile                | boolean | True/False parameter for the company being from 'mobile' category.                                                                 |
| 13     | is_enterprise            | boolean | True/False parameter for the company being from 'enterprise' category.                                                             |
| 14     | is_advertising           | boolean | True/False parameter for the company being from 'advertising' category.                                                            |
| 15     | is_gamesvideo            | boolean | True/False parameter for the company being from 'gamesvideo' category.                                                             |
| 16     | is_ecommerce             | boolean | True/False parameter for the company being from 'ecommerce' category.                                                              |
| 17     | is_biotech               | boolean | True/False parameter for the company being from 'biotech' category.                                                                |
| 18     | is_consulting            | boolean | True/False parameter for the company being from 'consulting' category.                                                             |
| 19     | is_othercategory         | boolean | True/False parameter for the company being in any category other than the ones mentioned above.                                    |
| 20     | has_VC                   | boolean | Determines if the company had Venture capital fund provided.                                                                       |
| 21     | has_angel                | boolean | Determines if the company had a guardian angel overseer to support their endeavours.                                               |
| 22     | has_roundA               | boolean | Parameter stating if company has reached round A level in funding.                                                                 |
| 23     | has_roundB               | boolean | Parameter stating if company has reached round B level in funding.                                                                 |
| 24     | has_roundC               | boolean | Parameter stating if company has reached round C level in funding.                                                                 |
| 25     | has_roundD               | boolean | Series D funding is the fourth stage of fundraising that a business reaches after the seed stage and all rounds of A-C completion. |
| 26     | avg_participants         | float   | Average amount of people involved in the startup throughout in the time of it running.                                             |
| 27     | is_top500                | boolean | Determines if the company has ever reached the Top 500 companies ranking.                                                          |

These columns have been included into our **X dataset**.

As for our target, we've established a column named '_labels_' will be sufficient. This column represents the same values (just in boolean
form instead of string) like column '_status_' of the dataset, which consists of values '**closed**' and '**acquired**'. If a startup was acquired, it means it was so successfull
that it was bought. That is our target of success then - based on the dataset, we have 65% startups acquired and the rest of them have been closed. 

| Number | Name    | Type    | Description                                                                                    |
|--------|---------|---------|------------------------------------------------------------------------------------------------|
| 1      | labels  | boolean | True/False parameter for the company being acquired or closed.|

This column has been included into our **y dataset**.

From different iterations of passing the data through different models, we established that the best 2 are: 

**RandomForestClassifier**, with results: <br>
Best parameters: {'max_depth': 12, 'min_samples_leaf': 2, 'min_samples_split': 12, 'n_estimators': 100} <br>
Best score: 0.8840662712724029 <br>
F1 score: 0.8272727272727273 <br>
Accuracy: 0.7738095238095238 <br>
Precision: 0.7844827586206896 <br>
Recall: 0.875

and **GradientBoostingClassifier**, with results: <br>
Best parameters: {'learning_rate': 0.2, 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 12, 'n_estimators': 100} <br>
Best score: 0.8910059290322788 <br>
F1 score: 0.8625592417061612 <br>
Accuracy: 0.8273809523809523 <br>
Precision: 0.8504672897196262 <br>
Recall: 0.875

The reason why we chose **not** to pick only 1 model is that both of them have results we are satisfied with and when using Shapely values, we established that 
there are little differences of the same parameters influencing those decent results:

| Random Forest | Gradient Boosting  |
|---------------|--------------------|
|![](Images_folder/RFC-shapely.png)|![](Images_folder/GDC-shapely.png)|


### You can check the final results of the model predictions based on your value parameters running the file 'check_your_startup.py' in your local environment. 

This is our first attempt at a Data Science project. We wish to expand our knowledge and improve the approach to our work. If you would like to provide us with good
or constructive feedback - please do so :) You can reach us at **kamil.bukalski@provosolutions.pl** or **gornykml@gmail.com**. In the future, we are planning to change the dataset to be based on Polish companies and to provide a dashboard, 
that will show different plots, graphs and the possibility to interact with the model as it is now via the console. 

