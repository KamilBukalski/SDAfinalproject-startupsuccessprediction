# SDAfinalproject-startupsuccessprediction
In this project we wanted to create a model that will predict the success rate of a startup based on multiple input parameters.

# Data
Source of the data used in this project is a dataset from Kaggle that can be found here:
https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

After initial data exploration and preparation we established a couple of important parameters:

| Number | Name                     | Type | Description |
|--------|--------------------------|------|-------------|
| 1      | age                      |      |             |
| 2      | age_first_funding_year   |      |             |
| 3      | age_last_funding_year    |      |             |
| 4      | age_first_milestone_year |      |             |
| 5      | age_last_milestone_year  |      |             |
| 6      | relationships            |      |             |
| 7      | funding_rounds           |      |             |
| 8      | funding_total_usd        |      |             |
| 9      | milestones               |      |             |
| 10     | is_software              |      |             |
| 11     | is_web                   |      |             |
| 12     | is_mobile                |      |             |
| 13     | is_enterprise            |      |             |
| 14     | is_advertising           |      |             |
| 15     | is_gamesvideo            |      |             |
| 16     | is_ecommerce             |      |             |
| 17     | is_biotech               |      |             |
| 18     | is_consulting            |      |             |
| 19     | is_othercategory         |      |             |
| 20     | has_VC                   |      |             |
| 21     | has_angel                |      |             |
| 22     | has_roundA               |      |             |
| 23     | has_roundB               |      |             |
| 24     | has_roundC               |      |             |
| 25     | has_roundD               |      |             |
| 26     | avg_participants         |      |             |
| 27     | is_top500                |      |             |

As for our target, we've established a column named 'labels' will be sufficient. This column represents the same values (just in boolean
form) like column 'status' of the dataset, which consists of values 'closed' and 'acquired'. If a startup was acquired, it means it was so successfull
that it was bought. That is our target of success then. The rate of this happening, will be explored later on.

| Parameter | Name   | Description |
|-----------|--------|-------------|
| y         | labels |             |

