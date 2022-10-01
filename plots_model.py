# nasze funkcje
import our_functions

"""Sprawdzanie zależności"""

columns = ['age_first_funding_year',
           'age_last_funding_year',
           'age_first_milestone_year',
           'age_last_milestone_year'
           ]

our_functions.scatter_all(columns, dataset)

columns = ['relationships', 'funding_rounds', 'milestones']

our_functions.scatter_all(columns, dataset)