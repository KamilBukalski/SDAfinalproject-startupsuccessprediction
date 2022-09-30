import matplotlib.pyplot as plt


def scatter_all(columns, dataset):
    plt.figure(figsize=(20, 10))
    for i in columns:
        dataset2 = dataset.copy()
        dataset2[i] = round(dataset2[i])
        dataset_age_group = dataset2[dataset2['labels'] == 1].groupby([i]).agg({'labels': 'count'}).reset_index()
        dataset_age_group.columns = [i, 'total_succes']

        dataset_age_group_total = dataset2.groupby([i]).agg({'labels': 'count'}).reset_index()
        dataset_age_group_total.columns = [i, 'total']

        dataset_age_group = dataset_age_group.merge(dataset_age_group_total, on=i)
        dataset_age_group['succes_rate'] = round((dataset_age_group['total_succes'] / dataset_age_group['total']) * 100,
                                                 2)

        plt.scatter(dataset_age_group[i], dataset_age_group['succes_rate'], label=i)
    plt.legend()
    # plt.show()


def bar_single(columns, dataset):
    plt.figure(figsize=(20, 10))
    for i in columns:
        dataset2 = dataset.copy()
        try:
            dataset2[i] = round(dataset2[i])
        except:
            pass
        dataset_age_group = dataset2[dataset2['labels'] == 1].groupby([i]).agg({'labels': 'count'}).reset_index()
        dataset_age_group.columns = [i, 'total_succes']

        dataset_age_group_total = dataset2.groupby([i]).agg({'labels': 'count'}).reset_index()
        dataset_age_group_total.columns = [i, 'total']

        dataset_age_group = dataset_age_group.merge(dataset_age_group_total, on=i)

        dataset_age_group['succes_rate'] = round((dataset_age_group['total_succes'] / dataset_age_group['total']) * 100,
                                                 2)
        plt.figure(figsize=(25, 2))

        plt.bar(dataset_age_group[i], dataset_age_group['succes_rate'], label=i)
        plt.title(i)
        plt.legend()
        plt.xticks(rotation=60)
        # plt.show()


def bar_categorical(columns, dataset):
    plt.figure(figsize=(20, 10))
    for i in columns:
        dataset2 = dataset.copy()
        a = dataset[i].value_counts(normalize=True)

        plt.bar(dataset[i].unique(), a, label=i)
        plt.title(i)
        plt.legend()
        plt.xticks(rotation=60)
        # plt.show()
