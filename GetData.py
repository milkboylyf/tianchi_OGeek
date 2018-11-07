import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


# ----- Abbreviation -----
# ttv_data: train_test_validate_data
# tpn_data: train_positive_negative_data
# -------------------------


# ----- get data function -----
def get_ttv_data(is_debug,is_add_0926_validate_data, is_test_b):  # 获取ttv数据
    # 导入数据
    col_names = ['prefix', 'query_prediction', 'title', 'tag', 'label']
    if is_debug == 1:
        train_data = pd.read_csv("data/oppo_round1_train_20180929.txt", names=col_names, sep="\t", low_memory=False, nrows=250)
        test_data = pd.read_csv("data/oppo_round1_test_A_20180929.txt", names=col_names, sep="\t", low_memory=False, nrows=250)
        test_data['label'] = '0'
        validate_data = pd.read_csv("data/oppo_round1_vali_20180929.txt", names=col_names, sep="\t", low_memory=False, nrows=250)
    else:
        train_data = pd.read_csv("data/oppo_round1_train_20180929.txt", names=col_names, sep="\t", low_memory=False)

        if is_test_b == 1:
            test_data = pd.read_csv("data/oppo_round1_test_B_20181106.txt", names=col_names, sep="\t", low_memory=False)
            test_data['label'] = '0'
        else:
            test_data = pd.read_csv("data/oppo_round1_test_A_20180929.txt", names=col_names, sep="\t", low_memory=False)
            test_data['label'] = '0'

        validate_data = pd.read_csv("data/oppo_round1_vali_20180929.txt", names=col_names, sep="\t", low_memory=False)
        if is_add_0926_validate_data == 1:
            train_0926_data = pd.read_csv("data/oppo_round1_train_20180926.txt", names=col_names, sep="\t", low_memory=False)
            train_data = pd.concat([train_data, train_0926_data], ignore_index=True)
            validate_0926_data = pd.read_csv("data/oppo_round1_vali_20180926.txt", names=col_names, sep="\t", low_memory=False)
            validate_data = pd.concat([validate_data, validate_0926_data], ignore_index=True)
    return train_data, test_data, validate_data


def get_merge_data(train_data,test_data,validate_data):
    # 合并ttv数据
    data = pd.concat([train_data, test_data, validate_data], ignore_index=True)
    return data


def get_positive_data(data):
    # 获取正样本数据
    positive_data = data[data['label'] == 1]
    return positive_data


def get_negative_data(data):
    # 获取负样本数据
    negative_data = data[data['label'] == 0]
    return negative_data


def get_group_data_by_col(data, group_col_name, new_col_name, static_name, static_method):
    # 按照某一列汇总
    group_data = data[[group_col_name, static_name]].groupby(by=group_col_name, as_index=False).agg({static_name: static_method})
    group_data.rename(columns={static_name: new_col_name}, inplace=True)
    return group_data


def get_group_data_by_collist(data, group_col_name, new_col_name, static_name, static_method):
    # 按照列表汇总
    group_data = data[group_col_name+[static_name]].groupby(by=group_col_name, as_index=False).agg({static_name: static_method})
    group_data.rename(columns={static_name: new_col_name}, inplace=True)
    return group_data


def get_tpn_group_data_by_col(train_data, positive_data, negative_data, group_col_name, static_name, static_method):
    # 对ttv数据按照某一列汇总
    train_new_col_name = group_col_name+"_"+static_method
    positive_new_col_name = 'positive_'+group_col_name+"_"+static_method
    negative_new_col_name = 'negative_'+group_col_name+"_"+static_method
    group_train_data = get_group_data_by_col(train_data, group_col_name, train_new_col_name, static_name, static_method)
    group_positive_data = get_group_data_by_col(positive_data, group_col_name, positive_new_col_name, static_name, static_method)
    group_negative_data = get_group_data_by_col(negative_data, group_col_name, negative_new_col_name, static_name, static_method)
    group_data = pd.merge(group_train_data, group_positive_data, on=group_col_name, how='left')
    group_data = pd.merge(group_data, group_negative_data, on=group_col_name, how='left')
    group_data.fillna(0, inplace=True)
    group_data = get_data_rate(group_data, positive_new_col_name, train_new_col_name)
    group_data.sort_values(by=train_new_col_name, ascending=False, inplace=True)
    return group_data


def get_merge_col_name(col_list):
    # 合并列表字符元素
    merge_col_name = "".join(["@"+x for x in col_list])
    merge_col_name = "merge"+merge_col_name
    return merge_col_name


def get_tpn_group_data_by_collist(train_data, positive_data, negative_data, group_col_name, static_name, static_method):
    merge_group_col_name = get_merge_col_name(group_col_name)
    train_new_col_name = merge_group_col_name+"_"+static_method
    positive_new_col_name = 'positive_'+merge_group_col_name+"_"+static_method
    negative_new_col_name = 'negative_'+merge_group_col_name+"_"+static_method

    group_train_data = get_group_data_by_collist(train_data, group_col_name, train_new_col_name, static_name, static_method)
    group_positive_data = get_group_data_by_collist(positive_data, group_col_name, positive_new_col_name, static_name, static_method)
    group_negative_data = get_group_data_by_collist(negative_data, group_col_name, negative_new_col_name, static_name, static_method)

    group_data = pd.merge(group_train_data, group_positive_data, on=group_col_name, how='left')
    group_data = pd.merge(group_data, group_negative_data, on=group_col_name, how='left')
    group_data.fillna(0, inplace=True)
    group_data = get_data_rate(group_data, positive_new_col_name, train_new_col_name)
    group_data = get_data_idf(group_data, train_new_col_name)
    group_data.sort_values(by=train_new_col_name, ascending=False, inplace=True)
    return group_data


def get_data_rate(data, col_name_1, col_name_2):
    data['rate'] = (data[col_name_1]).div(data[col_name_2])
    return data


def get_data_idf(data, col_name):
     sum_value = data[col_name].sum()
     data['idf'] = data[col_name]/sum_value
     return data


def detect_train_validate_distribution(train_data, validate_data):
    col_name = 'prefix'
    head_n = 10
    train_positive_data = get_positive_data(train_data)
    train_negative_data = get_negative_data(train_data)
    validate_positive_data = get_positive_data(validate_data)
    validate_negative_data = get_negative_data(validate_data)

    train_group_data = get_tpn_group_data_by_col(train_data, train_positive_data, train_negative_data, col_name, 'query_prediction', 'count')
    validate_group_data = get_tpn_group_data_by_col(validate_data, validate_positive_data, validate_negative_data, col_name, 'query_prediction', 'count')
    print(train_group_data.head(head_n))
    judge = validate_group_data[col_name].isin(train_group_data[col_name].head(head_n))
    print(validate_group_data[judge])

    train_value_rate = train_group_data['rate'].head(head_n)
    validate_value_rate = validate_group_data[judge]['rate']
    print("mean_value:", [train_value_rate.mean(), validate_value_rate.mean()])
    print("std_value:", [train_value_rate.std(), validate_value_rate.std()])