from pyecharts import Bar
from pyecharts import Scatter
from GetData import *
from PrintData import *
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


def visual_bar_group_data_by_col(group_data, col_name, page):
    bar = Bar(col_name)
    attr = list(group_data[col_name])
    train_value = list(group_data[col_name+'_count'])
    positive_value = list(group_data['positive_'+col_name+'_count'])
    negative_value = list(group_data['negative_'+col_name+'_count'])
    bar.add("train", attr, train_value)
    bar.add("positive", attr, positive_value)
    bar.add("negative", attr, negative_value)
    page.add(bar)
    return page


def visual_scatter_group_data_by_col(group_data, col_name, page):
    x_value = list(group_data[col_name])
    y_value = list(group_data['rate'])
    scatter = Scatter()
    scatter.add(col_name, x_value, y_value)
    page.add(scatter)
    return page


def visual_bar_tpn_data_by_col_list(train_data, positive_data, negative_data, page):
    col_list = ['tag', 'prefix', 'title', 'query_len', 'prefix_len', 'title_len', 'query_num_sum', 'query_num_max',
                'query_num_first']
    for col_name in col_list:
        group_data = get_tpn_group_data_by_col(train_data, positive_data, negative_data, col_name, 'query_prediction', 'count')
        print_data_num(group_data, col_name + "_data")
        extracted_group_data = group_data.sort_values(by= col_name + '_count', ascending=False)
        page = visual_bar_group_data_by_col(extracted_group_data.head(25), col_name, page)
    return page


def visual_scatter_tpn_data_by_col_list(train_data, positive_data, negative_data, page):
    col_list = ['query_num_sum', 'query_num_max', 'query_num_first']
    for col_name in col_list:
        group_data = get_tpn_group_data_by_col(train_data, positive_data, negative_data, col_name, 'query_prediction', 'count')
        print_data_num(group_data, col_name + "_data")
        judge = group_data[col_name + "_count"] > group_data[col_name + "_count"].mean()
        extracted_group_data = group_data[judge]
        extracted_group_data = extracted_group_data.sort_values(by=col_name)
        page = visual_scatter_group_data_by_col(extracted_group_data, col_name, page)
    return page
