import pandas as pd
import numpy as np
import re
import json
import jieba
import Levenshtein
import logging
import warnings
import pickle
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,format="[%(asctime)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# ----- Abbreviation -----
# ttv_data: train_test_validate_data
# tpn_data: train_positive_negative_data
# -------------------------


# ----- deal data function -----
def deal_data_label(data):  # 处理数据标签
    data['label'] = data['label'].astype(str)
    judge = data['label'] == '音乐'
    data = data[~judge]
    data.reset_index(inplace=True, drop=True)
    data['label'] = data['label'].astype(int)
    return data


def deal_data_flag(data, flag):
    # 处理数据标志用于区别数据性质，0表示训练数据 1表示测试数据 2表示验证数据
    data['data_flag'] = flag
    return data


def deal_ttv_data_flag(train_data, test_data, validate_data):  # 处理ttv数据标志
    train_data = deal_data_flag(train_data, 0)
    test_data = deal_data_flag(test_data, 1)
    validate_data = deal_data_flag(validate_data, 2)
    return train_data, test_data, validate_data


def deal_ttv_data_by_func(train_data, test_data, validate_data, deal_func):
    # 处理ttv数据
    train_data = deal_func(train_data)
    test_data = deal_func(test_data)
    validate_data = deal_func(validate_data)
    return train_data, test_data, validate_data


def deal_data_col_type(data):  # 处理数据类型
    data['label'] = data['label'].astype(int)
    data['prefix'] = data['prefix'].astype(str)
    data['title'] = data['title'].astype(str)
    data['query_prediction'] = data['query_prediction'].astype(str)
    return data


def deal_data_col_len(data):  # 处理数据长度
    data['prefix_len'] = data['prefix'].apply(lambda x: len(x))  # 增加prefix长度字段
    data['title_len'] = data['title'].apply(lambda x: len(x))  # 增加title长度字段
    data['title_diff_prefix_len'] = data['title_len'] - data['prefix_len']
    return data


#################################################################
#  query
def move_useless_char(s):
    # 提出无效字符
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+??！，。？?、~@#￥%……&*（）]+", "", s)


def query_prediction_text(query_prediction):
    if (query_prediction == "{}") | (query_prediction == "") | pd.isna(query_prediction) | (query_prediction == "nan"):
        return ["PAD"]
    json_data = json.loads(query_prediction)
    result = sorted(json_data.items(), key=lambda d: d[1], reverse=True)
    texts = [move_useless_char(item[0]) for item in result]
    return texts


def query_prediction_score(query_prediction):
    if (query_prediction == "{}") | (query_prediction == "") | pd.isna(query_prediction) | (query_prediction == "nan"):
        return [0]
    json_data = json.loads(query_prediction)
    result = sorted(json_data.items(), key=lambda d: d[1], reverse=True)
    scores = [float(item[1]) for item in result]
    return scores


def deal_data_query_score(data):
    data['query_score'] = data['query_prediction'].apply(lambda x: query_prediction_score(x))
    data['query_score_max'] = data['query_score'].apply(lambda x: np.max(x))
    data['query_score_min'] = data['query_score'].apply(lambda x: np.min(x))
    data['query_score_mean'] = data['query_score'].apply(lambda x: np.mean(x))
    data['query_score_median'] = data['query_score'].apply(lambda x: np.median(x))
    data['query_score_sum'] = data['query_score'].apply(lambda x: np.sum(x))
    data['query_score_std'] = data['query_score'].apply(lambda x: np.std(x))
    data['query_score'] = data['query_score'].apply(lambda x: sorted(x, reverse=True))
    data['query_score'] = data['query_score'].apply(lambda x: x+[0 for _ in range(10-len(x))])
    for i in range(10):
        data['query_score_'+str(i)] = data['query_score'].apply(lambda x: x[i])
    data = data.drop(['query_score'], axis =1)
    return data


def get_word_vector():
    word2vec = dict()
    with open("data/zh_word_vectors.txt", 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            word = tokens[0]
            vecs = tokens[1:]
            tmp = []
            for vec in vecs:
                try:
                    tmp.append(float(vec))
                except:
                    pass
            word2vec[word] = np.array(tmp)
    return word2vec


def get_text_vector(x, word2vec, default_vec):
    try:
        return word2vec[x]
    except:
        return default_vec


def deal_data_query_word(data):
    data['query_word'] = data['query_prediction'].apply(lambda x: query_prediction_text(x))
    data['query_len'] = data['query_word'].apply(lambda x: len(x))
    temp_data = data['query_word'].apply(lambda x: [len(_x) for _x in x])
    data['query_word_max_len'] = temp_data.apply(lambda x: np.max(x) if len(x) > 0 else 0)
    data['query_word_min_len'] = temp_data.apply(lambda x: np.min(x) if len(x) > 0 else 0)
    data['query_word_mean_len'] = temp_data.apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    data['query_word_median_len'] = temp_data.apply(lambda x: np.median(x) if len(x) > 0 else 0)
    data['query_word_sum_len'] = temp_data.apply(lambda x: np.sum(x) if len(x) > 0 else 0)
    data['query_word_std_len'] = temp_data.apply(lambda x: np.std(x) if len(x) > 0 else 0)
    data['query_word'] = data['query_word'].apply(lambda x: x+['PAD' for _ in range(10-len(x))])

    # word2vec = get_word_vector()
    # default_vec = np.array([0.0 for _ in range(len(word2vec[list(word2vec.keys())[0]]))])
    # temp_data = data[['prefix', 'query_word']].drop_duplicates('prefix')
    # for i in range(10):
    #     temp_data['query_word_seg_'+str(i)] = temp_data['query_word'].apply(lambda x: "|".join(jieba.cut(str(x[i]))))
    #     tmp_vec = temp_data['query_word_seg_'+str(i)].str.split("|").apply(lambda x: [get_text_vector(_x, word2vec, default_vec) for _x in x])
    #     temp_data['query_word_seg_' + str(i) + '_vec'] = tmp_vec.apply(lambda x: np.sum(x, axis=0))
    # temp_data.drop(columns=['query_word'], inplace= True)
    # data = data.merge(temp_data, on='prefix', how='left')
    return data


def deal_eig_value(similarity_matrix):
    # similarity_matrix: 对称矩阵
    similarity_matrix = np.array(similarity_matrix)
    similarity_matrix = similarity_matrix + similarity_matrix.T
    similarity_matrix[np.eye(similarity_matrix.shape[0]) == 1] = 1
    eig_value = np.linalg.eig(similarity_matrix)[0]
    eig_value = [float(x) for x in eig_value]
    eig_value = sorted(eig_value, reverse=True) + [0 for _ in range(10 - len(eig_value))]
    return eig_value


def deal_query_word_mutual_text_eig_vector(sub_word):
    # 计算query_word 中词组包含关系信息主向量
    sub_word = [x for x in sub_word if x != ""]
    if len(sub_word) > 0:
        similarity_matrix = []
        for _sw in sub_word:
            similarity = [1-(len(sw)-len(_sw))/max([len(sw), len(_sw)]) if _sw in sw else 0 for sw in sub_word ]
            similarity_matrix.append(similarity)
        eig_value = deal_eig_value(similarity_matrix)  # 计算特征向量特征值
    else:
        eig_value = [0 for _ in range(10)]
    return eig_value


def deal_query_word_levenshtein_ratio_eig_vector(sub_word):
    # 计算query_word的 levenshetein 相似度
    sub_word = [x for x in sub_word if x != ""]
    if len(sub_word) > 0:
        similarity_matrix = []
        for _sw in sub_word:
            similarity = [Levenshtein.ratio(_sw, sw) if _sw in sw else 0 for sw in sub_word ]
            similarity_matrix.append(similarity)
        eig_value = deal_eig_value(similarity_matrix) # 计算特征向量
    else:
        eig_value = [0 for _ in range(10)]
    return eig_value


def deal_query_word_levenshtein_distance_eig_vector(sub_word):
    # 计算query_word的 levenshetein 相似度
    sub_word = [x for x in sub_word if x != ""]
    if len(sub_word) > 0:
        similarity_matrix = []
        for _sw in sub_word:
            similarity = [Levenshtein.distance(_sw, sw) if _sw in sw else 0 for sw in sub_word ]
            similarity_matrix.append(similarity)
        eig_value = deal_eig_value(similarity_matrix) # 计算特征向量
    else:
        eig_value = [0 for _ in range(10)]
    return eig_value


def deal_query_word_levenshtein_jaro_eig_vector(sub_word):
    # 计算query_word的 levenshetein 相似度
    sub_word = [x for x in sub_word if x != ""]
    if len(sub_word) > 0:
        similarity_matrix = []
        for _sw in sub_word:
            similarity = [Levenshtein.jaro(_sw, sw) if _sw in sw else 0 for sw in sub_word ]
            similarity_matrix.append(similarity)
        eig_value = deal_eig_value(similarity_matrix) # 计算特征向量
    else:
        eig_value = [0 for _ in range(10)]
    return eig_value


def deal_data_query_sub_word_info(x):
    # 对每个 query_word 删除 prefix
    try:
        rst = [re.sub(x['prefix'], "", _x) for _x in x['query_word']] if len(x['query_word']) > 0 else ['NAN']
    except:
        rst = [_x for _x in x['query_word']]
    return rst


def deal_data_prefix_is_incomplete_input(detected_word, key_word):
    rest_word = detected_word.replace(key_word, "")
    if len(rest_word) > 0:
        return rest_word[0] == "|"
    else:
        return False


def deal_data_query_word_information(data):
    temp_data = data[['prefix', 'query_word', 'prefix_word_seg']].drop_duplicates('prefix')

    # 判断关键词是否输入完整
    temp_data['query_word_seg_0'] = temp_data['query_word'].apply(lambda x: "|".join(jieba.cut(str(x[0]))))
    temp_data['prefix_is_incomplete_input'] = temp_data.apply(lambda x: deal_data_prefix_is_incomplete_input(x['query_word_seg_0'], x['prefix_word_seg']), axis=1).astype(int)
    data = data.merge(temp_data[['prefix', 'prefix_is_incomplete_input']], on='prefix', how='left')
    temp_data = temp_data.drop(['prefix_is_incomplete_input', 'query_word_seg_0', 'prefix_word_seg'], axis=1)

    temp_data['query_sub_word'] = temp_data[['prefix', 'query_word']].apply(lambda x: deal_data_query_sub_word_info(x), axis=1)
    # query_word 交互文本信息
    eig_values = temp_data['query_sub_word'].apply(lambda x: deal_query_word_mutual_text_eig_vector(x))
    for i in range(10):
        temp_data['mutual_text_eig_value_'+str(i)] = eig_values.apply(lambda x: x[i])
    data = data.merge(temp_data.drop(['query_word', 'query_sub_word'], axis=1), on='prefix', how='left')
    temp_data = temp_data[['prefix', 'query_word', 'query_sub_word']]

    # levenshtein ratio 交互文本信息
    eig_values = temp_data['query_sub_word'].apply(lambda x: deal_query_word_levenshtein_ratio_eig_vector(x))
    for i in range(10):
        temp_data['levenshtein_ratio_eig_value_'+str(i)] = eig_values.apply(lambda x: x[i])
    data = data.merge(temp_data.drop(['query_word', 'query_sub_word'], axis=1), on='prefix', how='left')
    temp_data = temp_data[['prefix', 'query_word', 'query_sub_word']]

    # levenshtein distance 交互文本信息
    eig_values = temp_data['query_sub_word'].apply(lambda x: deal_query_word_levenshtein_distance_eig_vector(x))
    for i in range(10):
        temp_data['levenshtein_distance_eig_value_' + str(i)] = eig_values.apply(lambda x: x[i])
    data = data.merge(temp_data.drop(['query_word', 'query_sub_word'], axis=1), on='prefix', how='left')
    return data


#################################################################
# ----- is特征 + prefix -----
def deal_prefix_is_in_title(data):
    data['is_prefix_in_title'] = data.apply(lambda x: int(x['prefix'] in x['title']), axis=1)
    return data


def deal_title_is_in_query_keys(data):
    data['is_title_in_query_keys'] = data.apply(lambda x: int(sum([int(x['title'] in _x) for _x in x['query_word']])>0), axis=1)
    return data


# 是否全是中文
def deal_prefix_is_all_chinese_word(data):
    judge = data['prefix'].apply(lambda x:len(re.findall("[0-9|a-z|A-Z|+??！，。？?、~@#￥%……&*（）|\s+\.\!\/_,$%^*(+\"\']", x)) == 0)
    data['is_all_chinese_word'] = 0
    data.loc[judge, 'is_all_chinese_word'] = 1
    return data


# 是否全是数字
def deal_prefix_is_all_number(data):
    judge = data['prefix'].apply(lambda x:len(re.findall("\D", x))==0)
    data['is_all_number'] = 0
    data.loc[judge, 'is_all_number'] = 1
    return data


# 是否全是英文字母
def deal_prefix_is_all_english(data):
    judge = data['prefix'].apply(lambda x:len(re.findall("[a-z|A-Z]", x)) == len(x))
    data[judge]
    data['is_all_English'] = 0
    data.loc[judge, 'is_all_English'] = 1
    return data


# 是否全是大写英文字母
def deal_prefix_is_all_upper_english(data):
    judge = data['prefix'].apply(lambda x: len(re.findall("[A-Z]", x)) == len(x))
    # data[judge]
    data['is_all_upper_english'] = 0
    data.loc[judge, 'is_all_upper_english'] = 1
    return data


# 是否全是小写英文字母
def deal_prefix_is_all_lower_english(data):
    judge = data['prefix'].apply(lambda x: len(re.findall("[a-z]", x)) == len(x))
    data['is_all_upperEnglish'] = 0
    data.loc[judge, 'is_all_lower_english'] = 1
    return data


# 是否全是特殊符号
def deal_prefix_is_all_symbol(data):
    judge = data['prefix'].apply(lambda x:len(re.findall("\w", x))==0)
    data['is_all_symbol'] = 0
    data.loc[judge, 'is_all_symbol'] = 1
    return data


# 是否中英文一起出现
def deal_prefix_is_combine_chinese_english(data):
    judge = data['prefix'].apply(lambda x: len(re.findall("[\u4e00-\u9fa5]+[a-z|A-Z]+|[a-z|A-Z]+[\u4e00-\u9fa5]+", x))>0)
    data['is_combine_chinese_english'] = 0
    data.loc[judge, 'is_combine_chinese_english'] = 1
    return data


# 是否中文数字出现
def deal_prefix_is_combine_chinese_number(data):
    judge = data['prefix'].apply(lambda x: len(re.findall("[\u4e00-\u9fa5]+[0-9]+|[0-9]+[\u4e00-\u9fa5]+", x))>0)
    data['is_combine_chinese_number'] = 0
    data.loc[judge, 'is_combine_chinese_number'] = 1
    return data


# 是否英文和数字一起出现
def deal_prefix_is_combine_english_number(data):
    judge = data['prefix'].apply(lambda x: len(re.findall("[0-9]+[a-z|A-Z]+|[a-z|A-Z]+[0-9]+", x))>0)
    data['is_combine_english_number'] = 0
    data.loc[judge, 'is_combine_english_number'] = 1
    return data


# 是否网址 # .com 结尾
def deal_prefix_is_network_ip(data):
    judge = data['prefix'].apply(lambda x: len(re.findall("\.(com)$", x))>0)
    data['is_network_ip'] = 0
    data.loc[judge, 'is_network_ip'] = 1
    return data


# prefix归属于tag个数
def deal_prefix_belongs_tag_number(data):
    temp_data = data.groupby(['prefix', 'tag'], as_index=False)['query_prediction'].agg({'prefix_belongs_tag_count': 'count'})
    temp_data = temp_data.groupby('prefix', as_index=False)['prefix_belongs_tag_count'].count()
    data = data.merge(temp_data, on='prefix', how='left')
    return data


# prefix归属于title个数
def deal_prefix_belongs_title_number(data):
    temp_data = data.groupby(['prefix', 'title'], as_index=False)['query_prediction'].agg({'prefix_belongs_title_count': 'count'})
    temp_data = temp_data.groupby('prefix', as_index=False)['prefix_belongs_title_count'].count()
    data = data.merge(temp_data, on='prefix', how='left')
    return data


def deal_data_title_word(data):
    temp_data = data[['title']].drop_duplicates('title')
    temp_data['title_word_seg'] = temp_data['title'].apply(lambda x: "|".join(jieba.cut(x)))
    temp_data['title_word_seg_len'] = temp_data['title_word_seg'].apply(lambda x: len(x.split("|")))
    data = data.merge(temp_data, on='title', how='left')
    return data


def deal_data_prefix_word(data):
    temp_data = data[['prefix']].drop_duplicates('prefix')
    temp_data['prefix_word_seg'] = temp_data['prefix'].apply(lambda x: "|".join(jieba.cut(x)))
    temp_data['prefix_word_seg_len'] = temp_data['prefix_word_seg'].apply(lambda x: len(x.split("|")))
    data = data.merge(temp_data, on='prefix', how='left')
    return data


# static feature
def get_ctr_feature(cols, data, train_data, is_add_0926_data, is_debug):
    ctr_feature_dict = {}
    for col in cols:
        tmp = train_data.groupby(col, as_index=False)["label"].agg({col + "_click": "sum", col + "_show": "count"})
        tmp[col + "_ctr"] = tmp[col + "_click"] / (tmp[col + "_show"] + 3)
        for tmp_col in [col + "_show", col + "_click", col + "_ctr"]:
            tmp[tmp_col] = tmp[tmp_col].apply(lambda x: x if x != "PAD" else -1)
        ctr_feature_dict[col] = tmp
        data = pd.merge(data, tmp, on=col, how="left")

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            group_col = [cols[i], cols[j]]
            tmp = train_data.groupby(group_col, as_index=False)["label"].agg(
                {"_".join(group_col) + "_click": "sum", "_".join(group_col) + "_show": "count"})
            tmp["_".join(group_col) + "_ctr"] = tmp["_".join(group_col) + "_click"] / (
                        tmp["_".join(group_col) + "_show"] + 3)
            for tmp_col in ["_".join(group_col) + "_show", "_".join(group_col) + "_click",
                            "_".join(group_col) + "_ctr"]:
                tmp[tmp_col] = tmp[group_col + [tmp_col]].apply(
                    lambda x: x[tmp_col] if "PAD" not in x[group_col].values else -1, axis=1)
            ctr_feature_dict["_".join(group_col)] = tmp
            data = pd.merge(data, tmp, on=group_col, how="left")

    group_col = cols
    tmp = train_data.groupby(group_col, as_index=False)["label"].agg({"_".join(group_col) + "_click": "sum", "_".join(group_col) + "_show": "count"})
    tmp["_".join(group_col) + "_ctr"] = tmp["_".join(group_col) + "_click"] / (tmp["_".join(group_col) + "_show"] + 3)
    ctr_feature_dict["_".join(group_col)] = tmp
    data = pd.merge(data, tmp, on=cols, how="left")
    if is_debug == 0:  # 判断是否调试模式,调试模式不保存ctr_feature数据
        if is_add_0926_data == 1:  # 判断是否加载0926数据
            with open('data/ctr_feature_dict_0926.pickle', 'wb') as f:
                pickle.dump(ctr_feature_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('data/ctr_feature_dict.pickle', 'wb') as f:
                pickle.dump(ctr_feature_dict, f, pickle.HIGHEST_PROTOCOL)

    data = data.fillna(-1)
    return data


def deal_static_feature(data, mode='unload', is_add_0926_data=1, is_debug=1):
    if mode == 'load':  # 加载数据模式
        if is_add_0926_data == 1:
            with open("data/ctr_feature_dict_0926.pickle", 'rb') as f:
                ctr_feature_dict = pickle.load(f)
        else:
            with open("data/ctr_feature_dict.pickle", 'rb') as f:
                ctr_feature_dict = pickle.load(f)
        for key in list(ctr_feature_dict.keys()):
            tmp_data = ctr_feature_dict[key]
            data = data.merge(tmp_data, on=key.split("_"), how='left')
        data = data.fillna(-1)
    else:
        train_data = data[data['data_flag'] == 0]
        train_data.columns.tolist()
        cols = ["prefix", "title", "tag"]
        data = get_ctr_feature(cols, data, train_data, is_add_0926_data, is_debug)
    return data


def deal_drop_data(data):
    data = data.select_dtypes(include=['number'])
    # -------- 分割线 --------
    # data = data.drop([x for x in data.columns.tolist() if 'mutual_text_eig_value' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'levenshtein_ratio' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'levenshtein_distance' in x], axis=1)

    # data = data.drop([x for x in data.columns.tolist() if 'prefix_is_incomplete_input' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'title_word_seg_len' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'prefix_word_seg_len' in x], axis=1)

    # 如果是版本6 注释上面语句 执行下面语句
    # # data = data.drop([x for x in data.columns.tolist() if 'mutual_text_eig_value' in x], axis=1)
    # # data = data.drop([x for x in data.columns.tolist() if 'levenshtein_ratio' in x], axis=1)
    # # data = data.drop([x for x in data.columns.tolist() if 'levenshtein_distance' in x], axis=1)
    #
    # data = data.drop([x for x in data.columns.tolist() if 'prefix_is_incomplete_input' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'title_word_seg_len' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'prefix_word_seg_len' in x], axis=1)

    return data


def extral_drop_feature(data):
    # data = data.drop([x for x in data.columns.tolist() if 'mutual_text_eig_value' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'levenshtein_ratio' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'levenshtein_distance' in x], axis=1)

    # data = data.drop([x for x in data.columns.tolist() if 'prefix_is_incomplete_input' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'title_word_seg_len' in x], axis=1)
    # data = data.drop([x for x in data.columns.tolist() if 'prefix_word_seg_len' in x], axis=1)
    return data


def deal_data_main(data, static_feature_mode="unload", is_add_0926_data=1, is_debug=1):
    # 处理特征代码主程序
    logging.info("start deal data feature ...")
    data = deal_data_label(data)
    data = deal_data_col_type(data)  # 处理指定col的数据类型
    logging.info("col type finish ...")
    data = deal_data_col_len(data)  # 处理指定col的长度
    logging.info("col len finish ...")

    data = deal_prefix_is_in_title(data)   # 判断prefix是否在title里面
    data = deal_prefix_is_all_chinese_word(data)  # 判断是否全部中文
    data = deal_prefix_is_all_number(data)  # 判断是否全部数字
    data = deal_prefix_is_all_english(data)  # 判断是否全英文
    data = deal_prefix_is_all_upper_english(data)  # 判断是否全部英文大写
    data = deal_prefix_is_all_lower_english(data)  # 判断是否全部英文小写
    data = deal_prefix_is_all_symbol(data)  # 判断是否全部符号
    data = deal_prefix_is_combine_chinese_english(data)  # 判断是否中英字符结合
    data = deal_prefix_is_combine_chinese_number(data)  # 判读是否中文数字结合
    data = deal_prefix_is_combine_english_number(data)  # 判断是否英文数字结合
    data = deal_prefix_is_network_ip(data)  # 判断是否网址
    data = data.fillna(0)  # 将缺失值补充为0
    logging.info("is feature finish ...")

    data = deal_prefix_belongs_tag_number(data)  # 计算prefix归属tag数量
    data = deal_prefix_belongs_title_number(data)  # 计算prefix归属title数量
    logging.info("belongs finish ...")

    data = deal_data_query_score(data)  # 处理query_score 分数特征
    logging.info("query score finish ...")

    data = deal_data_query_word(data)  # 处理query_word
    logging.info("query word finish ...")

    data = deal_data_title_word(data)  # 处理title分词
    logging.info("title word finish...")

    data = deal_data_prefix_word(data)
    logging.info("prefix word finish...")

    data = deal_static_feature(data, static_feature_mode, is_add_0926_data, is_debug)  # 获取统计特征
    logging.info("static finish ...")

    data = deal_data_query_word_information(data)
    logging.info("query_word_information finish ...")

    data = deal_drop_data(data)

    return data


if __name__ == '__main__':
    from GetData import *
    is_deal_data = 1  # 1 表示处理数据，0 表示直接读入处理好数据
    is_add_0926_data = 1  # 1 表示加入0926数据，0 表示不加入0926数据
    is_debug = 1   # 1 表示调试，0表示不调试

    train_data, test_data, validate_data = get_ttv_data(is_debug, is_add_0926_data)
    train_data, test_data, validate_data = deal_ttv_data_flag(train_data, test_data, validate_data)
    all_data = get_merge_data(train_data, test_data, validate_data)  # 合并数据
    all_data = deal_data_main(all_data, "unload", is_add_0926_data)  # 处理特征
    print(all_data.columns.tolist())
    # 新增加指标
    ['prefix_is_incomplete_input', 'prefix_word_seg', 'title_word_seg_len', 'prefix_word_seg_len']