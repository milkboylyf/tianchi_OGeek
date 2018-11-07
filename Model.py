from GetData import *
import DealData
from itertools import combinations
import pickle
import numpy as np
import logging
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
logging.basicConfig(level=logging.INFO,format="[%(asctime)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)

# ----- model class -----
class BaseModel:
    def __init__(self, data, positive_data, negative_data):
        self.data = data
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.__base_model_data = None
        self.predict_result = None
        self.threshold_value = 1/3
        self.f1_score = 0
        self.precision = 0
        self.recall = 0


    def set_threshold_value(self, value):
        self.threshold_value = value

    def find_best_threshold_value(self, value_list, data):
        for value in value_list:
            print("#" * 100)
            print(value / 100)
            self.set_threshold_value(value / 100)
            self.train()
            self.predict(data)
            self.score("BaseModel")

    def train(self):
        group_data = get_tpn_group_data_by_col(self.data, self.positive_data, self.negative_data, 'tag', 'prefix', 'count')
        group_data['predict_label'] = 0
        judge = group_data['rate'] > self.threshold_value
        group_data.loc[judge, 'predict_label'] = 1
        self.__base_model_data = group_data[['tag', 'predict_label']]

    def predict(self, data):
        new_predict_data = pd.merge(data, self.__base_model_data, on='tag', how='left')
        self.predict_result = new_predict_data[['label', 'predict_label']]

    def score(self, model_name):
        from sklearn.metrics import f1_score
        score = f1_score(self.predict_result['label'].astype(int), self.predict_result['predict_label'].astype(int), pos_label=1)
        print(model_name+" score:"+str(score))
        self.f1_score = score

    def precision_score(self, model_name):
        from sklearn.metrics import precision_score
        score = precision_score(self.predict_result['label'], self.predict_result['predict_label'], pos_label=1)
        self.precision = score
        print(model_name + " score:" + str(score))

    def recall_score(self, model_name):
        from sklearn.metrics import recall_score
        score = recall_score(self.predict_result['label'], self.predict_result['predict_label'], pos_label=1)
        self.recall = score
        print(model_name + " score:" + str(score))

    def output_result(self, path):
        self.predict_result['predict_label'].to_csv(path, header=False, index=False, encoding='utf8')

    def reverse_predict_result(self):
        judge1 = self.predict_result['predict_label'] == 0
        judge2 = self.predict_result['predict_label'] == 1
        self.predict_result.loc[judge1, 'predict_label'] = 1
        self.predict_result.loc[judge2, 'predict_label'] = 0


# ----- CombSearchModel -----
class CombSearchModel(BaseModel):
    def __init__(self, data, positive_data, negative_data):
        super().__init__(data, positive_data, negative_data)
        self.__cs_model_data = pd.DataFrame()
        self.support_num = 50  # the number to guarantee the model stability
        self.support_rate = 0.7  # the rate to guarantee the model score
        self.candidate_list = ['tag', 'prefix', 'title', 'query_len', 'prefix_len', 'title_len']
        self.comb_num = 4  # 特征组合数目
        self.count_num = 0
        self.combined_feature_data = {}  # 存储组合特征数据

    def set_candidate_list(self, candidate_list):  # 设置特征候选集合
        self.candidate_list = candidate_list

    def set_support_num(self, support_num):  # 设置支持数目
        self.support_num = support_num

    def set_support_rate(self, support_rate):  # 设置支持率
        self.support_rate = support_rate

    def get_train_combined_feature_data(self): # 获取组合特征数据
        print("CombModel Train")
        for comb_len in list(range(1, self.comb_num+1)):
            comb_lists = list(combinations(self.candidate_list, comb_len))
            for comb_list in comb_lists:
                group_col_list = [x for x in comb_list]
                merge_col_name = get_merge_col_name(group_col_list)
                group_data = get_tpn_group_data_by_collist(self.data, self.positive_data, self.negative_data, group_col_list, 'query_prediction', 'count')
                if group_col_list == ["tag"]:
                    train_group_data = group_data.copy()
                else:
                    judge1 = (group_data[merge_col_name + "_count"] > self.support_num)
                    judge2 = (group_data['rate'] >= self.support_rate) | (group_data['rate'] <= (1 - self.support_rate))
                    judge = judge1 & judge2
                    train_group_data = group_data[judge].copy()
                if train_group_data.shape[0] > 0:
                    self.count_num += train_group_data[merge_col_name + "_count"].sum()
                    print(group_col_list, 'data shape:', train_group_data.shape[0], "count_num",
                          train_group_data[merge_col_name + "_count"].sum())
                    self.combined_feature_data[merge_col_name] = train_group_data
                    
    def train(self):
        self.get_train_combined_feature_data()

    def predict(self, data):
        keys = list(self.combined_feature_data.keys())
        predict_data = data.copy()
        for key in keys:
            merge_col = key.split("@")[1:]
            new_merge_col = merge_col + ['rate']
            predict_data = pd.merge(predict_data, self.combined_feature_data[key][new_merge_col], on=merge_col, how='left')
            predict_data.rename(columns={"rate": "rate_"+key}, inplace=True)  # 重新命名
        rate_judge = ["rate" in x for x in list(predict_data.columns)]
        rate_cols = list(predict_data.columns[rate_judge])
        predict_data['positive_rate'] = predict_data[rate_cols].max(axis=1)
        predict_data['negative_rate'] = 1 - predict_data[rate_cols].min(axis=1)
        predict_data['predict_label'] = 1

        judge = predict_data['positive_rate'] < predict_data['negative_rate']
        predict_data.loc[judge, 'predict_label'] = 0
        predict_data[['positive_rate', 'label', 'predict_label']].sort_values(by='positive_rate')
        self.predict_result = predict_data[['label', 'predict_label']]


# ----- EnsembleModel -----
class BaseFeatureEnsembleModel(CombSearchModel):
    def __init__(self, data, positive_data, negative_data):
        super().__init__(data, positive_data, negative_data)
        self.ef_model = {}
        self.cv_k = 1
        self.data_col = []
    
    def save_combined_feature_data(self, data_name):  # 保存组合特征数据
        with open(data_name, 'wb') as f:
            pickle.dump(self.combined_feature_data, f, pickle.HIGHEST_PROTOCOL)

    def update_combined_feature_data(self, data_name): # 更新组合特征数据
        # 加载更新模型
        with open(data_name, 'rb') as f:
            self.combined_feature_data = pickle.load(f)

    def get_ef_data(self, data):
        _ef_data = data.select_dtypes(include=['number'])
        _ef_data = _ef_data.drop(['data_flag'], axis=1)
        return _ef_data


class LgbFeatureEnsembleModel(BaseFeatureEnsembleModel):
    def __init__(self, data, positive_data, negative_data):
        super().__init__(data, positive_data, negative_data)
        self.cut_value = 0.3
        self.train_device = 'cpu'  # 设置训练

    def f1_score_metric(self, pred, d_valid):
        label = d_valid.get_label()
        pred = [int(i >= self.cut_value) for i in pred]
        return "f1_score", f1_score(label, pred), True


    def set_train_device(self, device='cpu'):  # 设置训练设备
        self.train_device = device
        
    def train(self):
        self.data_col = self.data.columns.tolist()
        _ef_data = self.get_ef_data(self.data)
        X = np.array(_ef_data.drop(['label'], axis=1))
        y = np.array(_ef_data['label'])
        result_logloss = []
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        if self.train_device == 'cpu':
            params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss',
                      'num_leaves': 32, 'learning_rate': 0.05, 'feature_fraction': 0.3, 'bagging_fraction': 0.8,
                      'bagging_freq': 5, 'verbose': -1, 'device': 'cpu', }
        else:
            params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss',
                      'num_leaves': 32, 'learning_rate': 0.05, 'feature_fraction': 0.1, 'bagging_fraction': 0.8,
                      'bagging_freq': 5, 'verbose': -1, 'device': 'gpu', 'gpu_platform_id': 0,'gpu_device_id': 0,
                      }
        for k, (train_in, test_in) in enumerate(skf.split(X, y)):
            if k < self.cv_k:
                logging.info("train _K_ flod "+str(k))
                X_train, X_valid, y_train, y_valid = X[train_in], X[test_in], y[train_in], y[test_in]
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
                gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=500, verbose_eval=250, feval=self.f1_score_metric)
                valid_f1_score = f1_score(y_valid, np.where(gbm.predict(X_valid, num_iteration=gbm.best_iteration) > self.cut_value, 1, 0))
                print("best_iteration: ", gbm.best_iteration)
                print("valid_f1_score: ", valid_f1_score)
                result_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
                self.ef_model[str(k)] = gbm
                feature_importances = sorted(zip(_ef_data.columns.drop('label'), gbm.feature_importance()), key=lambda x: x[1], reverse=True)
                print('feature_importances', feature_importances)

    def save_model(self, model_name):
        with open(model_name, 'wb') as f:
            pickle.dump(self.ef_model, f, pickle.HIGHEST_PROTOCOL)

    def update_model(self, model_name):
        with open(model_name, 'wb') as f:
            pickle.load(self.ef_model, f, pickle.HIGHEST_PROTOCOL)

    def predict(self, data):
        result_submit = []
        _ef_data = self.get_ef_data(data)
        for key in self.ef_model.keys():
            gbm = self.ef_model[key]
            result_submit.append(gbm.predict(_ef_data.drop(columns=['label']), num_iteration=gbm.best_iteration))
        self.predict_result = data.copy()
        self.predict_result['predict_label'] = list(np.sum(np.array(result_submit), axis=0) / len(result_submit))
        self.predict_result['predict_label'] = self.predict_result['predict_label'].apply(lambda x: 1 if x > self.cut_value else 0)
        self.predict_result = self.predict_result[['label', 'predict_label']]


class LogisticRegression(BaseFeatureEnsembleModel):

    def __init__(self, data, positive_data, negative_data):
        super().__init__(data, positive_data, negative_data)

    def train(self):
        print('abc')

    def predict(self):
        print('abc')


class XgbFeatureEnsembleModel(BaseFeatureEnsembleModel):
    def __init__(self, data, positive_data, negative_data):
        super().__init__(data, positive_data, negative_data)
        self.cut_value = 0.3

    def f1_score_metric(self, pred, d_valid):
        label = d_valid.get_label()
        pred = [int(i >= self.cut_value) for i in pred]
        return "f1_score", f1_score(label, pred)

    def train(self):
        _ef_data = self.get_ef_data(self.data)
        X = np.array(_ef_data.drop(['label'], axis=1))
        y = np.array(_ef_data['label'])
        skf = StratifiedKFold(n_splits=5, random_state=34, shuffle=True)
        params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'eta': 0.05,
                  'max_depth': 5,  'colsample_bytree': 0.8, 'subsample': 0.8, 'alpha':1,
                  'min_child_weight': 1,  'seed': 10086, 'silent': 1}
        for k, (train_in, test_in) in enumerate(skf.split(X, y)):
            if k < self.cv_k:
                logging.info("train _K_ flod "+str(k))
                X_train, X_valid, y_train, y_valid = X[train_in], X[test_in], y[train_in], y[test_in]
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dvali = xgb.DMatrix(X_valid, label=y_valid)
                model = xgb.train(params, dtrain, evals=[(dtrain,"train"), (dvali, "vali")], num_boost_round=5000, early_stopping_rounds=500, verbose_eval=1000, feval=self.f1_score_metric)
                feature_importances = sorted(zip(_ef_data.columns.drop('label'), list(model.get_score().values())), key=lambda x: x[1], reverse=True)
                self.ef_model[str(k)] = model
                print("best_iteration: ", model.best_iteration)
                print('feature_importances', feature_importances)

    def predict(self, data):
        result_submit = []
        _ef_data = self.get_ef_data(data)
        X = np.array(_ef_data.drop(['label'], axis=1))
        for key in self.ef_model.keys():
            model = self.ef_model[key]
            result_submit.append(model.predict(xgb.DMatrix(X)))
        self.predict_result = data.copy()
        self.predict_result['predict_label'] = list(np.sum(np.array(result_submit), axis=0) / len(result_submit))
        self.predict_result['predict_label'] = self.predict_result['predict_label'].apply(lambda x: 1 if x > self.cut_value else 0)
        self.predict_result = self.predict_result[['label', 'predict_label']]


class SaveClassModel():
    def __init__(self):
        self.model = {}
        self.data = {}

    def save_model(self, model_path, model):
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)


class DirectlyPredictResult(SaveClassModel):
    # 该类用于根据已有模型和输入数据路径进行结果预测
    def __init__(self, model_path, data_path, is_ouput_score=1, is_add_0926_data=1, is_debug=1):
        super().__init__()
        self.is_ouput_score = is_ouput_score  # is_ouput_score =1 表示输出分数，is_ouput_score=0 表示不输出分数
        self.data_path = data_path
        self.is_add_0926_data = is_add_0926_data
        self.is_debug = is_debug
        self.model_path = model_path
        self.main()

    def main(self):
        self.load_model(self.model_path)
        self.import_data()

    def import_data(self):
        col_names = ['prefix', 'query_prediction', 'title', 'tag', 'label']
        data = pd.read_csv(self.data_path, names=col_names, sep="\t", low_memory=False)
        data.loc[pd.isna(data['label']), 'label'] = 0
        data = DealData.deal_data_flag(data, 1)
        data = DealData.deal_data_main(data, 'load', self.is_add_0926_data, self.is_debug)
        # data = DealData.extral_drop_feature(data)
        self.data = data[self.model.data_col]

    def predict(self):
        self.model.predict(self.data)
        if self.is_ouput_score == 1:
            self.model.precision_score("dpr precision score")
            self.model.recall_score("dpr recall score")
            self.model.score("dpr validate score")

    def output_result(self, output_result_path):
        self.model.output_result(output_result_path)


