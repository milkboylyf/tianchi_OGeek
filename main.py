from DealData import *
from Model import *
import time
import os
import logging
logging.basicConfig(level=logging.INFO,format="[%(asctime)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S",)

# ----- Abbreviation -----
# ttv_data: train_test_validate_data
# tpn_data: train_positive_negative_data
# -------------------------


if __name__ == '__main__':
    print("final")
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists("model"):
        os.mkdir("model")
    is_test_b = 1  # 1 表示b榜测试集 0表示a榜测试集
    is_deal_data = 1  # 1 表示处理数据，0 表示直接读入处理好数据
    is_add_0926_data = 0  # 1 表示加入0926数据，0 表示不加入0926数据
    is_debug = 0   # 1 表示调试，0表示不调试

    if is_deal_data == 1:
        train_data, test_data, validate_data = get_ttv_data(is_debug, is_add_0926_data,is_test_b )
        train_data, test_data, validate_data = deal_ttv_data_flag(train_data, test_data, validate_data)

        all_data = get_merge_data(train_data, test_data, validate_data)  # 合并数据
        all_data = deal_data_main(all_data, "unload", is_add_0926_data, is_debug)  # 处理特征

        train_data = all_data[all_data['data_flag'] == 0]
        test_data = all_data[all_data['data_flag'] == 1]
        validate_data = all_data[all_data['data_flag'] == 2]
        if is_debug == 0:
            if is_add_0926_data == 1:
                train_data.to_csv("data/train_data_add_0926.csv", header=True, index=False, encoding='utf8')
                test_data.to_csv("data/test_data_add_0926.csv", header=True, index=False, encoding='utf8')
                validate_data.to_csv("data/validate_data_add_0926.csv", header=True, index=False, encoding='utf8')
            else:
                train_data.to_csv("data/train_data.csv", header=True, index=False, encoding='utf8')
                test_data.to_csv("data/test_data.csv", header=True, index=False, encoding='utf8')
                validate_data.to_csv("data/validate_data.csv", header=True, index=False, encoding='utf8')
    else:
        if is_add_0926_data == 1:
            train_data = pd.read_csv("data/train_data_add_0926.csv")
            test_data = pd.read_csv("data/test_data_add_0926.csv")
            validate_data = pd.read_csv("data/validate_data_add_0926.csv")
        else:
            train_data = pd.read_csv("data/train_data.csv")
            test_data = pd.read_csv("data/test_data.csv")
            validate_data = pd.read_csv("data/validate_data.csv")


    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
    validate_data = pd.read_csv("data/validate_data.csv")

    train_data = pd.concat([train_data, validate_data], ignore_index=False)
    train_data = extral_drop_feature(train_data)  # 该函数用于调试特征

    train_positive_data = get_positive_data(train_data)
    train_negative_data = get_negative_data(train_data)
    validate_positive_data = get_positive_data(validate_data)
    validate_negative_data = get_negative_data(validate_data)

    time_name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))  # 获取当前时间
    save_assistant_name = time_name + "_is_used_0926data_"+str(is_add_0926_data)  # 保存文件辅助变量

    # xgb_fe_model = XgbFeatureEnsembleModel(train_data, train_positive_data, train_negative_data)
    # xgb_fe_model.train()
    # xgb_fe_model.predict(validate_data)
    # xgb_fe_model.precision_score("xgb_fe model precision score")
    # xgb_fe_model.recall_score("xgb_fe model recall score")
    # xgb_fe_model.score("xgb_fe model validate score")
    # xgb_fe_model.output_result("result"+"/validate_xgb_"+save_assistant_name + "_score_" + str(int(xgb_fe_model.f1_score*1000000)) + ".csv")
    #
    # xgb_fe_model.predict(test_data)
    # xgb_fe_model.output_result("result" + "/test_xgb_" + save_assistant_name + "_score_" + str(int(xgb_fe_model.f1_score*1000000)) + ".csv")
    #
    # xgb_fe_model.data = []
    # xgb_fe_model.positive_data = []
    # xgb_fe_model.negative_data = []
    # SaveClassModel().save_model("model/class_xgb_"+save_assistant_name + "_score_" + str(int(xgb_fe_model.f1_score*1000000)) + ".pickle", xgb_fe_model)

    lgb_fe_model = LgbFeatureEnsembleModel(train_data, train_positive_data, train_negative_data)
    lgb_fe_model.set_train_device()
    lgb_fe_model.train()
    lgb_fe_model.predict(validate_data)
    lgb_fe_model.precision_score("lgb_fe model precision score")
    lgb_fe_model.recall_score("lgb_fe model recall score")
    lgb_fe_model.score("lgb_fe model validate score")
    lgb_fe_model.output_result("result"+"/validate_lgb_"+save_assistant_name + "_score_" + str(int(lgb_fe_model.f1_score*1000000)) + ".csv")

    lgb_fe_model.predict(test_data)
    lgb_fe_model.output_result("result"+"/test_lgb_"+save_assistant_name + "_score_" + str(int(lgb_fe_model.f1_score*1000000)) + ".csv")

    lgb_fe_model.data = []
    lgb_fe_model.positive_data = []
    lgb_fe_model.negative_data = []
    SaveClassModel().save_model("model/class_lgb_" + save_assistant_name + "_score_" + str(int(lgb_fe_model.f1_score*1000000)) + ".pickle", lgb_fe_model)
    logging.info("model finish ...")

    # print("*"*100+"DirectlyPredictResult"+"*"*100)
    # model_path = "model/class_lgb_" + save_assistant_name + "_score_" + str(int(lgb_fe_model.f1_score*1000000)) + ".pickle"
    # data_path = "data/oppo_round1_vali_20180929.txt"
    #
    # dpr = DirectlyPredictResult(model_path=model_path, data_path=data_path, is_ouput_score=1, is_add_0926_data=is_add_0926_data, is_debug=is_debug)
    # dpr.predict()
    # rst_path = "result"+"/vali_dpr_"+save_assistant_name + "_score_" + str(int(dpr.model.f1_score*1000000)) + ".csv"
    # dpr.output_result(rst_path)

    # np.sum(dpr.data.drop(['label'],axis=1)==validate_data[lgb_fe_model.data_col].drop(['label'],axis=1),axis=0)