# tianchi_OGeek
在搜索业务下有一个场景叫实时搜索（Instance Search）,就是在用户不断输入过程中，实时返回查询结果。  此次赛题来自OPPO手机搜索排序优化的一个子场景，并做了相应的简化，意在解决query-title语义匹配的问题。简化后，本次题目内容主要为一个实时搜索场景下query-title的ctr预估问题。

0 分数
======
>(1) A榜：0.7347 <br>
>(2) B榜：0.7335 <br>
>(3) 比赛网址：https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11409106.5678.1.2c547b6fmKviKy&raceId=231688<br>
>(4) 数据下载地址：链接：https://pan.baidu.com/s/1NPUWzt7usUniogCJosWnzw  提取码：69xr  <br>

1 baseline 共享网址
======
>(1) 天池-OGeek算法挑战赛baseline(0.7016) https://zhuanlan.zhihu.com/p/46482521 <br>
>(2) OGEEK算法挑战赛代码分享 https://zhuanlan.zhihu.com/p/46479794 <br>
>(3) GrinAndBear/OGeek: https://github.com/GrinAndBear/OGeek <br>
>(4) flytoylf/OGeek 一个lgb和rnn的代码: https://github.com/flytoylf/OGeek <br>
>(5) https://github.com/search?q=OGeek <br>
>(6) https://github.com/search?q=tianchi_oppo <br>
>(7) https://github.com/luoling1993/TianChi_OGeek/stargazers <br>


2 CTR 参考资料
======
>(1) 推荐系统遇上深度学习: https://github.com/princewen/tensorflow_practice <br>
>(2) 推荐系统中使用ctr排序的f(x)的设计-dnn篇: https://github.com/nzc/dnn_ctr <br>
>(3) CTR预估算法之FM, FFM, DeepFM及实践: https://github.com/milkboylyf/CTR_Prediction <br>
>(4) MLR算法: https://wenku.baidu.com/view/b0e8976f2b160b4e767fcfdc.html <br>


3 nlp 参考资料
======
>(1) 用深度学习（CNN RNN Attention）解决大规模文本分类问题 - 综述和实践 https://zhuanlan.zhihu.com/p/25928551 <br>
>(2) 知乎“看山杯” 夺冠记：https://zhuanlan.zhihu.com/p/28923961 <br>
>(3) 2017知乎看山杯 从入门到第二 https://zhuanlan.zhihu.com/p/29020616 <br>
>(4) liuhuanyong  https://github.com/liuhuanyong <br>
>(5) Chinese Word Vectors 中文词向量 https://github.com/Embedding/Chinese-Word-Vectors 注释：这个链接收藏语料库 <br>

4 其他比赛总结参考链接
======
>(1) ML理论&实践  https://zhuanlan.zhihu.com/c_152307828?tdsourcetag=s_pctim_aiomsg <br>

5 未整理思路
======
>(1) 主线思路：CTR思路，围绕用户点击率做文章(如开源中：单字段点击率，组合字段点击率等等) (FM, FFM模型，参考腾讯社交广告比赛？？) <br>
>(2) 文本匹配思路（Kaggle Quora） 传统特征：抽取文本相似度特征，各个字段之间的距离量化 https://www.kaggle.com/c/quora-question-pairs https://github.com/qqgeogor/kaggle-quora-solution-8th https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question <br>
>(3) 深度学习模型(1DCNN, Esim, Decomp Attention，ELMO等等)： https://www.kaggle.com/rethfro/1d-cnn-single-model-score-0-14-0-16-or-0-23/notebook https://www.kaggle.com/lamdang/dl-models/comments 更多文本匹配模型见斯坦福SNLI论文集：https://nlp.stanford.edu/projects/snli/ <br>
>(4) 文本分类思想：主要是如何组织输入文本？另外query_prediction权重考虑？ 传统特征：tfidf，bow，ngram+tfidf，sent2vec，lsi，lda等特征 <br>
>(5) 深度学习模型： 参考知乎看山杯(知乎)以及Kaggle Toxic比赛<br>
>>https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge<br>
>>https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557<br>
>>https://www.kaggle.com/larryfreeman/toxic-comments-code-for-alexander-s-9872-model/comments<br>
>>https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52702<br>

>(6) Stacking无效(模型个数限制)，简单Blending，NN+LightGBM的方案比较靠谱？<br>
>(7) PS1：词向量可使用word2vec训练或者使用公开词向量数据：https://github.com/Embedding/Chinese-Word-Vectors PS2：分词需要加上自定义词典，分词质量对模型训练很重要！


6 基本思考
======
>(1)：如何选用一些泛化能力分类器 -> logistic regression; support vector machine; linear regression<br>
>(2)：如何构造文本特征 -> nlp分析<br>
>(3)：如何解决特征稀疏问题 -> deep-fm<br>
