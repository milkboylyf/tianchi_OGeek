# -*- coding: utf-8 -*-
"""
Created on 2018/10/17 9:21
@Author: Johnson
@Email:593956670@qq.com
@Software: PyCharm
"""
import os
import sys
import pyltp

personal_seg_dict = './tmp_file'
ltp_models_dir = 'D:/GithubRepos/gitdata/tcgame_ogeek/ltp_data_v3.4.0'

model_files = os.listdir(ltp_models_dir)
ltp_models = {os.path.splitext(fname)[0]:os.path.join(ltp_models_dir,fname) for fname in model_files}

sensplit = pyltp.SentenceSplitter.split
segmentor_ = None
postagger_ = None
ner_ = None
parser_ = None
srl_ = None


def segment(sentence):
    global segmentor_
    if segmentor_ is None:
        segmentor_ = pyltp.Segmentor()
        #segmentor_.load(ltp_models['cws'])
        # 加载模型，第二个参数是您的外部词典文件路径
        segmentor_.load_with_lexicon(ltp_models['cws'], personal_seg_dict)
    return segmentor_.segment(sentence)


def postag(words):
    global postagger_
    if postagger_ is None:
        postagger_ = pyltp.Postagger()
        postagger_.load(ltp_models['pos'])
    return postagger_.postag(words)


def ner(words, postags):
    global ner_
    if ner_ is None:
        ner_ = pyltp.NamedEntityRecognizer()
        ner_.load(ltp_models['ner'])
    return ner_.recognize(words, postags)


def parse(words, postags):
    global parser_
    if parser_ is None:
        parser_ = pyltp.Parser()
        parser_.load(ltp_models['parser'])
    return parser_.parse(words, postags)


def srl(words, postags, arcs):
    global srl_
    if srl_ is None:
        srl_ = pyltp.SementicRoleLabeller()
        srl_.load(ltp_models['pisrl_win'])
    return srl_.label(words, postags, arcs)


def release():
    global segmentor_, postagger_, ner_, parser_, srl_
    if segmentor_ is not None:
        segmentor_.release()
    if postagger_ is not None:
        postagger_.release()
    if ner_ is not None:
        ner_.release()
    if parser_ is not None:
        parser_.release()
    if srl_ is not None:
        srl_.release()