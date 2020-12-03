# -*- coding: utf-8 -*-
"""
Created by Dufy on 2019/12/2  11:00
IDE used: PyCharm
Description :
1)增加所有模型的预测效果
2)
Remark:
"""
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import fasttext as ff
# # from fastText.bui
# from fastText.build import fasttext as ff
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from data_operation.function import get_logger
from bom_read import excel_read2txt
from data_selection_new import mergeLabelTxt
from data_split import datasSplit
import pandas as pd
from data_operation import OperateExcel
import time
from data_operation.function import load_stop_word_list, standard, labelNewSubclass, path_clear
from data_operation.txt_operate import OperateTXT
import os, pickle
import numpy as np
from ft_plot import plotCompareModelAccuracy, plotScatterRightWrongMark, plotTrainEffect
np.set_printoptions(threshold=np.inf)
from data_operation.constant import label_name_forbid, re_match, dic_match, SUBCLASS2ENTITY,re_match_entity, pass_match
from styleframe import StyleFrame, Styler
from data_operation.segment_api import skk
stop_words = load_stop_word_list("stopwords_subclass.txt")
from data_operation.pn_class import createDic, createTRIE, load_dic, getClassFromPn
from data_operation.triee import Trie

class FastTextModel:
    def __init__(self, epoch, loss, learn_rate, n_gram):
        '''
        初始化网络，设置损失函数，迭代数///
        '''
        self.epoch = epoch
        self.loss = loss
        self.lr = learn_rate
        self.n_gram = n_gram

        pass
    def trainWithAllDatas(self, train_file_path):
        '''
        拿所有数据训练最终模型
        '''

        w = self.n_gram
        start_time = time.time()  # loss = softmax or hs
        classifier = ff.train_supervised(train_file_path,
                                         epoch=self.epoch,
                                         loss=self.loss,
                                         lr=self.lr,
                                         dim=35,
                                         wordNgrams=w,
                                         minCount=1,  # 词频阈值, 小于该值在初始化时会过滤掉
                                         minn=3,
                                         maxn=15)
        print("训练用时%s" % (time.time() - start_time))
        classifier.save_model(r"D:\dufy\code\local\model\ft_subclass\final_models\model" + "_e" + str(self.epoch))
        print('最终模型训练完成......')


    def train(self, train_file_path):
        '''
        依据训练数据不断更新权重
        '''
        for i in range(1, self.epoch):  # 迭代轮数
            w = self.n_gram
            start_time = time.time()  # loss = softmax or hs
            classifier = ff.train_supervised(train_file_path,
                                             epoch=i,
                                             loss=self.loss,
                                             lr=self.lr,
                                             dim=35,
                                             wordNgrams=w,
                                             minCount=1,  # 词频阈值, 小于该值在初始化时会过滤掉
                                             minn=3,
                                             maxn=15)
            print("ngram=%d,训练第%d轮，用时%s" % (w, i, time.time() - start_time))
            classifier.save_model(r"D:\dufy\code\local\model\ft_subclass\train_models\model_w" + str(w) + "_e" + str(i))
            print('============训练进度{:.2}============='.format((i - 1)/(self.epoch - 2)))
        print('训练完成......')

    @staticmethod
    def loadTrainData(file_path):
        """ 加载 BOM标注数据路径
        :return:标签和样本描述
        """
        pass
        correct_labels = []
        texts_content = []
        with open(file_path, "r", encoding="utf-8") as ft_test:
            for line in ft_test:
                correct_labels.append(line.strip().split(" , ")[0].replace('__label__',''))
                texts_content.append(line.strip().split(" , ")[1])
        return correct_labels, texts_content

    def evaluate(self, classifier_model, file_path):
        """
        评价模型效果
        :param classifier_model: 单个分类模型
        :param file_path:  Bom标注数据路径
        :return:
        """
        correct_labels, texts = self.loadTrainData(file_path)

        # print(f'验证集标签：{vali_correct_labels}')
        predict_labels = []
        model_predict_labels = classifier_model.predict(texts)[0]
        for i in model_predict_labels:
            predict_labels.append(i[0].replace('__label__', ''))
        # print(f'验证集预测标签：{vali_predict_labels}')
        # print(f'准确率计算：{accuracy_score(vali_correct_labels, vali_predict_labels)}')
        # print(f"f1宏平均：{metrics.f1_score(vali_correct_labels, vali_predict_labels, average='macro')}" )

        labels_ = list(set(correct_labels+predict_labels))

        # logger.debug(confusion_matrix(correct_labels, predict_labels,labels=labels_))
        # confusion_matrix_model_i = confusion_matrix(correct_labels, predict_labels,labels=labels_)
        # logger.debug(f'混淆矩阵：{confusion_matrix_model_i}')  # 横为预测，  竖为真实

        logger.debug('分类报告:')
        logger.debug(classification_report(correct_labels, predict_labels, target_names=labels_))

        return accuracy_score(correct_labels, predict_labels), metrics.f1_score(correct_labels, predict_labels, average='macro')

    @classmethod
    def test(cls, classifier_model, file_path):
        """
        测试集查看最终效果
        :param classifier_model: 单个分类模型
        :param file_path:  Bom标注数据路径
        :return:
        """
        correct_labels, texts = cls.loadTrainData(file_path)

        # print(f'验证集标签：{vali_correct_labels}')
        predict_labels = []
        model_predict_labels = classifier_model.predict(texts)[0]
        for i in model_predict_labels:
            predict_labels.append(i[0].replace('__label__', ''))
        # print(f'验证集预测标签：{vali_predict_labels}')
        # print(f'准确率计算：{accuracy_score(vali_correct_labels, vali_predict_labels)}')
        # print(f"f1宏平均：{metrics.f1_score(vali_correct_labels, vali_predict_labels, average='macro')}" )

        # label_list = SubclassLabelList.getLabel() # 想通过类加载，实现不了、、、
        f = open(r'.\data\variant\label_list.txt', 'rb')

        labels_ = list(set(correct_labels + predict_labels))
        logger.debug(confusion_matrix(correct_labels, predict_labels,labels=labels_))

        confusion_matrix_model_i = confusion_matrix(correct_labels, predict_labels,labels=labels_)

        logger.debug(f'混淆矩阵：{confusion_matrix_model_i}')  # 横为预测，  竖为真实

        logger.debug('分类报告:')
        logger.debug(classification_report(correct_labels, predict_labels, target_names=labels_))

        return accuracy_score(correct_labels, predict_labels), metrics.f1_score(correct_labels, predict_labels, average='macro')


def predict_output(str1, model, k0=1):
    print('前4预测： ', model.predict([str1], k=4))
    predict = model.predict([str1], k=k0)
    return predict

def entiyPredictOutput(str1, model):
    # 二级预测
    print('前3预测： ', model.predict([str1], k=3))
    predict = model.predict([str1])
    return predict


class TestExcel(OperateExcel):  # 重写函数
    def __init__(self, url):
        OperateExcel.__init__(self, url)

    def predict_result(self, model): # 处理单个文件
        true_false_list = []
        probability_list = []

        _, row = self.excel_matrix() # 读取列
        row = list(range(1, row))
        if row:
            j = 0
            for line_read in self.excel_content_all().splitlines():  # 先遍历行
                j += 1
                true_label = line_read.split()[0].replace('/', '')  # 替换标签里面 '/'
                if true_label in label_name_forbid:
                    continue
                true_label = labelNewSubclass(true_label)

                if true_label != 'nan':
                    print('#{}{}:'.format(j, true_label))
                    aa_description = " ".join(line_read.split()[1:])
                    aa_description_standard = standard(aa_description, stop_words)  # 标准化处理
                    predicted_result = predict_output(aa_description_standard, model)
                else:
                    continue

                predicted_label = predicted_result[0][0][0].replace('__label__', '')
                predicted_probability = predicted_result[1][0][0]
                print(predicted_probability, '!!!!!!')
                probability_list.append(predicted_probability)
                if true_label == predicted_label:
                    true_false_list.append(1)
                    print("预测实体为：\033[1;32m {} {}\033[0m".format(predicted_label, '√'))

                else:
                    print('\033[1;31m error!!【{}】\033[0m预测为\033[1;31m 【{}】\033[0m]'.format(
                        true_label, predicted_label))
                    print(self.file_path)
                    error_info = true_label + '     预测为     ' + predicted_label + ' '*10 + '概率:' + \
                                 str(predicted_probability) + f'    \Bom片段：【{aa_description_standard[:10]}】 ' + \
                                 str(self.file_path).replace(excel_path, '') + \
                                 f'{model.predict([aa_description_standard], k=3)}'.replace('__label__', '')
                    save_test_info(error_info, true_label, aa_description_standard, aa_description, predicted_probability)
                    true_false_list.append(0)
                    print("预测实体为：\033[1;31m {} {}\033[0m".format(predicted_label, '×'))
                print('========================')
            return true_false_list, probability_list
        else:
            return None, None

    @staticmethod
    def entityCheckLogic(content, sub_model, entity_label, number=4):
        """
        加入二级校验过程，默认更信任二级，防止出现三级预测不对应二级情况
        目前考虑加入 连接器', '电感'， '开关', '光电器件', '二极管',
        :param self:
        :param content: 规范化后的bom行
        :param sub_model: 三级模型
        :param entity_label: 实体预测类目
        :param number: 返回三级分类预测前 number 个
        :return: 校验是否生效、校验后的三级分类
        """
        predicted_result = predict_output(content, sub_model, number)
        for i in range(number):
            print(predicted_result[0][0][i].replace('__label__', ''), '@@@')
            subclass_label_i = predicted_result[0][0][i].replace('__label__', '')
            print(SUBCLASS2ENTITY[subclass_label_i], '###')
            if entity_label == '电阻':
                pattern = re.compile(r'\b\d+\.?\d* *%')   #精度匹配
                try:
                    string = pattern.findall(content)[0]  # '900ma'
                    magnitude = float(re.findall(r"\d+\.?\d*", string)[0])  # 量值

                    if magnitude<1:  # 精度＜1%
                        print('^^^^', magnitude)
                        return 1, '贴片电阻'
                    else:
                        if subclass_label_i =='金属膜电阻' and bool(re.search(r'\b603\b|\b0603\b|\b1206\b', content)):  #封装
                            continue
                        elif subclass_label_i =='采样电阻' and bool(re.search(r'\b\d+\.?\d*k', content)):  #阻值
                            continue
                except IndexError:
                    pass  # 保证dic_match互斥，所以用break
                r_pattern = re.compile(r'\b\d+\.?\d* *[ω|r]\b')  # 阻匹配
                try:
                    r_string = r_pattern.findall(content)[0]  # '900ohm'
                    r_magnitude = float(re.findall(r"\d+\.?\d*", r_string)[0])  # 量值
                    if r_magnitude >=1 and subclass_label_i =='采样电阻':
                        continue
                except IndexError:
                    pass
                r_pattern = re.compile(r'\b\d+\.?\d*\sm ω\b')  # 阻匹配，匹配MΩ
                try:
                    r_string = r_pattern.findall(content)[0]  # '900ohm'
                    r_magnitude = float(re.findall(r"\d+\.?\d*", r_string)[0])  # 量值
                    if r_magnitude and subclass_label_i == '保险电阻':  # 保险电阻没有MΩ
                        continue
                except IndexError:
                    pass
                if subclass_label_i =='采样电阻' and (bool(re.search(r'\b1 / \d*w', content)) or bool(re.search(r'\b\d+(k|m)\b', content))):
                    continue
                elif subclass_label_i =='压敏电阻' and bool(re.search(r'\s(\d+k\d+\b|\d+\.?\d*\s*(k|ohm)\b)', content)):  #压敏电阻没有阻值，匹配5k2|1.2k
                    continue
                elif subclass_label_i =='可调电阻电位器' and bool(re.search(r'\b0805\b|\b0603\b|\b805\b|\b603\b', content)):
                    continue
                elif (subclass_label_i =='碳膜电阻' or subclass_label_i =='金属氧化膜电阻' or\
                      subclass_label_i =='金属膜电阻' or subclass_label_i =='排阻' or subclass_label_i =='绕线电阻')\
                        and bool(re.search(r'\bsmd\b|\b0805\b|\b0603\b|\b805\b|\b603\b|\b1206\b', content)):
                    continue
                elif subclass_label_i =='高压电阻' and not bool(re.search(r'\b\d+\s*(v|kv)\b', content)):  # 需要有电压
                    continue

            elif entity_label == '连接器':
                if bool(re.search(r'(\b(ph|vh|xh|zh|sh)\d+\.?\d*)|\b(ph|vh|xh|zh|sh|wafer)\b', content)) or\
                        bool(re.search(r'\b(卧贴|立贴).*?1.25\b|(\b1.25.*?(卧贴|立贴)\b)', content)):  # 正则匹配到
                    return 1, '线对板线对线连接器'
                elif bool(re.search(r'\bheader\b', content)):
                    if subclass_label_i in ['线对板线对线连接器', 'IDC连接器(牛角)']:
                        return 1, subclass_label_i
                    else:
                        continue
                elif bool(re.search(r'\bsim卡', content)):
                    return 1, '内存连接器'
                elif bool(re.search(r'插拔 座', content)):
                    return 1, '插拔式连接器'
                elif subclass_label_i =='排针排母' and bool(re.search(r'(带锁|自锁)', content)):
                    continue
                elif bool(re.search(r'(\b(dr|dp|hdr|hdp)\s*\d+\b)', content)):
                    return 1, 'D-Sub连接器附件'
                elif bool(re.search(r'(\b(抽屉式|前锁|后锁|上 接|下 接|双面 接|上下 接)\b)', content)):
                    return 1, 'FFCFPC连接器'
                elif bool(re.search(r'(\b(micro\s*ab|micro\s*b|micro\s*a|type\s*c|usb\s*\d+\.?\d*)\b)', content)):
                    return 1, 'USB连接器'
                elif bool(re.search(r'(\b(btb|b2b)\b)', content)):
                    return 1, '板对板连接器'


            elif entity_label == '电容':
                if bool(re.search(r'(\bb 型\b|\bc 型\b|\bd 型\b|\b(3528|3216)\b)', content)) or bool(re.search(r'(钽)', content)):
                    return 1, '钽电容'
                elif subclass_label_i == '贴片电容' and bool(re.search(r'\b5\s\*\s5.4\b', content)):  # 封装不对
                    continue
                elif subclass_label_i == '直插电解电容' and bool(re.search(r'\b6.3 x5.7\b', content)):  # 封装不对
                    continue
                elif (subclass_label_i == '电容器阵列与网络' or subclass_label_i == '直插瓷片电容') and\
                        bool(re.search(r'\b0805|0603|1206\b', content)):  # 封装不对
                    continue
                elif (subclass_label_i == '贴片电解电容' or subclass_label_i == '薄膜电容') and bool(re.search(r'\b1206\b', content)):  # 封装不对
                    continue
                elif (subclass_label_i == '钽电容') and bool(re.search(r'\b1812\b', content)):  # 封装不对
                    continue

            elif entity_label == '传感器':
                if bool(re.search(r'(\bcompass\b)', content)):
                    return 1, '磁性传感器'
                elif bool(re.search(r'(\baccelerometer\b)', content)):
                    return 1, '加速度传感器'
            elif entity_label == '二极管':
                if bool(re.search(r'(\b红|橙|黄|绿|蓝|靛|紫\b)', content)):
                    return 1, '发光二极管'
            elif entity_label == '晶振':
                if bool(re.search(r'(\b有源\b)', content)):
                    return 1, '有源晶体振荡器'
                elif bool(re.search(r'(\b无源\b)', content)):
                    return 1, '无源晶体振荡器'
            elif entity_label == '线材配件':
                if bool(re.search(r'(\bobd\b)', content)):
                    return 1, '数据线信号线'

            if SUBCLASS2ENTITY[subclass_label_i] == entity_label:  # 直接输出
                tag = 1
                return tag, subclass_label_i

        return 0, None


    def predict_rewrite_excel(self, model, entityPredicitonModel, trie, class_pn_dic):
        """
        预测，重写excel
        :return: 返回预测标签和预测概率
        """
        predicted_label_lists = []
        predicted_probability_list = []
        for line_read in self.excel_content_all().splitlines():  # 先遍历行
            # split_symbol = ['_',
            #                 '-',
            #                 ',',
            #                 '/',
            #                 '（',
            #                 '）',
            #                 '(',
            #                 ')',
            #                 '"',
            #                 '，',
            #                 '\\',
            #                 ':',
            #                 '：',
            #                 '@',
            #                 '【',
            #                 '】',
            #                 ';',
            #                 '；']
            split_symbol = []
            aa_description = line_read

            tag = 0
            for expression in pass_match:  # 不对bom内容判断，自动忽略
                if bool(re.search(expression, aa_description)):  # 正则匹配到
                    tag = 1
                    predicted_label_lists.append('pass')
                    predicted_probability_list.append(0.01)  # 概率
                    print('----------------------------------------')
                    print(aa_description)
                    print('\033[1;36m  {}：\033[0m {:.2f} !!!!!!'.format('pass', float(0.01)))
                    break
            if tag:
                continue

            aa_description_standard = standard(aa_description, stop_words, split_symbol)  # 标准化处理

            # aa_description_standard = OperateExcelSubclass.removeDuplicates(aa_description_standard)  # 句子去重

            # 添加规则
            tag = 0
            # -------------  词典匹配
            for expression, label in dic_match.items():
                pass
                judge_tag = all(bool(x in aa_description_standard) for x in expression)  # 是否同时存在
                if judge_tag:  # 转入判断
                    if label == 'to_judge':
                        if ('μ h' in aa_description_standard or 'µ h' in aa_description_standard) and 'smd' in aa_description_standard:
                                pattern = re.compile(r'\b\d+\.?\d* *ma\b')
                                try:
                                    string = pattern.findall(aa_description_standard)[0]  # '900ma'
                                    number = float(re.findall(r"\d+\.?\d*", string)[0])  # 量值
                                except IndexError:
                                    break   # 保证dic_match互斥，所以用break
                                if number/1000 < 1:
                                    tag = 1
                                    label = '贴片电感'
                                    predicted_label_lists.append(label)
                                    predicted_probability_list.append(2.0)  # 概率
                                    print(label, 2.0, '!!!!!!')
                                    break
                                else:
                                    tag = 1
                                    label = '功率电感'
                                    predicted_label_lists.append(label)
                                    predicted_probability_list.append(2.0)  # 概率
                                    print(label, 2.0, '!!!!!!')
                                    break
                        elif 'uh' in aa_description_standard and 'a' in aa_description_standard:
                            pattern2 = re.compile(r'\b\d+\.?\d* *a\b')
                            try:
                                a_string = pattern2.findall(aa_description_standard)[0]  # '900a'
                                a_number = float(re.findall(r"\d+\.?\d*", a_string)[0])  # 量值
                            except IndexError:
                                break
                            if a_number >= 1:
                                tag = 1
                                label = '功率电感'
                                predicted_label_lists.append(label)
                                predicted_probability_list.append(2.0)  # 概率
                                print(label, 2.0, '!!!!!!')
                                break
                        elif '电感' in aa_description_standard and 'ma' in aa_description_standard:
                            pattern3 = re.compile(r'\b\d+\.?\d*\s*ma\b')
                            try:
                                a_string = pattern3.findall(aa_description_standard)[0]  # '900ma'
                                a_number = float(re.findall(r"\d+\.?\d*", a_string)[0])  # 量值
                            except IndexError:
                                break
                            if a_number >= 100:
                                tag = 1
                                label = '高频电感'
                                predicted_label_lists.append(label)
                                predicted_probability_list.append(2.0)  # 概率
                                print(label, 2.0, '!!!!!!')
                                break

                    else:
                        if bool(re.search(r'\b\d+\.?\d*mh\b', aa_description_standard)) and 'smd' in aa_description_standard:
                            tag = 1
                            label = '固定电感'
                            predicted_label_lists.append(label)
                            predicted_probability_list.append(2.0)  # 概率
                            print(label, 2.0, '!!!!!!')
                            break
            if tag:
                continue
            # -------------  正则匹配
            for expression, label in re_match.items():
                if bool(re.search(expression, aa_description_standard)):  # 正则匹配到
                    tag = 1
                    predicted_label_lists.append(label)
                    predicted_probability_list.append(2.0)  # 概率
                    print(label, 2.0, '!!!!!!')
                    break
            if tag:
                continue

            # 校验逻辑
            # -------------  二级正则匹配
            predicted_result = ([[]], [[]])
            flag = 1
            for expression, label in re_match_entity.items():
                if bool(re.search(expression, aa_description_standard)):
                    predicted_result[0][0].append(label)
                    predicted_result[1][0].append(2.0)  # 概率
                    flag = 0
                    break
            if flag:
                predicted_result = entiyPredictOutput(aa_description_standard, entityPredicitonModel)  # 二级结果
            print('####', predicted_result)
            entity_predicted_label = predicted_result[0][0][0].replace('__label__', '')
            entity_predicted_probability = format(predicted_result[1][0][0], '.2f')  # 保留2位小数
            print('\033[1;36m  二级分类--{}：\033[0m {:.2f} !!!!!!'.format(entity_predicted_label, float(entity_predicted_probability)))

            if float(entity_predicted_probability) > 0.9:
                # 不考虑的如下（语料太少）：
                check_entity = {'嵌入式外围芯片':1,'射频无线电':2,'变压器':4,'继电器':5}
                if not check_entity.get(entity_predicted_label):
                    tag, subclass_label = self.entityCheckLogic(aa_description_standard, model,entity_predicted_label, 10)
                    if tag:  # 校验生效
                        predicted_label_lists.append(subclass_label)
                        predicted_probability_list.append(1.5)  # 概率
                        print('\033[1;36m  {}：\033[0m {:.2f} !!!!!!'.format(subclass_label, float(1.5)))
                        continue

            predicted_result = predict_output(aa_description_standard, model)
            predicted_label = predicted_result[0][0][0].replace('__label__', '')
            predicted_probability = format(predicted_result[1][0][0],'.2f')  # 保留2位小数
            print('\033[1;36m  {}：\033[0m {:.2f} !!!!!!'.format(predicted_label, float(predicted_probability)))

            prob = float(predicted_probability)  # 标签概率
            if prob < 0.5:
                origin_input = aa_description.replace('nan','')  # 原始bom 输入
                print('启用型号预测模型......')
                label = TestExcel.pnClassJudge(origin_input, trie, class_pn_dic)
                if label:
                    predicted_label = label
                    predicted_probability = 3  # 标志为依据型号判断的类目
                    print('\033[1;31m  型号判别 ：{}：\033[0m {:.2f} !!!!!!'.format(predicted_label, float(predicted_probability)))
            predicted_label_lists.append(predicted_label)
            predicted_probability_list.append(predicted_probability)

        return predicted_label_lists, predicted_probability_list

    @staticmethod
    def pnClassJudge(bom_content, trie,class_pn_dic):
        """
        依据型号前缀进行类目推断
        :return:
        """
        pn_list = skk(bom_content)
        if pn_list:
            for i in pn_list:
                query_pn = i.replace('/partnum','')  # 待查询型号
                return getClassFromPn(query_pn, trie, class_pn_dic)
        else:
            return None


def save_test_info(error_info, true_label, aa_description_standard, aa_description, predicted_probability):
    pass
    OperateTXT().txt_write_line(r'.\test\aaa.txt', error_info)
    OperateTXT().txt_write_line(r'.\test\bbb.txt',
                                '__label__' + true_label + ' , ' + aa_description_standard)
    OperateTXT().txt_write_line(r'.\test\ccc.txt',
                                '__label__' + true_label + ' , ' + aa_description)
    if predicted_probability > 0.6:
        OperateTXT().txt_write_line(r'.\test\error_0.6.txt', error_info + '\n'
                                    +'__label__' + true_label + ' , '+aa_description+ '\n' +'======' )


if __name__ == '__main__':
    logger = get_logger()
    id_name_dict = {47: '32位微控制器-MCU',
                    -1: 'None',
                    48: '16位微控制器-MCU',
                    49: 'ARM微控制器-MCU',
                    50: '微处理器-MPU',
                    51: '其他处理器',
                    52: '数字信号处理器和控制器',
                    53: '8位微控制器-MCU',
                    54: 'CPLD-FPGA芯片',
                    55: '安全(加密)IC',
                    56: 'MCU监控芯片',
                    57: '时钟计时芯片',
                    58: '实时时钟芯片',
                    59: '字库芯片',
                    60: '时钟缓冲器驱动器',
                    61: '时钟发生器频率合成器',
                    62: 'FLASH存储器',
                    63: 'SRAM存储器',
                    64: 'EPROM存储器',
                    65: 'SDRAM存储器',
                    66: 'SDMicro-SDT-Flash卡',
                    67: 'PROM存储器',
                    68: 'EEPROM存储器',
                    69: 'DDRSSD存储器',
                    70: 'FPGA-配置存储器',
                    71: '时基芯片',
                    72: '信号开关多路复用解码器',
                    73: '锁存器',
                    74: '编解码芯片',
                    75: '4000系列逻辑芯片',
                    76: '74系列逻辑芯片',
                    77: '缓冲器驱动器接收器收发器',
                    78: '移位寄存器',
                    79: '专用逻辑芯片',
                    80: '触发器',
                    81: '多频振荡器',
                    82: '门极反相器',
                    83: '计数器除法器',
                    84: '电平转换移位器',
                    85: '隔离芯片',
                    86: '以太网芯片',
                    87: 'USB芯片',
                    88: 'RS485RS422芯片',
                    89: '直接数字合成器(DDS)',
                    90: 'RS232芯片',
                    91: 'LVDS芯片',
                    92: '传感器接口芯片',
                    93: 'LIN收发器',
                    94: '信号缓冲器中继器分配器',
                    95: 'IO扩展器',
                    96: '控制器',
                    97: '音频视频接口芯片',
                    98: '电信',
                    99: '接口专用芯片',
                    100: '串行接口芯片',
                    101: '触摸屏控制器',
                    102: 'CAN芯片',
                    103: '磁珠',
                    104: '有源滤波器',
                    105: 'RF滤波器',
                    106: '共模扼流圈滤波器',
                    107: '馈通式电容器',
                    108: 'EMIRFI滤波器',
                    109: '铁氧体磁芯与配件',
                    110: '信号调节器',
                    111: '线对板线对线连接器',
                    112: '排针排母',
                    113: '板对板连接器',
                    114: '背板连接器',
                    115: 'USB连接器',
                    116: '插拔式连接器',
                    117: '螺丝钉接线端子',
                    118: '弹簧式接线端子',
                    119: '轨道式接线端子',
                    120: '压接端子',
                    121: 'IO连接器',
                    122: 'D-Sub连接器附件',
                    123: '音频视频链接器',
                    124: 'IEEE1394连接器',
                    125: '连接器附件套件',
                    126: '照明连接器',
                    127: 'RF同轴连接器',
                    128: '栅栏式接线端子',
                    129: 'FFCFPC连接器',
                    130: '圆形连接器',
                    131: 'IDC连接器(牛角)',
                    132: '汽车连接器',
                    133: '电源连接器',
                    134: '军工连接器',
                    135: '以太网连接器',
                    136: 'IC与器件插座',
                    137: '内存连接器',
                    138: '卡缘连接器',
                    139: '鳄鱼夹测试夹',
                    140: '贴片电感',
                    141: '高频电感',
                    142: '工字电感',
                    143: '功率电感',
                    144: '色环电感',
                    145: '固定电感',
                    146: '可变电感器套件配件',
                    147: '贴片电容',
                    148: '贴片电解电容',
                    149: '直插电解电容',
                    150: '直插瓷片电容',
                    151: '云母电容',
                    152: '安装型大容量电容',
                    153: '固态电解电容',
                    154: '钽电容',
                    155: '安规电容',
                    156: '薄膜电容',
                    157: '可调电容',
                    159: '超级电容器',
                    160: '氧化铌电容',
                    161: '电容器阵列与网络',
                    162: '电容套件及附件',
                    163: '直插独石电容',
                    164: '校正电容',
                    165: '铝电解电容',
                    166: 'NTC热敏电阻',
                    167: '贴片电阻',
                    169: '采样电阻',
                    170: '光敏电阻',
                    171: '压敏电阻',
                    172: 'MELF晶圆电阻',
                    173: '金属膜电阻',
                    174: 'PTC热敏电阻',
                    175: '金属氧化膜电阻',
                    176: '碳膜电阻',
                    177: '薄膜电阻-透孔',
                    178: '厚膜电阻-透孔',
                    179: '铝壳大功率电阻',
                    180: '射频高频电阻',
                    181: '陶瓷复合电阻器',
                    182: '可调电阻电位器',
                    183: '精密可调电阻',
                    184: '绕线电阻',
                    185: '保险电阻',
                    186: '水泥电阻',
                    187: '金属箔电阻',
                    188: '高压电阻',
                    189: '排阻',
                    190: '直插通孔电阻',
                    191: '电阻套件及附件',
                    192: '贴片高精密-低温漂电阻',
                    193: '无源晶体振荡器',
                    194: '有源晶体振荡器',
                    195: '圆柱体晶振',
                    196: '谐振器',
                    197: '可编程振荡器',
                    198: '标准时钟振荡器',
                    199: '稳压二极管',
                    200: '通用二极管',
                    201: '功率二极管',
                    202: '开关二极管',
                    203: '超快快恢复二极管',
                    204: '肖特基二极管',
                    205: '双向触发二极管DIAC',
                    206: 'TVS二极管(瞬态电压抑制二极管)',
                    207: 'ESD二极管',
                    208: '变容二极管VaractorDiode',
                    209: '稳流二极管CRD',
                    210: 'PIN二极管',
                    211: '整流器',
                    212: '整流桥',
                    213: '放电管',
                    214: '特殊功能放大器',
                    215: '视频放大器',
                    216: '放大器',
                    217: '高速宽带运放',
                    218: '仪表运放',
                    219: '精密运放',
                    220: 'FET输入运放',
                    221: '低噪声运放',
                    222: '低功耗比较器运放',
                    223: '差分运放',
                    224: '电压比较器',
                    225: '采样保持放大器',
                    226: 'LCDGamma缓冲器',
                    227: 'LED驱动',
                    228: 'LCD驱动',
                    229: '电机马达点火驱动器IC',
                    230: 'MOS驱动',
                    231: '激光驱动器',
                    232: '达林顿晶体管阵列驱动',
                    233: '驱动芯片',
                    234: '门驱动器',
                    235: '全桥半桥驱动',
                    236: '电子辅料',
                    237: '风扇散热片热管理产品',
                    238: '焊接脱焊',
                    239: '罩类盒类及壳类产品',
                    240: '胶带标签',
                    241: '容器类',
                    242: '螺丝刀镊子扳手工具',
                    243: '化学物质',
                    244: 'PCB等原型产品',
                    245: '配件',
                    246: '螺丝紧固件硬件',
                    247: '机架机柜',
                    248: '机架/机柜',
                    249: '线性稳压芯片LDO',
                    250: '开关电源芯片',
                    251: 'DC-DC芯片',
                    252: '电池电源管理芯片PMIC',
                    253: '电池保护芯片',
                    254: '电压基准芯片',
                    255: '电源监控芯片',
                    256: '功率开关芯片',
                    257: 'DC-DC电源模块',
                    258: 'AC-DC电源模块',
                    259: '无线充电IC',
                    260: 'LEDUPS等其他类型电源模块',
                    261: '开发板套件',
                    262: 'WiFi物联网模块',
                    263: 'WiFi/物联网模块',
                    264: '无线模块',
                    265: '传感器模块',
                    266: '电力线滤波器模块',
                    267: '其他模块',
                    268: '蜂鸣器',
                    269: '扬声器喇叭',
                    270: '咪头麦克风',
                    272: '电源变压器',
                    273: '电流变压器',
                    274: '网口变压器',
                    275: '工业控制变压器',
                    276: '脉冲变压器',
                    277: '音频及信号变压器',
                    278: '自耦变压器',
                    279: '信号继电器',
                    280: '继电器插座配件',
                    281: '车用继电器',
                    282: '固态继电器',
                    283: '安全继电器',
                    284: '簧片继电器',
                    285: '高频射频继电器',
                    286: '延时计时继电器',
                    287: '工业继电器',
                    288: '按键开关',
                    289: '船型开关',
                    290: '行程开关',
                    291: '拨码开关',
                    292: '拨动开关',
                    293: '五向开关',
                    294: '多功能开关',
                    295: '锅仔片',
                    296: '旋转编码开关',
                    297: '开关插座',
                    298: '带灯开关',
                    299: '旋转波段开关',
                    300: '交流接触器',
                    301: '压接接触器',
                    302: '专用开关',
                    303: '开关配件-盖帽',
                    304: '轻触开关',
                    305: '发光二极管',
                    306: '光耦',
                    307: '红外发射管',
                    308: '红外接收管',
                    309: 'LED显示模组',
                    310: 'LED数码管',
                    311: 'LCD显示模组',
                    312: '光可控硅',
                    313: 'OLED显示模组',
                    314: 'LED灯柱导光管配件',
                    315: '真空荧-VFD光显示器',
                    316: '等离子体显示器',
                    317: '红外收发器',
                    318: '光纤收发器',
                    319: '光电开关',
                    320: '激光器件配件',
                    321: '温度传感器',
                    322: '超声波传感器',
                    323: '气体传感器',
                    324: '光学传感器',
                    325: '压力传感器',
                    326: '颜色传感器',
                    327: '图像传感器',
                    328: '环境光传感器',
                    329: '红外传感器',
                    330: '角速度传感器',
                    331: '加速度传感器',
                    332: '角度传感器',
                    333: '位置传感器',
                    334: '姿态传感器',
                    335: '磁性传感器',
                    336: '电流传感器',
                    337: '湿度传感器',
                    338: '专用传感器',
                    339: '模数转换芯片',
                    340: '数模转换芯片',
                    341: '模拟开关芯片',
                    342: '电流监控芯片',
                    343: '电量计芯片',
                    344: '数字电位器芯片',
                    345: '电池',
                    346: '电源充电器',
                    347: '电池座夹附件',
                    348: '线材配件附件',
                    349: '数据线信号线',
                    350: '电源线',
                    351: '多芯电缆',
                    352: '同轴电缆',
                    353: '电子线材连接线',
                    354: '贴片式一次性保险丝',
                    355: 'PTC自恢复保险',
                    356: '通孔型保险丝',
                    357: '保险丝管',
                    358: '工业与电气保险丝',
                    359: '汽车保险丝',
                    360: '特种保险丝',
                    361: '温度保险丝',
                    362: '保险丝座夹',
                    363: '断路器',
                    364: 'MOSFET',
                    365: '结型场效应晶体管(JFET)',
                    366: '可控硅SCR',
                    367: '达林顿管',
                    368: '数字三极管',
                    369: 'IGBT管',
                    370: '双极晶体管(三极管)',
                    371: 'LAN电信电缆测试',
                    372: '万用表与电压表',
                    373: '测试与测量',
                    374: '仪器设备与配件',
                    375: '无线收发芯片',
                    376: '射频卡芯片',
                    377: '天线',
                    378: '射频开关',
                    379: 'RF放大器',
                    380: 'RF混频器',
                    381: 'RF检测器',
                    382: 'RF衰减器',
                    383: 'RF耦合器',
                    385: 'RFFETMOSFET',
                    409: '第一次导入',
                    411: '11是发广告的的股份',
                    412: '机电电气',
                    413: '高频继电器',
                    414: '通信卫星定位模块',
                    415: 'RF双工器',
                    416: 'IGBT驱动',
                    417: 'FRAM存储器',
                    418: '温控开关',
                    419: '温湿度传感器',
                    420: '电路保护套件',
                    422: 'LED管'}
    # createDic()  # 只需构建一次
    trie_ = Trie()
    createTRIE(trie_)
    print('^^^^', trie_.search('atmega64rzav-10pu'.upper()))
    class_pn_dic = load_dic()  # {pn:id}
    # print('TRIE树构建完成')

    excel_read_tag = 10
    if excel_read_tag == 1:
        excel_read2txt()

# ===============训练========================
    epoch_begin = 2
    epoch_ = 130  # 100
    loss_name = 'softmax'
    learn_rate = 0.5  # 0.5, 0.8
    n_gram = 2

    train_tag = 10
    if train_tag == 1:
        # 2 读取上一步不同txt 融合，写入'selection_data.txt'
        label_list = mergeLabelTxt(1500000, shuffle_tag=1)  ## 选取行数
        # SubclassLabelList.setLabel(label_list)

        f = open(r'.\data\variant\label_list.txt', 'wb')
        pickle.dump(label_list, f)
        f.close()

        # # # 3 划分数据集
        test_number = 4000  # 测试集序号索引0--test_index
        vali_number = 2000
        print('开始划分数据...')
        time0 = time.time()
        datasSplit(test_number, vali_number)
        print(f'划分数据 耗时: {time.time()-time0}')


        # 读取误分类数据到训练集
        # with open(r'.\data\error_record.txt', 'r', encoding='utf-8') as file:
        #     for line in file.readlines():
        #         OperateTXT().txt_write_line(r'.\data\corpus\train_data.txt', line.replace('\n', ''))

        # 4 训练-评价

        ft_ = FastTextModel(epoch_, loss_name, learn_rate, n_gram)
        ft_.train(r'.\data\corpus\train_data.txt')  # 训练

        train_accuracy_list = []  # 准确率
        train_f1_macro_list = []  # f1 宏平均
        val_accuracy_list = []
        val_f1_macro_list = []

        for i in range(1, epoch_):
            pass
            try:
                w = ft_.n_gram
                classifier_model_i = ff.load_model(r"D:\dufy\code\local\model\ft_subclass\train_models\model_w" + str(w) + "_e" + str(i))
                logger.debug('============')
                logger.debug(f'epoch_{i}, 训练集: ')
                accuarcy, f1_score = ft_.evaluate(classifier_model_i, r'.\data\corpus\train_data.txt')
                train_accuracy_list.append(accuarcy)
                train_f1_macro_list.append(f1_score)

                logger.debug('============')
                logger.debug(f'epoch_{i}, 验证集: ')
                accuarcy, f1_score = ft_.evaluate(classifier_model_i, r'.\data\corpus\vali_data.txt')
                val_accuracy_list.append(accuarcy)
                val_f1_macro_list.append(f1_score)
            except ValueError as err:
                print(err)
                continue

        logger.debug(f'训练集准确率：{train_accuracy_list}')
        logger.debug(f'训练集f1：{train_f1_macro_list}')
        logger.debug(f'验证集准确率：{val_accuracy_list}')
        logger.debug(f'验证集f1：{val_f1_macro_list}')

        # 绘制训练集和验证集数据比较
        plotTrainEffect(ft_,train_accuracy_list,train_f1_macro_list,val_accuracy_list,val_f1_macro_list)

    # ===============测试集========================
    test_tag = 10  # 查看测试集效果
    if test_tag == 1:
        pass
        test_accuracy_list = []
        test_f1_macro_list = []
        dict_model_test_accu = {}
        dict_model_test_f1 = {}
        model_folder = r'D:\dufy\code\local\model\ft_subclass\test_models'
        model_names = os.listdir(model_folder)

        for i, name0 in enumerate(model_names):  # 文件夹下文件循环
            modle_path = model_folder + '\\' + name0
            classifier_model_i = ff.load_model(modle_path)
            logger.debug('============')
            logger.debug(f'测试集: ')
            accuarcy, f1_score = FastTextModel.test(classifier_model_i, r'.\data\corpus\test_data.txt')
            test_accuracy_list.append(accuarcy)
            test_f1_macro_list.append(f1_score)
            dict_model_test_accu[name0] =accuarcy
            dict_model_test_f1[name0] =f1_score

        print(f'测试集准确率：{test_accuracy_list}')
        print(f'测试集f1：{test_f1_macro_list}')
        plotCompareModelAccuracy(dict_model_test_accu)

    # ===============利用所有数据重新训练得到最终模型========================
    trian_with_alldatas = 10
    if trian_with_alldatas == 1:
        print('使用全部数据开始重新训练....')
        ft_ = FastTextModel(180, loss_name, learn_rate, n_gram)
        ft_.trainWithAllDatas(r'.\data\selection_data_shuffle.txt')  # 训练

    # ===============BOM测试========================
    test_flag = 0
    time0 = time.time()
    if test_flag == 0:  # 对不带有标注的excel 预测
        pass
        modle_path = r'D:\dufy\code\local\model\ft_subclass\test_rewrite_models\model_e180'  #
        entity_modle_path = r'D:\dufy\code\local\model\ft_entity\final_models\model_e162'  # 二级分类模型
        excel_path = r'D:\dufy\code\local\corpus\bom_subclass\bom_test'
        output_path = r'D:\dufy\code\local\corpus\bom_subclass\bom_test_output'

        dict_model_test = {}
        record_right_probability_list = []
        record_wrong_probability_list = []
        file_names = os.listdir(excel_path)
        prediciton_model = ff.load_model(modle_path)
        entityPredicitonModel = ff.load_model(entity_modle_path)  # 二级分类模型

        number = 0
        for i, name1 in enumerate(file_names):
            if '~$' in name1:
                continue
            number += 1
            bom_path = os.path.join(excel_path, name1)
            print(bom_path)

            aa = TestExcel(bom_path)
            predict_labels, predict_probabilities = aa.predict_rewrite_excel(prediciton_model, entityPredicitonModel, trie_, class_pn_dic)  # 预测 excel 一行,除去标签
            # print(predict_labels, predict_probabilities)

            df1 = pd.read_excel(bom_path)
            df1['预测类目'] = pd.DataFrame(predict_labels)
            df1['预测概率'] = pd.DataFrame(predict_probabilities)

            output_name = output_path + '\\' + 'output_'+name1

            # df1.to_excel(output_name)
            # ---------背景色设置-------
            sf = StyleFrame(df1)

            sf.apply_column_style(cols_to_style=["预测类目"],
                                  styler_obj=Styler(bg_color='yellow'),
                                  style_header=True)
            sf.apply_column_style(cols_to_style=["预测概率"],
                                  styler_obj=Styler(bg_color='green'),
                                  style_header=True)
            ew = StyleFrame.ExcelWriter(output_name)
            sf.to_excel(ew)
            ew.save()
        print(f'文件个数：{number}, 耗时{time.time() -time0}')
        print('done!!!!!!')

    if test_flag == 1:  # 对带有标注的excel 预测
        excel_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\test00'
        excel_path = r'D:\dufy\code\公测'

        model_folder = r'D:\dufy\code\local\model\ft_subclass\test_models'  #
        model_names = os.listdir(model_folder)

        dict_model_test = {}
        record_right_probability_list = []
        record_wrong_probability_list = []

        for i, name0 in enumerate(model_names):  # 文件夹下文件循环
            modle_path = model_folder + '\\' + name0
            prediciton_model = ff.load_model(modle_path)
            path_clear(r'.\test')

            all_record = 0
            right_record = 0

            file_names = os.listdir(excel_path)
            for i, name1 in enumerate(file_names):
                file_path_combine = excel_path + '\\' + name1
                aa = TestExcel(file_path_combine)

                TF_record, probability_record = aa.predict_result(prediciton_model)  # 预测 excel 一行,除去标签
                if probability_record:
                    for index, value in enumerate(TF_record):
                        if value:
                            record_right_probability_list.append(probability_record[index])
                        else:
                            record_wrong_probability_list.append(probability_record[index])

                if TF_record:
                    print('正确率:{:.2f}'.format(sum(TF_record) / len(TF_record)))
                    all_record += len(TF_record)
                    for i in TF_record:
                        if i == 1:
                            right_record += 1
                else:
                    print('{} 无法识别'.format(file_path_combine))
                    logger.info(f'有问题的excel；{file_path_combine}')
                print(file_path_combine)
                print('\033[1;32m =\033[0m' * 120)
                pass
            print('标注数据量:{}'.format(all_record))
            print('预测正确量:{}'.format(right_record))
            print('测试集全部数据正确率:{:.2f}'.format(right_record / all_record))
            print('全部结束！！！！')
            dict_model_test[name0] = right_record / all_record  # 此处。。。。
        print(dict_model_test)

        # 绘制分类正确、错误的散点图
        plotScatterRightWrongMark(record_wrong_probability_list, record_right_probability_list)
        # 绘制不同模型的准确率比较
        plotCompareModelAccuracy(dict_model_test)


    if test_flag == 100000:
        ft_vec = ff.load_model(r"D:\dufy\code\ft_BOM\model\model_w2_e98")
        print(ft_vec.get_word_vector('3v'))
        print(ft_vec.words)
        print(ft_vec.get_nearest_neighbors('3v'))
        print(ft_vec.get_nearest_neighbors('0402B104K160CT'.lower()))
        print(ft_vec.get_nearest_neighbors('0402B104K160Cj'.lower()))
        print(ft_vec.get_nearest_neighbors('50v-0402B104K160-xxxxxxx'.lower()))
        print(ft_vec.get_nearest_neighbors('50 v'.lower()))
        print(ft_vec.get_nearest_neighbors('50v'.lower()))
        print(ft_vec.get_nearest_neighbors('MCS0630-3R3MN2'.lower()))
        print(ft_vec.get_nearest_neighbors('AOD510'.lower()))
        print(ft_vec.get_nearest_neighbors('电器'.lower()))
        print(ft_vec.get_nearest_neighbors('MCS0630-3R3MN2'.lower()))
    # aa = ['v', 'p2sd0301000026']
    # for i in aa:
    #     print(classifier.get_word_vector(i),'=======')
    #
    # print(dir(classifier))
    # print(help(classifier.ge))



