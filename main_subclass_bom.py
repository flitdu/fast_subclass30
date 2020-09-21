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
import time
import fasttext as ff
# # from fastText.bui
# from fastText.build import fasttext as ff
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from data_operation.function import get_logger
from bom_read import excel_read2txt, OperateExcelSubclass
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
from data_operation.constant import label_name_forbid, SubclassLabelList, rule_dict, re_match, dic_match, SUBCLASS2ENTITY,re_match_entity
from styleframe import StyleFrame, Styler, utils

stop_words = load_stop_word_list("stopwords_subclass.txt")


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
                if subclass_label_i =='排阻' and bool(re.search(r'\b0603\b', content)):
                    continue
                elif subclass_label_i =='采样电阻' and bool(re.search(r'\b1 / \d*w', content)):
                    continue
                elif subclass_label_i =='压敏电阻' and bool(re.search(r'\b\d+k\d+\b', content)):  #压敏电阻没有阻值
                    continue
                elif subclass_label_i =='可调电阻电位器' and bool(re.search(r'\b0805\b|\b0603\b|\b805\b|\b603\b', content)):
                    continue

            elif entity_label == '连接器':
                if bool(re.search(r'\b(ph|vh|xh|zh)\d+\.?\d*', content)):  # 正则匹配到
                    return 1, '线对板线对线连接器'
                elif bool(re.search(r'\bheader\b', content)):
                    if subclass_label_i in ['线对板线对线连接器', 'IDC连接器(牛角)']:
                        return 1, subclass_label_i
                    else:
                        continue
                elif bool(re.search(r'\bsim卡', content)):
                    return 1, '内存连接器'
                elif subclass_label_i =='排针排母' and bool(re.search(r'(带锁|自锁)', content)):
                    continue

            elif entity_label == '电容':
                if bool(re.search(r'(\bb 型\b|\bc 型\b|\bd 型\b)', content)):
                    return 1, '钽电容'
                elif subclass_label_i == '贴片电容' and bool(re.search(r'\b5\s\*\s5.4\b', content)):  # 封装不对
                    continue

            if SUBCLASS2ENTITY[subclass_label_i] == entity_label:  # 直接输出
                tag = 1
                return tag, subclass_label_i

        return 0, None


    def predict_rewrite_excel(self, model, entityPredicitonModel):
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
                predicted_result = entiyPredictOutput(aa_description_standard, entityPredicitonModel)
            print('####', predicted_result)
            entity_predicted_label = predicted_result[0][0][0].replace('__label__', '')
            entity_predicted_probability = format(predicted_result[1][0][0], '.2f')  # 保留2位小数
            print('\033[1;36m  二级分类--{}：\033[0m {:.2f} !!!!!!'.format(entity_predicted_label, float(entity_predicted_probability)))

            if float(entity_predicted_probability) > 0.9:
                # 不考虑的如下（语料太少）：
                check_entity = {'嵌入式外围芯片':1,'射频无线电':2,'线材配件':3,'变压器':4,'继电器':5}
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
            predicted_label_lists.append(predicted_label)
            predicted_probability_list.append(predicted_probability)

        return predicted_label_lists, predicted_probability_list


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
            predict_labels, predict_probabilities = aa.predict_rewrite_excel(prediciton_model, entityPredicitonModel)  # 预测 excel 一行,除去标签
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



