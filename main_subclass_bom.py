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
import matplotlib.pyplot as plt
from data_operation.function import get_logger
from bom_read import excel_read2txt
from data_selection_new import merge_txt_files
from data_split import train_datas_split
import pandas as pd
from data_operation import OperateExcel
import time
from data_operation.function import load_stop_word_list, standard, label_new
from data_operation.txt_operate import OperateTXT
import os
from data_operation.constant import label_name_forbid
stop_words = load_stop_word_list("stopwords_subclass.txt")


class FastTextModel:
    def __init__(self, epoch, loss, learn_rate):
        '''
        初始化网络，设置损失函数，迭代数///
        '''
        self.epoch = epoch
        self.loss = loss
        self.lr = learn_rate
        self.n_gram = 2
        pass

    def fit(self, train_file_path):
        '''
        依据训练数据不断更新权重
        '''
        for i in range(1, self.epoch):  # 迭代轮数
            w = self.n_gram
            #for w in range(2, 3):  # 连词数，取1、2
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
            # --------------------------------
            # classifier = ff.train_supervised(train_file_path,
            #                                   epoch=i,
            #                                   loss=self.loss,
            #                                   lr=self.lr,
            #                                   wordNgrams=w)

            # print(classifier.)
            print("ngram=%d,训练第%d轮，用时%s" % (w, i, time.time() - start_time))
            classifier.save_model(r"D:\dufy\code\ft_BOM\model\model_w" + str(w) + "_e" + str(i))
            print('============训练进度{:.2}============='.format((i - 1)/(self.epoch - 2)))
        print('训练完成......')

    def evaluate(self, train_file_path, test_file_path):
        '''
        调参
        :return:
        '''
        plot_x_epoch = list(range(2, self.epoch))
        # 加载测试数据
        correct_labels = []
        texts = []
        test_accuracy = []
        train_accuracy = []
        test_f1 = []
        with open(test_file_path, "r", encoding="utf-8") as ft_test:
            for line in ft_test:
                # print(line)
                correct_labels.append(line.strip().split(" , ")[0])
                texts.append(line.strip().split(" , ")[1])
        # print('correct_labels 为：{}'.format(correct_labels))
        # 加载分类模型
        #for w in range(1, 2):
        for i in range(2, self.epoch):
            w = self.n_gram
            classifier = ff.load_model(r"D:\dufy\code\ft_BOM\model\model_w" + str(w) + "_e" + str(i))
            # print("Model/model_w" + str(w) + "_e" + str(i))
            # 预测
            # classifier.get_word_vector()
            predict_labels = classifier.predict(texts)[0]
            # print(dir(classifier),';;;;;;;')
            # print('测试集predict_labels 为：', predict_labels, type(predict_labels))
            print(confusion_matrix(correct_labels, predict_labels,
                                   labels=label_list))
            f1_score = metrics.f1_score(correct_labels, predict_labels,
                                        average='weighted')
            print('\033[1;32m 测试集F1: {:.3}\033[0m'.format(f1_score))
            test_f1.append(f1_score)
            # 计算预测结果
            # print(len(texts))
            accuracy_num = 0
            for j in range(len(texts)):
                if predict_labels[j][0] == correct_labels[j]:
                    # print(predict_labels[j][0], correct_labels[j], '===~~~~~~~')
                    accuracy_num += 1

            accuracy = accuracy_num / len(texts)
            test_accuracy.append(accuracy)
            # print("正确率：%s" % accuracy)
            print('Model/model_w{}_e{}正确率：{:.2}'.format(w, i, accuracy))
            print('=====分隔符======')
        print(test_accuracy, test_f1)  # 包括了n1, 和n2
        test_accuracy_n1 = test_accuracy
        # test_accuracy_n2 = test_accuracy[(epoch_ - epoch_begin):]
        test_f1_n1 = test_f1
        # test_f1_n2 = test_f1[(epoch_ - epoch_begin):]
        # ====================训练数据==================
        print('计算训练数据......')
        correct_labels_train = []
        texts1 = []
        # with open("fasttext.train.txt", "r", encoding="utf-8") as ft_train:
        # with open("train_split_data.txt", "r", encoding="utf-8") as ft_train:
        with open(train_file_path, "r", encoding="utf-8") as ft_train:
            for line in ft_train:
                # print(line, '--------------------')
                # print(line.strip().split(" , ")[0], '--------------------')
                try:
                    correct_labels_train.append(line.strip().split(" , ")[0])
                    texts1.append(line.strip().split(" , ")[1])
                except:
                    continue
        # print('correct_labels 为：{}'.format(correct_labels_train))
        # 加载分类模型
     #  for w in range(1, 2):
        for i in range(2, self.epoch):
            w = self.n_gram
            classifier = ff.load_model(r"D:\dufy\code\ft_BOM\model\model_w" + str(w) + "_e" + str(i))
            # print("Model/model_w" + str(w) + "_e" + str(i))
            # 预测
            predict_labels = classifier.predict(texts1)[0]
            # print(predict_labels,'--------------------')
            # 计算预测结果
            # print(len(texts))
            accuracy_num = 0
            for j in range(len(texts1)):
                # print(predict_labels[j][0],correct_labels_train[j],'--------------------')
                if predict_labels[j][0] == correct_labels_train[j]:
                    accuracy_num += 1
            # print(accuracy_num,'--------------------')
            accuracy = accuracy_num / len(texts1)
            train_accuracy.append(accuracy)
            # print("训练集正确率：%s" % accuracy)
            print('训练集Model/model_w{}_e{}正确率：{:.2}'.format(w, i, accuracy))

        train_accuracy_n1 = train_accuracy
        # train_accuracy_n2 = train_accuracy[(epoch_ - epoch_begin):]
        plt.figure()
        plt.plot(plot_x_epoch, test_accuracy_n1, color="r", linestyle="-", marker="^", linewidth=1,
                 label="validation_accu")
        plt.plot(plot_x_epoch, test_f1_n1, color="b", linestyle="-", marker="o", linewidth=1, label="validation_f1")
        plt.plot(plot_x_epoch, train_accuracy_n1, color="k", linestyle="-", marker="s", linewidth=1, label="train_accu")
        # plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("{}-gram".format(w))
        plt.grid()
        time_temp = time.strftime("(T%Y-%m-%d-%H-%M)", time.localtime())
        plt.savefig(r"D:\dufy\code\fast_subclass30\pic\{}.png".format(time_temp+"rate" + str(self.lr) + '_'+str(w)+'-gram_' + self.loss))
        # plt.savefig("rate" + str(self.lr) + '_'+str(w)+'-gram_' + self.loss + ".png")
        # plt.figure()
        plt.show()
        logger.info('训练结束...')


def predict_output(str1):
    print('前3预测： ', classfier.predict([str1], k=3))
    predict = classfier.predict([str1])
    print(predict)
    return predict


class TestExcel(OperateExcel):  # 重写函数
    pass

    def predict_result(self): # 处理单个文件
        true_false_list = []
        _, row = self.excel_matrix() # 读取列
        row = list(range(1, row))
        if row != []:
            j = 0
            for line_read in self.excel_content_all().splitlines():  # 先遍历行
                j += 1
                true_label = line_read.split()[0].replace('/', '')  # 替换标签里面 '/'
                if true_label in label_name_forbid:
                    continue
                true_label = label_new(true_label)

                if true_label != 'nan':
                    print('#{}{}:'.format(j, true_label))
                    # print('\033[1;32m # {}\033[0m,excel原始输入：{}'.format(0, line_read))
                    aa_description = " ".join(line_read.split()[1:])
                    aa_description_standard = standard(aa_description, stop_words)  # 标准化处理
                    predicted_label_name = predict_output(aa_description_standard)
                else:
                    continue

                predicted_label = predicted_label_name[0][0][0].replace('__label__', '')
                if true_label == predicted_label:
                    true_false_list.append(1)
                    print("预测实体为：\033[1;32m {} {}\033[0m".format(predicted_label, '√'))

                else:
                    print('\033[1;31m error!!【{}】\033[0m预测为\033[1;31m 【{}】\033[0m]'.format(
                        true_label, predicted_label))
                    print(self.file_path)
                    error_infor = true_label + '     预测为     ' + predicted_label
                    OperateTXT().txt_write_line(r'D:\dufy\code\fast_subclass30\test\aaa.txt', error_infor)
                    OperateTXT().txt_write_line(r'D:\dufy\code\fast_subclass30\test\bbb.txt',
                                   '__label__' + true_label + ' , ' + aa_description_standard)
                    OperateTXT().txt_write_line(r'D:\dufy\code\fast_subclass30\test\ccc.txt', '__label__' + true_label + ' , ' + aa_description)
                    true_false_list.append(0)
                    print("预测实体为：\033[1;31m {} {}\033[0m".format(predicted_label, '×'))
                print('========================')
            return true_false_list
        else:
            return None


if __name__ == '__main__':
    logger = get_logger()

    # 1 读取excel写入不同的标签txt
    # 读取txt路径：ft_BOM\data\bom_subclass30'，  写入r'D:\dufy\code\ft_BOM\data\excel_write'
    # '''''''''''''''''bom_read.py

    # excel_read2txt()

    train_tag = 100
    if train_tag == 1:
        # # 2 读取上一步不同txt 融合，写入'selection_data.txt'
        # # '''''''''''''''''data_selection_new.py

        label_list = merge_txt_files(1500, shuffle_tag=1)  ## 选取行数

        # # # 3 划分数据集, 读取selection_data.txt'， 写入：'test_split_data.txt' 与 ‘train_split_data.txt'
        # # # # # # # # # '''''''''''''''''data_split.py

        train_datas_split()

        with open(r'.\data\error_record.txt', 'r', encoding='utf-8') as file:
            # 读文件
            for line in file.readlines():
                OperateTXT().txt_write_line(r'.\data\train_split_data.txt', line.replace('\n', ''))

        # OperateTXT().txt_write_line(target_path_temp, aa_description)


    # 4 训练-调参
    # 初始化
        epoch_begin = 2
        epoch_ = 100
        loss_name = 'softmax'
        learn_rate = 0.5  # 0.5, 0.8

        ft_ = FastTextModel(epoch_, loss_name, learn_rate)
        ft_.fit(r'.\data\train_split_data.txt')  # 训练
        ft_.evaluate(r'.\data\train_split_data.txt', r'.\data\test_split_data.txt')   # 评价

    # ########## 5 测试
    test_flag = 1

    if test_flag == 0:
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
    #
    if test_flag == 1:
        tag = 1
        if tag == 1:
            excel_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\test00'
            excel_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\4.1'

            model_folder = r'D:\dufy\code\ft_BOM\model_1'  # 单个模型测试
            model_names = os.listdir(model_folder)
            # excel_test1(txt_names)
            dict_model_test = {}
            for i, name0 in enumerate(model_names):  # 文件夹下文件循环
                modle_path = model_folder + '\\' + name0
                classfier = ff.load_model(modle_path)

                f_train = open(r'D:\dufy\code\fast_subclass30\test\aaa.txt', 'w')
                f_train.truncate()
                f_train.close()
                f_test = open(r'D:\dufy\code\fast_subclass30\test\bbb.txt', 'w')
                f_test.truncate()
                f_test.close()
                f_test1 = open(r'D:\dufy\code\fast_subclass30\test\ccc.txt', 'w')  # 增加原始信息输出
                f_test1.truncate()
                f_test1.close()

                all_record = 0
                right_record = 0

                file_names = os.listdir(excel_path)
                for i, name1 in enumerate(file_names):
                    file_path_combine = excel_path + '\\' + name1
                    aa = TestExcel(file_path_combine)
                    TF_record = aa.predict_result()  # 预测 excel 一行,除去标签
                    print(TF_record)
                    # if TF_record != None and []:  # 之前这样写，不对！！！！
                    if TF_record:
                        print(name1)
                        print(TF_record, '~~~~~~~~~~~~~~~~')
                        print('正确率:{:.2f}'.format(sum(TF_record) / len(TF_record)))
                        all_record += len(TF_record)
                        for i in TF_record:
                            if i == 1:
                                right_record += 1
                    else:
                        print('{} 无法识别'.format(file_path_combine))
                    print(file_path_combine)
                    print('\033[1;32m =\033[0m' * 120)
                    pass
                print('标注数据量:{}'.format(all_record))
                print('预测正确量:{}'.format(right_record))
                print('测试集全部数据正确率:{:.2f}'.format(right_record / all_record))
                print('全部结束！！！！')
                dict_model_test[name0] = right_record / all_record  # 此处。。。。
            print(dict_model_test)

            x = []
            y = []
            for key, value in dict_model_test.items():
                print(value)
                #     print(key.strip('model_w1_e'), value)
                x.append(key.strip('model_'))  # append() 方法用于在列表末尾添加新的对象。
                y.append(value)

            plt.plot(x, y, "b-o", linewidth=2)
            plt.xlabel("model")  # X轴标签
            plt.ylabel("accu")  # Y轴标签
            plt.title("Line plot")  # 图标题
            plt.grid()
            plt.show()  # 显示图





