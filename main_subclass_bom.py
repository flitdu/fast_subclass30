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
import fastText.FastText as ff
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import jieba                         # 组合使用】
jieba.load_userdict('dict_boom.txt') # 组合使用】
from jieba_used_1209 import excel_read2txt
from data_selection import merge_txts
from data_split import train_datas_split
import pandas as pd
import time
from func import load_stop_word_list, string_split_combine, txt_write_line, standard
import os


class FastTextModel:
    def __init__(self, epoch, loss, learn_rate):
        '''
        初始化网络，设置损失函数，迭代数///
        '''
        self.epoch = epoch
        self.loss = loss
        self.lr = learn_rate
        pass
    def fit(self, train_file_path):
        '''
        依据训练数据不断更新权重
        '''
        for i in range(2, self.epoch):  # 迭代轮数
            for w in range(1, 2):  # 连词数，取1、2
                start_time = time.time()            # loss = softmax or hs
                classifier = ff.train_supervised(train_file_path, epoch=i, loss=self.loss, lr=self.lr, wordNgrams=w)
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
                print(line)
                correct_labels.append(line.strip().split(" , ")[0])
                texts.append(line.strip().split(" , ")[1])
        print('correct_labels 为：{}'.format(correct_labels))
        # 加载分类模型
        for w in range(1, 2):
            for i in range(2, self.epoch):
                classifier = ff.load_model(r"D:\dufy\code\ft_BOM\model\model_w" + str(w) + "_e" + str(i))
                # print("Model/model_w" + str(w) + "_e" + str(i))
                # 预测
                predict_labels = classifier.predict(texts)[0]
                print('测试集predict_labels 为：', predict_labels, type(predict_labels))
                print(confusion_matrix(correct_labels, predict_labels,
                                       labels=['__label__功率电感', '__label__圆柱体晶振', '__label__工字电感', '__label__晶振直插',
                                               '__label__晶振贴片', '__label__电位器-其他可调电阻', '__label__直插压敏电阻', '__label__直插晶体谐振器(无源)',
                                               '__label__直插独石电容', '__label__直插瓷片电容', '__label__直插电解电容', '__label__精密可调电阻',
                                               '__label__线绕电阻器-透孔', '__label__贴片低阻值采样电阻', '__label__贴片排阻','__label__贴片晶体振荡器(有源)',
                                               '__label__贴片晶体振荡器(无源)', '__label__贴片电容', '__label__贴片电感', '__label__贴片电解电容',
                                               '__label__贴片电阻', '__label__贴片线绕电感', '__label__贴片高精密-低温漂电阻', '__label__超级电容器',
                                               '__label__通孔电阻器', '__label__金属薄膜电阻器', '__label__钽质电容器-固体SMD钽电容器','__label__铝电解电容器-带引线',
                                               '__label__铝质电解电容器-SMD',  '__label__高频电感']))
                f1_score = metrics.f1_score(correct_labels, predict_labels,
                                            average='weighted')
                print('\033[1;32m 测试集F1: {:.3}\033[0m'.format(f1_score))
                test_f1.append(f1_score)
                # 计算预测结果
                # print(len(texts))
                accuracy_num = 0
                for j in range(len(texts)):
                    if predict_labels[j] == correct_labels[j]:
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
                print(line)
                correct_labels_train.append(line.strip().split(" , ")[0])
                texts1.append(line.strip().split(" , ")[1])
        print('correct_labels 为：{}'.format(correct_labels_train))
        # 加载分类模型
        for w in range(1, 2):
            for i in range(2, self.epoch):
                classifier = ff.load_model(r"D:\dufy\code\ft_BOM\model\model_w" + str(w) + "_e" + str(i))
                # print("Model/model_w" + str(w) + "_e" + str(i))
                # 预测
                predict_labels = classifier.predict(texts1)[0]
                # 计算预测结果
                # print(len(texts))
                accuracy_num = 0
                for j in range(len(texts1)):
                    if predict_labels[j] == correct_labels_train[j]:
                        accuracy_num += 1

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
        plt.title("1-gram")
        plt.grid()
        plt.savefig("./rate" + str(self.lr) + '_1-gram_' + self.loss + ".png")
        # plt.figure()
        plt.show()
        print('endddddd!!!!!!!!!!!!!!')

def standard_qq(str1):
    aa_description = standard(str1)
    print('前3预测： ', classfier.predict([aa_description], k=3))
    predict = classfier.predict([aa_description])
    print(predict)
    return predict, aa_description


def predict_line_label_out(file_path_combine):
    ss = pd.read_excel(file_path_combine)
    ss_count = ss.shape
    ss_han = ss_count[0]
    ss_lie = ss_count[1]
    true_false_list = []
    # row_selection = row_return_list(file_path_combine)   # 添加列的选择
    row_selection = list(range(1, ss_lie))   # 添加列的选择
    if row_selection != []:
        # 先遍历行再遍历列
        for h in range(ss_han):
            # 拼接一行所有内容
            zonhe = ""
            yuanlai = ""
            shiti = ""
            true_label = ''
            # for l in range(1, ss_lie):  # 列全部选择
            for l in row_selection:
                if "nan" == str(ss.loc[h].ix[l]):
                    continue                               # replace('\n', '') 去掉excel中的换行符
                zonhe = zonhe + str(ss.loc[h].ix[l]).strip().replace('\n', '') + ' '
                true_label = ss.loc[h].ix[0]
                true_label = str(true_label).replace('/', '').strip()
                if true_label == '电池电池配件':
                    true_label = '电池配件'
                # if true_label == '功能模块开发板方案验证板':
                #     true_label = '方案验证板'
                # if true_label == '二级管':
                #     true_label = '二极管'
                # if true_label == '天线':
                #     true_label = '射频无线电'
                # if true_label == '仪器仪表及配件':
                #     true_label = '仪器仪表'
                # if true_label == '处理器和控制器':
                #     true_label = '处理器和微控制器'
                # if true_label == '光耦':
                #     true_label = '光电器件'
                # if true_label == '险丝座':
                #     true_label = '保险丝'
                # if true_label == '模拟开关':
                #     true_label = '模拟芯片'
                # if true_label == '逻辑器件':
                #     true_label = '逻辑芯片'

            if len(zonhe.strip()) == 0:
                continue
            print(true_label, '.....')
            print('excel 列选取结果：{}{}'.format(zonhe, type(zonhe)))

            try:
                aa, line_process = standard_qq(zonhe)
            except:
                continue

            shiti = aa[0][0].replace('__label__', '')
            if true_label != 'nan':
                if true_label == shiti:
                    true_false_list.append(1)
                else:
                    print('\033[1;31m error!!【{}】\033[0m预测为\033[1;31m 【{}】\033[0m--{}]'.format(true_label, shiti, file_path_combine))
                    error_infor = true_label+'     预测为     '+shiti
                    txt_write_line(r'D:\dufy\code\work_record\aaa.txt', error_infor)
                    txt_write_line(r'D:\dufy\code\work_record\bbb.txt', '__label__'+true_label+' , '+line_process)
                    true_false_list.append(0)
            print('\033[1;32m # {}\033[0m,excel原始输入：{}'.format(h, zonhe))
            print("预测实体为：\033[1;31m {}\033[0m".format(shiti))
            print('========================')

        return true_false_list
    else:
        return None


if __name__ == '__main__':
    # 1 读取excel写入不同的标签txt
    # 读取txt路径：ft_BOM\data\bom_subclass30'，  写入r'D:\dufy\code\ft_BOM\data\excel_write'
    # '''''''''''''''''jieba_used.py

    excel_read2txt()
    #
    # # 2 读取上一步不同txt 融合，写入'selection_data.txt'
    # # '''''''''''''''''data_selection.py
    #
    # merge_txts(2000)  ## 读取行数
    # # # # #
    # # # # # 3 划分数据集, 读取selection_data.txt'， 写入：'test_split_data.txt' 与 ‘train_split_data.txt'
    # # # # # '''''''''''''''''data_split.py
    # # # #
    # train_datas_split()

    # 4 训练-调参
    # 初始化

    # epoch_begin = 2
    # epoch_ = 150
    # loss_name = 'softmax'
    # learn_rate = 0.5
    #
    # ft_ = FastTextModel(epoch_, loss_name, learn_rate)
    # ft_.fit('train_split_data.txt')  # 训练
    # ft_.evaluate('train_split_data.txt', 'test_split_data.txt')   # 评价

    # ########## 5 测试
    #
    # # excel_path = r'D:\dufy\code\ft_BOM\data\bom_test'
    # # excel_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\mike2019-12-18\新建文件夹 (2)'
    # excel_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\三级标注1-7'
    # # excel_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\bom_test_random'
    #
    #
    # # txt_filePath = r'D:\dufy\code\ft_BOM\model'  # 读取文件夹路径,
    # txt_filePath = r'D:\dufy\code\ft_BOM\model_1'  # 单个模型测试
    # print(txt_filePath)
    # txt_names = os.listdir(txt_filePath)
    # # excel_test1(txt_names)
    # dict_model_test = {}
    # for i, name0 in enumerate(txt_names):  # 文件夹下文件循环
    #
    #     if str(txt_filePath).strip(r'D:\dufy\code\ft_BOM\\') == 'model':
    #         modle_path = r'D:\dufy\code\ft_BOM\model' + '\\' + name0
    #
    #     else:
    #         modle_path = r'D:\dufy\code\ft_BOM\model_1' + '\\' + name0
    #         pass
    #
    #     # modle_path = 'model_' + '\\' + name0
    #     # print(modle_path)
    #     classfier = ff.load_model(modle_path)
    #
    #     f_train = open(r'D:\dufy\code\work_record\aaa.txt', 'w')
    #     f_train.truncate()
    #     f_train.close()
    #     f_test = open(r'D:\dufy\code\work_record\bbb.txt', 'w')
    #     f_test.truncate()
    #     f_test.close()
    #     all_record = 0
    #     right_record = 0
    #     # folder_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\mike2019-12-6'
    #     # folder_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\已标注bom1206\已标注\1206'
    #     # folder_path = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\bom_test_random'
    #     folder_path = excel_path
    #     file_names = os.listdir(folder_path)
    #     for i, name1 in enumerate(file_names):
    #         file_path_combine = folder_path + '//' + name1
    #         # ss = pd.read_excel(file_path_combine)
    #         # r"C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\mike2019-12-6\2c93ea3b6dfd8eaa016e0fd1bc90012f-U226.xlsx")
    #         # predict_line_all(ss)  #预测 excel 一行所有数据
    #         print(file_path_combine)
    #         TF_record = predict_line_label_out(file_path_combine)  # 预测 excel 一行,除去标签
    #         print(TF_record)
    #         # if TF_record != None and []:  # 之前这样写，不对！！！！
    #         if TF_record:
    #             print(name1)
    #             print(TF_record, '~~~~~~~~~~~~~~~~')
    #             print('正确率:{:.2f}'.format(sum(TF_record) / len(TF_record)))
    #             all_record += len(TF_record)
    #             for i in TF_record:
    #                 if i == 1:
    #                     right_record += 1
    #         else:
    #             print('{} 无法识别'.format(file_path_combine))
    #         print(file_path_combine)
    #         print('\033[1;32m =\033[0m'*120)
    #         pass
    #     print('标注数据量:{}'.format(all_record))
    #     print('预测正确量:{}'.format(right_record))
    #     print('测试集全部数据正确率:{:.2f}'.format(right_record / all_record))
    #     print('全部结束！！！！')
    #     dict_model_test[name0] = right_record / all_record #此处。。。。
    # print(dict_model_test)
    #
    # x = []
    # y = []
    # for key, value in dict_model_test.items():
    #     print(key.strip('model_w1_e'), value)
    #     x.append(int(key.strip('model_w1_e')))  # append() 方法用于在列表末尾添加新的对象。
    #     y.append(value)
    # print(x, y)
    # plt.plot(x, y, "b-o", linewidth=2)
    # plt.xlabel("epoch")  # X轴标签
    # plt.ylabel("accu")  # Y轴标签
    # plt.title("Line plot")  # 图标题
    # plt.show()  # 显示图
    #




