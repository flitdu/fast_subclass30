# -*- coding: utf-8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import os
import random
import linecache
import matplotlib.pyplot as plt
from data_operation.txt_operate import OperateTXT
from data_operation.constant import label_subclass_database
from data_operation.function import get_logger
from data_operation.function import path_clear, file_clear
import pandas as pd

logger = get_logger()


def shuffle(origin_txt, shuffle_txt):
    out = open(shuffle_txt, 'w', encoding='utf-8')
    lines = []
    with open(origin_txt, 'r', encoding='utf-8') as infile:
        for line in infile:
            lines.append(line)
    random.shuffle(lines)
    for line in lines:
        out.write(line)
    infile.close()
    out.close()


def mergeLabelTxt(limit_number, shuffle_tag):
    """
        将不同txt 文件融入到一起
        :param limit_number: 极限数值，超过则选取读到txt 里
        :param shuffle_tag:
        :return:
        """
    # f_1 = open(r'.\data\selection_data.txt', 'w')
    # f_1.truncate()
    # f_1.close()
    file_clear(r'.\data\selection_data.txt')
    path = r'D:\dufy\code\local\corpus\bom_subclass\subclass_txt'

    file_names = tuple(os.listdir(path))  # 转为tuple

    label_number = {}

    listfile = os.listdir(path)
    df_list = []
    number_limit = limit_number  # 超过3 个则随机选择写入
    for k, i in enumerate(listfile):

        file_name_i = path + '/' + i
        print('进度：', k / len(listfile), '读取', file_name_i)

        df_i = pd.read_csv(file_name_i, sep='/t', skip_blank_lines=False, names=['参数'], engine='python',
                           encoding='utf8')

        name_ = i.replace('.txt', '')
        label_name0 = '__label__' + name_ + ' , '

        df_i['类目'] = label_name0
        df_i = df_i[['类目', '参数']]  # 交换显示的列名字顺序
        print('数据条数：', df_i.shape[0])
        label_number[i.replace('.txt', '')] = df_i.shape[0]

        if df_i.shape[0] < number_limit:
            df_list.append(df_i)
        else:
            pass  # 随机从中选择number_limit 条
            data_sample = df_i.sample(n=number_limit, random_state=None, replace=False)  # 随机选取
            df_list.append(data_sample)

    frames = df_list
    result = pd.concat(frames)
    result.to_csv(r'./data/selection_data.txt', encoding='utf-8', sep='\t', header=None, index=None)

    print('文件合并完成')
    # ===============================

    print(label_number)
    list1 = []
    for key, value in label_number.items():
        #     print(key, value)
        list1.append('__label__' + key)
        # list1.append(key)
    # print('标签列表 \n：{}'.format(list1))
    print('标签列表 \n：{}'.format([i.replace('__label__','') for i in list1]))

    for i in label_number:
        plt.plot(i, label_number[i], 'r:o')
    plt.xticks(size=6)
    plt.grid()
    plt.xticks(rotation=270)  # 标签旋转
    plt.show()
    print('txt_get_somelines: done!!!!')

    if shuffle_tag == 1:
        shuffle(r'.\data\selection_data.txt', r'.\data\selection_data_shuffle.txt')
        # shuffle(r'.\data\selection_data_shuffle.txt', r'.\data\selection_data_shuffle1.txt')
        # shuffle(r'.\data\selection_data_shuffle1.txt', r'.\data\selection_data_shuffle2.txt')

    return list1


if __name__ == '__main__':
    pass
    # merge_txt_files(12, 1)