# -*- coding: utf-8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import os
import random
import linecache
import matplotlib.pyplot as plt
from data_operation.txt_operate import OperateTXT
from data_operation.constant import label_name_refer
from data_operation.function import get_logger
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


def merge_txt_files(lines_number, shuffle_tag):
    f_1 = open(r'.\data\selection_data.txt', 'w')
    f_1.truncate()
    f_1.close()
    PATH = r'D:\dufy\code\fast_subclass30\data\excel_write'

    file_names = tuple(os.listdir(PATH))  # 转为tuple

    label_number = {}
    # for i in range(len(file_names)):  # 遍历各txt
    for i, name0 in enumerate(file_names):  # 遍历各txt
        txt_path = PATH + '\\' + name0
        print('读取', txt_path)
        txt = open(txt_path, 'rb')

        data = txt.read().decode('utf-8')  # python3一定要加上这句不然会编码报错！
        txt.close()
        n = data.count('\n')
        label_number[name0.replace('.txt', '')] = n
        # n += 1
        print("总行数:", n)
        num = list(range(1, n))
        # test_size = round(size * n)
        name_ = name0.replace('.txt', '')
        label_name0 = '__label__' + name_ + ' , '
        if lines_number < n:
            test_slice = random.sample(num, lines_number)  # 从list中随机获取个元素，作为一个片断返回
            # print('测试取值：{}'.format(test_slice))
            for i in num:
                # print(linecache.getline(txt_path, i))
                # if linecache.getline(txt_path, i):
                line = label_name0 + linecache.getline(txt_path, i)  # 待写入文件行
                # print(line.strip('\n'))
                # if linecache.getline(txt_path, i):
                if i in test_slice:  # 如果在随机取的值里
                    # print('test')
                    OperateTXT().txt_write_line(r'.\data\selection_data.txt', line.strip('\n'))
        else:  # 直接全部写进去
            num.append(n)
            for i in num:
                line = label_name0 + linecache.getline(txt_path, i)  # 待写入文件行
                OperateTXT().txt_write_line(r'.\data\selection_data.txt', line.strip('\n'))
    print(label_number)
    list1 = []
    for key, value in label_number.items():
        #     print(key, value)
        list1.append('__label__' + key)
        # list1.append(key)
    # print('标签列表 \n：{}'.format(list1))
    print('标签列表 \n：{}'.format([i.replace('__label__','') for i in list1]))

    list_temp = [i.replace('__label__','') for i in list1]
    # for i in list_temp:
    #     if i not in label_name_refer:
    #         logger.critical('产生新的标签：{}'.format(i))


    for i in label_number:
        #     print(i, a[i])
        plt.plot(i, label_number[i], 'r:o')
    plt.grid()
    plt.xticks(rotation=270)  # 标签旋转
    plt.show()
    print('txt_get_somelines: done!!!!')

    if shuffle_tag == 1:
        shuffle(r'.\data\selection_data.txt', r'.\data\selection_data_shuffle.txt')
    
    return list1
if __name__ == '__main__':
    merge_txt_files(12, 1)