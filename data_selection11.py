# -*- coding: utf-8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import os
import random
import linecache
import matplotlib.pyplot as plt
from data_operation.txt_operate import OperateTXT


def merge_txt_files(lines_number=0):
    f_1 = open('selection_data.txt', 'w')
    f_1.truncate()
    f_1.close()


    merge_path = r'D:\dufy\code\fast_subclass30\data\excel_write'  # 读取文件夹路径!!!!!!!!!!!!
    file_names = os.listdir(merge_path)

    for i, name0 in enumerate(file_names):  # 文件夹下文件循环
        path = merge_path + '\\' + name0
        print('读取txt： ', i+1, path)

    file_names = tuple(os.listdir(merge_path))  # 转为tuple
    fs_list = []
    label_number = {}
    for filename in file_names:
        fs_list.append(open(filename, 'w', encoding='utf-8'))
    # for i in range(len(file_names)):  # 遍历各txt
    for i, name0 in enumerate(file_names):  # 遍历各txt
        txt_path = merge_path + '\\' + name0
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
                # print(line)
                # if linecache.getline(txt_path, i):
                if i in test_slice:  # 如果在随机取的值里
                    # print('test')
                    OperateTXT.txt_write_line('selection_data.txt', line)
        else:  # 直接全部写进去
            num.append(n)
            for i in num:
                line = label_name0 + linecache.getline(txt_path, i)  # 待写入文件行
                OperateTXT.txt_write_line('selection_data.txt', line)
    print(label_number)
    for i in label_number:
        #     print(i, a[i])
        plt.plot(i, label_number[i], 'r:o')
    plt.grid()
    plt.xticks(rotation=270)  # 标签旋转
    plt.show()
    print('txt_get_somelines: done!!!!')



if __name__ == '__main__':
    merge_txt_files()