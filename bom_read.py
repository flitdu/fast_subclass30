# -*- coding: utf-8 -*-
"""
Created by Dufy on 2020/2/27  10:27
IDE used: PyCharm 
Description :
1) 替换之前的 jieba_used_1209.py
2)  target_path = 'data\excel_write'
3) 原先的 excel2different_files()，现在重写OperateExcelSubclass.excel_write_in

Remark:  注意target_path = 'data\excel_write'要与   txt_filePath = r'D:\dufy\code\ft_BOM\data\excel_write'  # 读取文件夹路径,
         保持一致
"""
from data_operation import OperateExcel, function
from data_operation.function import load_stop_word_list, label_new, standard
# from data_operation import OperateTXT
from data_operation.txt_operate import OperateTXT
from data_operation.constant import label_name_forbid
import os
import jieba
jieba.load_userdict('dict_boom.txt')
stop_words = load_stop_word_list("stopwords_subclass.txt")


class OperateExcelSubclass(OperateExcel):  # 重写函数
    def excel_write_in(self, target_path):
        pass
        try:
            # print(target_path, '~~~~~~~')
            # fs_list.append(open(filenames, 'w', encoding='utf-8'))
            for line_read in self.excel_content_all().splitlines():
                target_path_temp = target_path   # 由于此处要循环，所以设置临时变量代替

                aa_label = line_read.split()[0].replace('/', '')  # 替换标签里面 '/'
                # aa_label = aa.split()[0] # 替换标签里面 '/'
                if aa_label in label_name_forbid:
                    continue
                aa_label = label_new(aa_label)

                if aa_label != 'nan':
                    # print(aa_label, '~~~~~~~')
                    aa_description = " ".join(line_read.split()[1:])
                    aa_description = standard(aa_description, stop_words)  # 标准化处理

                    print('最终写入行为：{}'.format(aa_description))
                    aa_description_length = 0
                    for i in aa_description.split(' '):
                        if i != '':
                            aa_description_length += 1
                    # print(length)

                    target_path_temp = target_path_temp + '\\' + aa_label + '.txt'
                    # print(target_path, '-', aa_label, '!!!!')
                    if aa_description_length > 1:  # 选取训练数据的长度，大于3才算
                        OperateTXT().txt_write_line(target_path_temp, aa_description)

        except IOError as ex:
            print(ex)
            print('bom_read.py,写文件时发生错误!!!!!!')  # \033[1;31m 字体颜色：红色\033[0m

        print('操作完成!')



def excel_read2txt():
    # 先清空：
    txt_filePath = r'D:\dufy\code\fast_subclass30\data\excel_write'  # 读取文件夹路径,
    function.files_clear(txt_filePath)

    # filePath = r'C:\Users\Administrator\Documents\Tencent Files\3007490756\FileRecv\bom_test_random'  # 读取文件夹路径
    filePath = r'D:\dufy\code\ft_BOM\data\bom_subclass30'  # 读取文件夹路径!!!!!!!!!!!!
    file_names = os.listdir(filePath)

    for i, name0 in enumerate(file_names):  # 文件夹下文件循环
        print('==========================')
        path = filePath + '\\' + name0
        print('path为： ', path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        aa = OperateExcelSubclass(path)
        # aa.excel_data2temp_files()  # 生成temp @文件，为后续处理做准备
        aa.excel_write_in(r'data\excel_write')  # 读取当前excel覆盖写入
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('path为： ', path)
        print('==========================')

if __name__ == "__main__":
    pass
