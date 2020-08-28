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
from data_operation import OperateExcel
from data_operation.function import load_stop_word_list, labelNewSubclass, standard, file_clear
# from data_operation import OperateTXT
from data_operation.txt_operate import OperateTXT
from data_operation.constant import label_name_forbid, label_subclass_database
import os
import pandas as pd
import jieba                          # 组合使用】
from data_operation.function import get_logger
from collections import Counter

jieba.load_userdict('dict_boom.txt')  # 组合使用】
stop_words = load_stop_word_list("stopwords_subclass.txt")
logger = get_logger()


class OperateExcelSubclass(OperateExcel):  # 重写函数

    @staticmethod
    def removeDuplicates(string):
        """
        去掉字符串中重复的词，只保留一个
        :param string:
        :return:去重后的句子
        """
        pass
        a = string.split()
        dic = Counter(a)
        for i, v in dic.items():
            while v > 1:
                a.remove(i)
                v -= 1
        return ' '.join(a)

    def excel_write_in(self, target_path):
        pass
        try:
            corpus_check = []
            for line_read in self.excel_content_all().splitlines():
                target_path_temp = target_path   # 由于此处要循环，所以设置临时变量代替

                aa_label = line_read.split()[0].replace('/', '')  # 替换标签里面 '/'
                # aa_label = aa.split()[0] # 替换标签里面 '/'
                if aa_label in label_name_forbid:
                    continue

                aa_label = labelNewSubclass(aa_label)  # label名字规范化

                if aa_label != 'nan':
                    split_symbol = ['_',
                                    '-',
                                    ',',
                                    '/',
                                    '（',
                                    '）',
                                    '(',
                                    ')',
                                    '"',
                                    '，',
                                    '\\',
                                    ':',
                                    '：',
                                    '@',
                                    '【',
                                    '】',
                                    ';',
                                    '；']
                    split_symbol = []
                    # print(aa_label, '~~~~~~~')
                    aa_description = " ".join(line_read.split()[1:])
                    aa_description0 = aa_description

                    logger.debug('标签：{}， 初始输入：{}'.format(aa_label, aa_description0))
                    aa_description = standard(aa_description, stop_words, split_symbol)  # 标准化处理

                    # 贴片电感过滤，有关键字'fixed'的重新纠正
                    current_bom_name = self.file_path.split('\\')[-1].split('.')[0]  # 当前BOM名称
                    if aa_label=='贴片电感' and 'fixed' in aa_description and current_bom_name!='贴片电感fixed纠正':
                        continue
                    elif aa_label=='贴片电感' and 'power' in aa_description:
                        continue

                    # aa_description = self.removeDuplicates(aa_description)  # 去掉重复词后的句子

                    logger.debug('最终写入行为：{}'.format(aa_description))

                    target_path_temp = target_path_temp + '\\' + aa_label + '.txt'
                    # print(target_path, '-', aa_label, '!!!!')
                    if len(aa_description.split()) > 3:  # 选取训练数据的长度，大于3才算
                        if aa_label not in label_subclass_database:
                            logger.critical('路径"{},产生错误标签：{}'.format(self.file_path, aa_label))
                        OperateTXT().txt_write_line(target_path_temp, aa_description)

                        if aa_label in ['排针排母', '线对板线对线连接器','贴片电感']:  # 取回待检查语料集，排故
                            corpus_check_dict = {}
                            corpus_check_dict['参数'] = aa_description0
                            corpus_check_dict['类别'] = aa_label
                            corpus_check_dict['文件名'] = self.file_path.split('\\')[-1]
                            corpus_check.append(corpus_check_dict)

            return corpus_check  # 返回内容，用于语料检查


        except IOError as ex:
            print(ex)
            print('bom_read.py,写文件时发生错误!!!!!!')  # \033[1;31m 字体颜色：红色\033[0m
            return None
        # print('操作完成!')


def excel_read2txt():
    # # 先清空：
    txt_file_path = r'D:\dufy\code\local\corpus\bom_subclass\subclass_txt'  # 读取文件夹路径,

    # file_clear(txt_filePath)
    txt_names = os.listdir(txt_file_path)
    for i, name0 in enumerate(txt_names):  # 文件夹下文件循环
        path = txt_file_path + '\\' + name0
        os.remove(path)

    # bom_path = r'D:\dufy\code\ft_BOM\data\bom_subclass30'  # 读取文件夹路径!!!!!!!!!!!!
    bom_path = r'D:\dufy\code\local\corpus\bom_subclass\subclass_excel'  # 读取文件夹路径!!!!!!!!!!!!
    file_names = os.listdir(bom_path)

    data1 = pd.DataFrame()  # 排针排母
    data2 = pd.DataFrame()
    data3 = pd.DataFrame()
    j = 0
    for i, name0 in enumerate(file_names):  # 文件夹下文件循环
        j += 1
        if '~$' in name0:
            continue
        print('\033[1;36m  文件读取进度：\033[0m {:.3f}'.format(j / len(file_names)))
        logger.debug('==========================')
        path = bom_path + '\\' + name0
        logger.debug('path为：{} '.format(path))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        aa = OperateExcelSubclass(path)
        # aa.excel_data2temp_files()  # 生成temp @文件，为后续处理做准备
        corpus_check = aa.excel_write_in(txt_file_path)  # 读取当前excel覆盖写入
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        logger.debug('path为： '.format(path))
        logger.debug('==========================')

        if bool(corpus_check):  # 待检查语料合集
            pass
            for i in corpus_check:
                # print(i)
                if i['类别'] =='排针排母':
                    data1 = data1.append(i, ignore_index=True)
                elif i['类别'] =='线对板线对线连接器':
                    data2 = data2.append(i, ignore_index=True)
                elif i['类别'] =='贴片电感':
                    data3 = data3.append(i, ignore_index=True)
    data1.to_excel(r'./data/check/排针排母.xls')
    data2.to_excel(r'./data/check/线对板线对线连接器.xls')
    data3.to_excel(r'./data/check/贴片电感.xls')


if __name__ == "__main__":
    pass

    path = r'C:\Users\Administrator\Desktop\2c93ea3b6e217bf2016e257babb70044-U954.xls'
    # path = r'C:\Users\Administrator\Desktop\2c93ea3b6dd7ced7016dd86e98da0069-U558.xls'

    aa = OperateExcelSubclass(path)
    aa.excel_write_in(r'C:\Users\Administrator\Desktop\test')  # 读取当前excel覆盖写入


