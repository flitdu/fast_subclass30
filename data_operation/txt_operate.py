# -*- coding: utf-8 -*-
"""
Created by Dufy on 2020/2/26  16:21
IDE used: PyCharm 
Description :
1)
2)  
Remark:      
"""
import random

class OperateTXT:   # 针对单个文件
    def __init__(self, url=None):
        self.file_path = url   # 文件路径,
        # print('OperateTXT init.....')

    def txt_write_line(self, save_path, line): #将某一行写入txt
        filenames = save_path
        # print("wenjianm:" + filenames)
        # fs_list = []
        with open(filenames, mode='a+', encoding='utf-8') as f:
            f.write(line + '\n')

        # try:
        #     # fs_list.append(open(filenames, 'a', encoding='utf-8'))  # a 指定打开文件的模式，a为追加   r为只读
        #     with open(filenames, mode='a+', encoding='utf-8') as f:
        #         f.write(line + '\n')
        #         # f.write(line)
        # except IOError as ex:
        #     print(ex)
        #     print('\033[1;31m OperateTXT.txt_write_line()写文件时发生错误!!!!\033[0m')
        #     print(self.file_path)





