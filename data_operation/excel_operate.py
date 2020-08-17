# -*- coding: utf-8 -*-
"""
Created by Dufy on 2020/2/26  16:05
IDE used: PyCharm 
Description :
1)
2)  
Remark:      
"""
import os

import pandas as pd
from data_operation.txt_operate import OperateTXT



class OperateExcel:   # 针对单个文件
    def __init__(self, url):
        self.file_path = url   # 文件路径,
        self.data_read = ''    # 后续使用
        # print('OperateExcel init.....')

    def excel_matrix(self):  # 显示excel 行、列
        pass
        # print('===========当前excel基本情况：===========')
        try:
            ss = pd.read_excel(self.file_path)  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
        except PermissionError:  # 针对 ~ 已打开文件
            return 0, 0
        # print(ss.index,'---------')  # 获取行的索引名称
        # print('各列含义:{}'.format(ss.columns))  # 获取列的索引名称
        # print(ss)
        # 获取总和、得出行数和列数
        ss_count = ss.shape
        line = ss_count[0]
        row = ss_count[1]
        # print('ss_count = {}, 数据记录条数：行={}, 列={}'.format(ss_count, line, row))
        return line, row

    def excel_content_all(self, mark=' '):
        ss = pd.read_excel(self.file_path)  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
        # 获取总和、得出行数和列数
        ss_count = ss.shape
        line = ss_count[0]
        row = ss_count[1]

        aa = ''
        for j_line in range(line):  # j为行
            #     aa += '@' +'\n'
            for i in range(row):
                # print('excel矩阵单元格：',ss.loc[j_line].ix[i], end='')
                # print(ss.loc[j_line].ix[i], end='')
                # aa += str(ss.loc[j_line].iloc[i]) + '@'
                aa += str(ss.loc[j_line].iloc[i]).replace('\n', '').replace('\r', '') + mark  # 同时去除换行符
            aa += '\n'

        # for line in aa.splitlines():  # 对字符串按行读取
        #     print(line)
        # self.data_read = aa   # 生成读取数据，为后续操作准备
        # print(aa)
        return aa

    def excel_write_in(self, target_path):
        try:
            for line_temp in self.excel_content_all().splitlines():
                line_temp = line_temp.replace('nan', '')
                OperateTXT().txt_write_line(target_path, line_temp)
        except IOError as ex:
            print(ex)
            print('写文件时发生错误!!!!!!')  # \033[1;31m 字体颜色：红色\033[0m

        print('操作完成!')




if __name__ == "__main__":
    pass
    aa= OperateExcel(r'C:\Users\Administrator\Desktop\连接器数据0731.xlsx')
    print(aa.excel_content_all())
    aa.excel_write_in(r'C:\Users\Administrator\Desktop\连接器数据0731.txt')
    # df = pd.read_excel(r'C:\Users\Administrator\Desktop\连接器数据0731.xlsx',
    #                    )
    # df.to_csv(r'C:\Users\Administrator\Desktop\连接器数据0731.txt')