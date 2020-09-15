# -*- coding: utf-8 -*-

"""
-------------------------------------------------
Created by Dufy on 2019/11/25
IDE used: PyCharm Community Edition
Description :
1)定义经常用到的函数
2)
-------------------------------------------------
Change Activity:

-------------------------------------------------
"""

# import fastText.FastText as ff
import fasttext as ff
import jieba
# jieba.load_userdict('dict_boom.txt')
import os
# classfier = ff.load_model("Model/model_w1_e23")

import logging.config
import yaml

# # 日志文件配置
def get_logger():
    log_conf = 'logging.config.yaml'
    with open(log_conf, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    return logging.getLogger()

def test():
    print('success!')


def path_clear(txt_filePath):   # 文件夹路径下全部清空
    pass
    # txt_filePath = r'D:\dufy\code\fast_subclass30\data\excel_write'  # 读取文件夹路径,
    txt_names = os.listdir(txt_filePath)
    for i, name0 in enumerate(txt_names):  # 文件夹下文件循环
        path = txt_filePath + '\\' + name0
        f_1 = open(path, 'w')
        f_1.truncate()
        f_1.close()


def file_clear(file):   # 单个文件清空
    f_1 = open(file, 'w')
    f_1.truncate()
    f_1.close()


def deleteFile(file_path):
    os.remove(file_path)


def load_stop_word_list(file_path):
    """
    加载停用词表
    :param file_path: 停用词表路径
    :return:
    """
    stop_words = set()
    with open(file_path, "r", encoding="utf-8") as f_stopwords:
        for line in f_stopwords:
            stop_words.add(line.strip())
    return stop_words


def standardize_text_similarity(content):
    standard_after = content.lower().replace('nan', '')
    return standard_after


def standard(str1, stop_words=None, split_symbol=None):  # 标准化处理

    aa_description = str1.replace('nan', '')
    print('--'*20)
    print('\033[1;32m  原始输入：\033[0m {}'.format(aa_description))
    if bool(split_symbol):  # 分隔符和 jieba 选一种
        pass
        for i in split_symbol:
            b = str(aa_description).lower().replace(i, ' ')
            aa_description = b
        print(f'分隔符：{aa_description}')
    else:
        aa_description = ' '.join(jieba.cut(str(aa_description).lower()))
        print('\033[1;32m # jieba分词：\033[0m {}'.format(aa_description))

    if stop_words:
        for word in aa_description.split():  # 停用词使用
            if str(word) in stop_words:
                aa_description = aa_description.replace(word, '')

    # python 合并多个空格为1个,以下效果不一样
    aa_description = ' '.join(filter(lambda x: x, aa_description.split(' ')))
    # aa_description = ' '.join(aa_description.split(' '))

    print('\033[1;32m # 停用词标准化：\033[0m{}'.format(aa_description))
    return aa_description


def fast_vec_standard(str1, stop_words=None):  #标准化处理
    # print('原始输入： ', str1.lower())

    # 不分词：
    # str1 = ' '.join(jieba.cut(str(str1).lower().replace('-','').replace('#', '').replace('=', '').replace('+', '')))  # 先jieba 分词
    str1 = str(str1).lower().replace('-','').replace('#', '').replace('=', '').replace('+', '').replace('(', '').replace(')', '') # 先jieba 分词
    # str1 = str(str1).lower().replace('-','').replace('#', '').replace('=', '').replace('+', '') # 先jieba 分词
    aa_description = str1.replace('nan', '').replace('/', ' ').replace('\\', ' ')\
        .replace('（', ' ').replace('）', ' ').replace('_', ' ').replace(',',' ').replace('，',' ')\
        .replace('[',' ').replace(']',' ').replace(':',' ').replace('：',' ').replace(';',' ').replace('；',' ')
    # print('\033[1;32m # 初步处理：\033[0m{}'.format(aa_description))
    # for word in aa_description.split():  # 停用词使用
    if stop_words:
        for word in stop_words:  # 停用词使用
            # print(word)
            # print(stop_words)
            if str(word) in aa_description:
                aa_description = aa_description.replace(word, '')

    # python 合并多个空格为1个,以下效果不一样
    aa_description = ' '.join(filter(lambda x: x, aa_description.split(' ')))
    # aa_description = ' '.join(aa_description.split(' '))

    # print('\033[1;32m # 停用词标准化：\033[0m{}'.format(aa_description))
    return aa_description


def hanlp_standard(str1, stop_words):  #标准化处理
    # print('原始输入： ', str1.lower())

    # 不分词：
    str1 = ' '.join(jieba.cut(str(str1).lower().replace('-','').replace('#', '').replace('=', '').replace('+', '')))  # 先jieba 分词
    # str1 = str(str1).lower().replace('-','').replace('#', '').replace('=', '').replace('+', '') # 先jieba 分词
    aa_description = str1.replace('nan', '').replace('/', ' ').replace('(', ' ')\
        .replace(')', ' ').replace('（', ' ').replace('）', ' ').replace('_', ' ').replace(',',' ').replace('，',' ')\
        .replace('[',' ').replace(']',' ').replace(':',' ').replace('：',' ').replace(';',' ').replace('；',' ')
    # print('\033[1;32m # 初步处理：\033[0m{}'.format(aa_description))
    for word in aa_description.split():  # 停用词使用
        # print(word)
        if str(word) in stop_words:
            aa_description = aa_description.replace(word, '')

    # python 合并多个空格为1个,以下效果不一样
    aa_description = ' '.join(filter(lambda x: x, aa_description.split(' ')))
    # aa_description = ' '.join(aa_description.split(' '))

    # print('\033[1;32m # 停用词标准化：\033[0m{}'.format(aa_description))
    return aa_description


def label_new_entity(label_origin):
    true_label = label_origin

    if true_label == '电池电池配件':
        true_label = '电池配件'
    elif true_label == '功能模块开发板方案验证板':
        true_label = '方案验证板'
    elif true_label == '二级管':
        true_label = '二极管'
    elif true_label == '天线':
        true_label = '射频无线电'
    elif true_label == '仪器仪表及配件':
        true_label = '仪器仪表'
    elif true_label == '处理器和控制器':
        true_label = '处理器和微控制器'
    elif true_label == '光耦':
        true_label = '光电器件'
    elif true_label == '险丝座':
        true_label = '保险丝'
    elif true_label == '模拟开关':
        true_label = '模拟芯片'
    elif true_label == '逻辑器件':
        true_label = '逻辑芯片'

    return true_label

def labelNewSubclass(label_origin):
    aa_label = label_origin

    if aa_label == '贴片电容;':
        aa_label = '贴片电容'
    elif aa_label == '铝质电解电容器-SMD：' or aa_label=='铝质电解电容器-SMD' or aa_label == '铝有机聚合物电容器':
        aa_label = '贴片电解电容'
    elif aa_label == 'MLCC-SMDSMT':
        aa_label = '贴片电容'
    elif aa_label == '厚膜电阻器' or aa_label == '薄膜电阻器' or aa_label == '芯片电阻-表面安装' or aa_label == '电阻贴片' or aa_label=='贴片超低阻值电阻':
        aa_label = '贴片电阻'
    elif aa_label == '铝电解电容器-带引线' or aa_label == '直插电解电容:' or aa_label=='铝电铝电解电容器-带引线解电容器-带引线' or aa_label=='铝质电解电容器-螺旋式接线端':
        aa_label = '直插电解电容'
    elif aa_label == '直插瓷片电容:':
        aa_label = '直插瓷片电容'
    elif aa_label == '安规X电容:' or aa_label == '安规X电容' or aa_label == '安规电容直插' or aa_label == '安规Y电容':
        aa_label = '安规电容'
    elif aa_label == '超级电容' or aa_label == '固态电解电容':
        aa_label = '超级电容器'
    elif aa_label == '碳膜电阻器' or aa_label == '碳质电阻器':
        aa_label = '碳膜电阻'
    elif aa_label == '电阻器网络与阵列' or aa_label == '贴片排阻':
        aa_label = '排阻'
    elif aa_label == '电位器-其他可调电阻' or aa_label =='电位计':
        aa_label = '可调电阻电位器'
    elif aa_label == '贴片低阻值采样电阻' or aa_label == '电流传感电阻器' or aa_label == '直插低阻值采样电阻' or aa_label=='采样电阻:':
        aa_label = '采样电阻'
    elif aa_label == '直插压敏电阻' or aa_label == '贴片压敏电阻':
        aa_label = '压敏电阻'
    elif aa_label == 'MELF电阻':
        aa_label = 'MELF晶圆电阻'
    elif aa_label == 'NTC':
        aa_label = 'NTC热敏电阻'
    elif aa_label == 'PTC':
        aa_label = 'PTC热敏电阻'
    elif aa_label == '金属氧化物电阻器':
        aa_label = '金属氧化膜电阻'
    elif aa_label == '碳质电阻器' or aa_label == '碳膜电阻器':
        aa_label = '碳膜电阻'
    elif aa_label == '高功率贴片电阻' or aa_label == '铝壳电阻' or aa_label == '直插功率电阻' or aa_label == '贴片功率电阻' or aa_label == 'TO封装平面功率电阻':
        aa_label = '铝壳大功率电阻'
    elif aa_label == '高频/射频电阻':
        aa_label = '射频高频电阻'
    elif aa_label == '电位器-其他可调电阻' or aa_label == '变阻器' or aa_label == '电位计' or aa_label == '电位计工具及硬件' or aa_label == '可调功率电阻' or aa_label == '精度电位计' or aa_label == '微调电阻器SMD' or aa_label == '微调电阻器通孔' or aa_label=='精密可调电阻':
        aa_label = '可调电阻电位器'
    elif aa_label == '线绕电阻' or aa_label == '线绕电阻器-透孔' or aa_label == '线绕电阻器':
        aa_label = '绕线电阻'
    elif aa_label == '金属玻璃釉电阻' or aa_label == '通孔电阻器':
        aa_label = '直插通孔电阻'
    elif aa_label == '金属薄膜电阻器':
        aa_label = '金属膜电阻'
    elif aa_label == '电阻套件' or aa_label == '电阻硬件':
        aa_label = '电阻套件及附件'
    elif aa_label == '贴片精密电阻' or aa_label=='贴片高精密、低温漂电阻' or aa_label == '贴片高精密-低温漂电阻':
        aa_label = '贴片电阻'
    elif aa_label == '高压陶瓷电容' or aa_label == '高压瓷片电容' or aa_label == '瓷片电容器':
        aa_label = '直插瓷片电容'
    elif aa_label == '钽质电容器-固体铅钽电容器' or aa_label =='液体钽电容器' or aa_label == '钽质电容器-固体SMD钽电容器' or aa_label == '钽质电容器-SMD聚合物液体钽电容器' or aa_label=='钽质电容器-固体SMD钽电容器:':
        aa_label = '钽电容'
    elif aa_label == '云母电容器':
        aa_label = '云母电容'
    elif aa_label == '微调电容器与可变电容器':
        aa_label = '可调电容'
    elif aa_label == '薄膜电容器' or aa_label =='CL21电容' or aa_label == '聚酯薄膜电容' or aa_label == 'CBB电容':
        aa_label = '薄膜电容'
    elif aa_label == '电容套件' or aa_label == '电容硬件':
        aa_label = '电容套件及附件'
    elif aa_label == 'MLCC-含引线':
        aa_label = '直插瓷片电容'
    elif aa_label == '贴片绕线电感' or aa_label=='贴片线绕电感':
        aa_label = '贴片电感'
    elif aa_label == '固定值电感':
        aa_label = '固定电感'
    elif aa_label == '电感套件' or aa_label == '电感套件及配件' or aa_label == '可变电感' or aa_label == '可调电感':
        aa_label = '可变电感器套件配件'
    elif aa_label =='贴片晶体谐振器(有源)' or aa_label == '直插晶体振荡器(有源)' or aa_label == '贴片晶体振荡器(有源)' or aa_label == '压控振荡器（VCO）' or aa_label == '压控式晶体振荡器(VCXO)' or aa_label == '温度补偿晶体振荡器(TCXO)' or aa_label == '恒温晶体振荡器' or aa_label == '压控振荡器' or aa_label == '温度补偿压控晶体振荡器':
        aa_label = '有源晶体振荡器'
    elif aa_label == '无源晶体振荡器:' or aa_label == '贴片晶体振荡器(无源)' or aa_label =='直插晶体谐振器(无源)' or aa_label == '贴片晶体谐振器(无源)' or aa_label=='贴片晶体振荡器(无源):' or aa_label=='贴片晶体谐振器(无源):' or aa_label=='圆柱体晶振':
        aa_label = '无源晶体振荡器'
    elif aa_label == '陶瓷谐振器' or aa_label == '压控SAW振荡器' or aa_label == '声表谐振器':
        aa_label = '谐振器'
    elif aa_label == '电容器网络，阵列':
        aa_label = '电容器阵列与网络'
    elif aa_label == 'TVS二极管（瞬态电压抑制二极管）':
        aa_label = 'TVS二极管(瞬态电压抑制二极管)'
    elif aa_label == '结型场效应晶体管（JFET）':
        aa_label = '结型场效应晶体管(JFET)'
    elif aa_label == 'IDC连接器（牛角）':
        aa_label = 'IDC连接器(牛角)'
    elif aa_label == '线对板/线对线连接器':
        aa_label = '线对板线对线连接器'
    elif aa_label == '结型场效应晶体管（JFET）':
        aa_label = '结型场效应晶体管(JFET)'
    elif aa_label == '电机马达点火驱动器I':
        aa_label = '电机马达点火驱动器IC'
    elif aa_label == 'LEDUPS等其他类型电源模块':
        aa_label = '其他模块'
    elif aa_label == '8位微控制器':
        aa_label = '8位微控制器-MCU'
    elif aa_label == '16位微控制器':
        aa_label = '16位微控制器-MCU'
    elif aa_label == '32位微控制器':
        aa_label = '32位微控制器-MCU'
    elif aa_label =='ARM微控制器':
        aa_label = 'ARM微控制器-MCU'
    elif aa_label == 'CPLDFPGA芯片':
        aa_label = 'CPLD-FPGA芯片'
    elif aa_label =='罩类、盒类及壳类产品':
        aa_label = '罩类盒类及壳类产品'
    elif aa_label=='光藕':
        aa_label = '光耦'
    elif aa_label=='安全（加密）IC':
        aa_label = '安全(加密)IC'
    elif aa_label=='高速、宽带运放':
        aa_label = '高速宽带运放'
    elif aa_label=='音视频IC':
        aa_label = '音频视频接口芯片'



    return aa_label