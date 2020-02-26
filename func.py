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
jieba.load_userdict('dict_boom.txt')

# classfier = ff.load_model("Model/model_w1_e23")
def txt_write_line(save_path, line):  # 将某一行写入txt
    filenames = save_path
    # print("wenjianm:" + filenames)
    # fs_list = []
    try:
        # fs_list.append(open(filenames, 'a', encoding='utf-8'))  # a 指定打开文件的模式，a为追加   r为只读
        with open(filenames, mode='a+', encoding='utf-8') as f:
            f.write(line + '\n')
            # f.write(line)
    except IOError as ex:
        print(ex)
        print('\033[1;31m 写文件时发生错误!!!!\033[0m')
        # print(self.file_path)
        return False

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



# def new_string(str1,spilt_str=None):  # 分词规范化, ''__label__电阻	精密贴片电阻	1KΩ ±1%   0603	   5000--->>''__label__电阻 精密贴片电阻 1KΩ ±1% 0603 5000
def string_split_combine(str1,spilt_str=None):  # 分词规范化, ''__label__电阻	精密贴片电阻	1KΩ ±1%   0603	   5000--->>''__label__电阻 精密贴片电阻 1KΩ ±1% 0603 5000
    new_str = ''
    new_list = str1.split(spilt_str)
    for i in new_list:
        new_str += i + ' '
    return new_str.lstrip()

# def sort_test(str1):


# def standard(str1, model_path):
def standard(str1):
    # 加载停用词表
    stop_words = load_stop_word_list("stopwords.txt")
    # classfier = ff.load_model(model_path)
    # print(model_path,'--------')
    # label_to_name = load_label_name_map()[0]
    # classifier = ff.load_model("Model/model_w1_e177")   #
    # classifier = ff.load_model("Model/model_w1_e48")
    # classifier = ff.load_model("Model/model_w1_e223")

    aa_description = str1.lower()
    aa_description = aa_description.replace('nan', '')

    aa_description = ' '.join(jieba.cut(str(aa_description).lower()))  # 先jieba 分词
    print('jieba 分词后输入：{}'.format(aa_description))
    for word in aa_description.split():  # 停用词使用
        # print(word)
        if str(word) in stop_words:
            aa_description = aa_description.replace(word, '')
    aa_description = ' '.join(filter(lambda x: x, aa_description.split(' ')))
    # for word in input_:
    #     # print(word)
    #     if str(word) in stop_words:
    #         input_ = input_.replace(word, ' ')
    print('excel行最终输入：{}'.format(aa_description))
    return aa_description


def label_new(label_origin):
    aa_label = label_origin

    if aa_label == '贴片电容;':
        aa_label = '贴片电容'
    if aa_label == '铝质电解电容器-SMD：' or aa_label=='铝质电解电容器-SMD' or aa_label == '铝有机聚合物电容器':
        aa_label = '贴片电解电容'
    if aa_label == '电阻贴片':
        aa_label = '贴片电阻'
    if aa_label == '电阻贴片':
        aa_label = '贴片电阻'
    if aa_label == 'MLCC-SMDSMT':
        aa_label = '贴片电容'
    if aa_label == '厚膜电阻器' or aa_label == '薄膜电阻器' or aa_label == '芯片电阻-表面安装':
        aa_label = '贴片电阻'
    if aa_label == '铝电解电容器-带引线' or aa_label == '直插电解电容:' or aa_label=='铝电铝电解电容器-带引线解电容器-带引线' or aa_label=='铝质电解电容器-螺旋式接线端':
        aa_label = '直插电解电容'
    if aa_label == '直插瓷片电容:':
        aa_label = '直插瓷片电容'
    if aa_label == '安规X电容:' or aa_label == '安规X电容' or aa_label == '安规电容直插' or aa_label == '安规Y电容':
        aa_label = '安规电容'
    if aa_label == '超级电容' or aa_label == '固态电解电容':
        aa_label = '超级电容器'
    if aa_label == '碳膜电阻器':
        aa_label = '碳质电阻器'
    if aa_label == '电阻器网络与阵列' or aa_label == '贴片排阻':
        aa_label = '排阻'
    if aa_label == '电位器-其他可调电阻':
        aa_label = '电位计'
    if aa_label == '钽质电容器-固体SMD钽电容器:':
        aa_label = '钽质电容器-固体SMD钽电容器'
    if aa_label == '贴片低阻值采样电阻' or aa_label == '电流传感电阻器' or aa_label == '直插低阻值采样电阻':
        aa_label = '采样电阻'
    if aa_label == '直插压敏电阻' or aa_label == '贴片压敏电阻':
        aa_label = '压敏电阻'
    if aa_label == 'MELF电阻':
        aa_label = 'MELF/晶圆电阻'
    if aa_label == 'NTC热敏电阻':
        aa_label = 'NTC'
    if aa_label == 'PTC热敏电阻':
        aa_label = 'PTC'
    if aa_label == '金属氧化物电阻器':
        aa_label = '金属氧化膜电阻'
    if aa_label == '碳质电阻器' or aa_label == '碳膜电阻器':
        aa_label = '碳膜电阻'
    if aa_label == '高功率贴片电阻' or aa_label == '铝壳电阻' or aa_label == '直插功率电阻' or aa_label == '贴片功率电阻' or aa_label == 'TO封装平面功率电阻':
        aa_label = '铝壳/大功率电阻'
    if aa_label == '高频/射频电阻':
        aa_label = '射频/高频电阻'
    if aa_label == '电位器-其他可调电阻' or aa_label == '变阻器' or aa_label == '电位计' or aa_label == '电位计工具及硬件' or aa_label == '可调功率电阻' or aa_label == '精度电位计':
        aa_label = '可调电阻/电位器'
    if aa_label == '微调电阻器SMD' or aa_label == '微调电阻器通孔':
        aa_label = '精密可调电阻'
    if aa_label == '线绕电阻' or aa_label == '线绕电阻器-透孔' or aa_label == '线绕电阻器':
        aa_label = '绕线电阻'
    if aa_label == '金属玻璃釉电阻' or aa_label == '金属薄膜电阻器' or aa_label == '通孔电阻器':
        aa_label = '直插/通孔电阻'
    if aa_label == '电阻套件' or aa_label == '电阻硬件':
        aa_label = '电阻套件及附件'
    if aa_label == '贴片精密电阻' or aa_label=='贴片高精密、低温漂电阻':
        aa_label = '贴片高精密-低温漂电阻'
    if aa_label == '高压陶瓷电容' or aa_label == '高压瓷片电容' or aa_label == '瓷片电容器':
        aa_label = '直插瓷片电容'
    if aa_label == '钽质电容器-固体铅钽电容器' or aa_label =='液体钽电容器' or aa_label == '钽质电容器-固体SMD钽电容器' or aa_label == '钽质电容器-SMD聚合物液体钽电容器':
        aa_label = '钽电容'
    if aa_label == '云母电容器':
        aa_label = '云母电容'
    if aa_label == '微调电容器与可变电容器':
        aa_label = '可调电容'
    if aa_label == '薄膜电容器' or aa_label =='CL21电容' or aa_label == '聚酯薄膜电容':
        aa_label = '薄膜电容'
    if aa_label == '电容套件' or aa_label == '电容硬件':
        aa_label = '电容套件及附件'
    if aa_label == 'MLCC-含引线':
        aa_label = '直插独石电容'
    if aa_label == '贴片绕线电感' or aa_label=='贴片线绕电感':
        aa_label = '贴片电感'
    if aa_label == '固定值电感':
        aa_label = '固定电感'
    if aa_label == '电感套件' or aa_label == '电感套件及配件' or aa_label == '可变电感' or aa_label == '可调电感':
        aa_label = '可变电感器/套件/配件'
    if aa_label =='贴片晶体谐振器(有源)' or aa_label == '直插晶体振荡器(有源)' or aa_label == '贴片晶体振荡器(有源)' or aa_label == '压控振荡器（VCO）' or aa_label == '压控式晶体振荡器(VCXO)' or aa_label == '温度补偿晶体振荡器(TCXO)' or aa_label == '恒温晶体振荡器' or aa_label == '压控振荡器' or aa_label == '温度补偿压控晶体振荡器':
        aa_label = '有源晶体振荡器'
    if aa_label == '无源晶体振荡器:' or aa_label == '贴片晶体振荡器(无源)' or aa_label =='直插晶体谐振器(无源)' or aa_label == '贴片晶体谐振器(无源)':
        aa_label = '无源晶体振荡器'
    if aa_label == '陶瓷谐振器' or aa_label == '压控SAW振荡器' or aa_label == '声表谐振器':
        aa_label = '谐振器'
    if aa_label == '电容器网络，阵列':
        aa_label = '电容器阵列与网络'
    return aa_label