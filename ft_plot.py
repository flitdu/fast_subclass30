# -*- coding: utf-8 -*-
"""
@Time : 2020/5/28 17:23
@Author : Dufy
@Email : 813540660@qq.com
@File : ft_plot.py
@Software: PyCharm 
Description :
1)
2)
Reference :       
"""
import os
import time
import matplotlib.pyplot as plt


def plotTrainEffect(ft_,train_accuracy_list,train_f1_macro_list,val_accuracy_list,val_f1_macro_list):
    # 绘制训练集和验证集数据比较
    plot_x = list(range(1, ft_.epoch))
    plt.figure()
    plt.plot(plot_x, train_accuracy_list, color="k", linestyle="-", marker="^", linewidth=1, label="train_accu")
    plt.plot(plot_x, train_f1_macro_list, color="k", linestyle="-", marker="X", linewidth=1, label="train_f1")
    plt.plot(plot_x, val_accuracy_list, color="r", linestyle="-", marker="^", linewidth=1, label="val_accu")
    plt.plot(plot_x, val_f1_macro_list, color="r", linestyle="-", marker="X", linewidth=1, label="val_f1")
    # plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("property", fontsize=20)
    plt.legend()
    # plt.title("{}-gram".format(w))
    plt.grid()
    # time_temp = time.strftime("(T%Y-%m-%d-%H-%M)", time.localtime())
    time_temp = time.strftime("(T%Y-%m-%d-%H)", time.localtime())
    plt.savefig(r".\pic\{}.png".format(time_temp + "rate" + str(ft_.lr) + '_' + str(ft_.n_gram) + '-gram_' + ft_.loss))
    plt.show()


def plotCompareModelAccuracy(dict_model_record):
    # 绘制不同模型的准确率比较
    x = []
    y = []
    for key, value in dict_model_record.items():
        print(value)
        x.append(key.strip('model_'))  # append() 方法用于在列表末尾添加新的对象。
        y.append(value)

    plt.plot(x, y, "b-o", linewidth=2)
    plt.xlabel("model", fontsize=20)  # X轴标签
    plt.ylabel("accu", fontsize=20)  # Y轴标签
    plt.title("不同模型准确率比较")  # 图标题
    plt.grid()
    plt.show()  # 显示图

def plotScatterRightWrongMark(record_wrong_probability_list, record_right_probability_list):
    # 绘制分类正确、错误的散点图
    plt.scatter(list(range(len(record_wrong_probability_list))), record_wrong_probability_list, color="r",
                marker="x", linewidth=1, label="wrong label")
    plt.scatter(list(range(len(record_right_probability_list))), record_right_probability_list, color="b",
                marker="o", linewidth=1, label="right label")
    plt.grid()
    plt.xlabel("Sample Number")
    plt.ylabel("Probability")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":

    pass

