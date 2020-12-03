# -*- coding: utf-8 -*-
"""
@Time : 2020/11/16 17:09
@Author : Dufy
@Email : 813540660@qq.com
@File : main.py
@Software: PyCharm
Description :
1)实现 对输入字符串 返回前缀最大匹配集
2)  构建字典树还存在一些问题，如 输入 ‘stm8’显示型号在库，却拿不到具体类目，返回 None（权宜之计）
Reference :
"""
import os
import pandas as pd
import time
from data_operation.triee import Trie
import collections
import pickle
from tqdm import tqdm


def levenshteinDistance(word1: str, word2: str) -> int:
    visited = set()
    queue = collections.deque([(word1, word2, 0)])

    while True:
        w1, w2, d = queue.popleft()
        if (w1, w2) not in visited:
            if w1 == w2:
                return d
            visited.add((w1, w2))
            while w1 and w2 and w1[0] == w2[0]:
                w1 = w1[1:]
                w2 = w2[1:]
            d += 1
            queue.extend([(w1[1:], w2[1:], d), (w1, w2[1:], d), (w1[1:], w2, d)])

def createTRIE(trie):
    """
    建TRIE树
    :param trie:
    :return:
    """
    print('TRIE树构建中....')
    df = pd.read_csv(r'D:\py_grpc\model\tree_pn-class1.csv', sep=',',
                     header=None,
                     names=['id', 'name'])
    length = df.shape[0] # 文本总行数
    df = pd.DataFrame(None)
    i=0
    with open(r'D:\py_grpc\model\tree_pn-class1.csv',mode='r',encoding='utf-8') as f1:
        for line in f1:
            i += 1
            pn_str = standard(line.strip().split(',',1)[1])
            # print(pn_str)
            trie.insert(pn_str)
            if i%100000 == 0:
                print('进度：{:.2f}'.format(i/length))

    print('TRIE树构建完成')

def createDic():
    """
    建型号字典, {pn:id}
    :param trie:
    :return:
    """
    dic = {}
    with open(r'D:\py_grpc\model\tree_pn-class1.csv', mode='r', encoding='utf-8') as f1:
        for line in f1:
            # pn_str = standard(line.strip().split(',', 1)[1])
            pn_str = standard(line.strip().split(',', 1)[1])
            class_id = line.strip().split(',', 1)[0]
            dic[pn_str]= class_id

    pickle.dump(dic, open(r'D:\dufy\code\class_pn.data', mode='wb'))

def standard(string):
    string = string.strip().replace('"','').replace(' ','').replace(',','')
    return string

def load_dic():
    return pickle.load(open(r'D:\dufy\code\class_pn.data', mode='rb'))  # {pn:id}


id_name_dict = {47: '32位微控制器-MCU',
                -1: 'None',
 48: '16位微控制器-MCU',
 49: 'ARM微控制器-MCU',
 50: '微处理器-MPU',
 51: '其他处理器',
 52: '数字信号处理器和控制器',
 53: '8位微控制器-MCU',
 54: 'CPLD-FPGA芯片',
 55: '安全(加密)IC',
 56: 'MCU监控芯片',
 57: '时钟计时芯片',
 58: '实时时钟芯片',
 59: '字库芯片',
 60: '时钟缓冲器驱动器',
 61: '时钟发生器频率合成器',
 62: 'FLASH存储器',
 63: 'SRAM存储器',
 64: 'EPROM存储器',
 65: 'SDRAM存储器',
 66: 'SDMicro-SDT-Flash卡',
 67: 'PROM存储器',
 68: 'EEPROM存储器',
 69: 'DDRSSD存储器',
 70: 'FPGA-配置存储器',
 71: '时基芯片',
 72: '信号开关多路复用解码器',
 73: '锁存器',
 74: '编解码芯片',
 75: '4000系列逻辑芯片',
 76: '74系列逻辑芯片',
 77: '缓冲器驱动器接收器收发器',
 78: '移位寄存器',
 79: '专用逻辑芯片',
 80: '触发器',
 81: '多频振荡器',
 82: '门极反相器',
 83: '计数器除法器',
 84: '电平转换移位器',
 85: '隔离芯片',
 86: '以太网芯片',
 87: 'USB芯片',
 88: 'RS485RS422芯片',
 89: '直接数字合成器(DDS)',
 90: 'RS232芯片',
 91: 'LVDS芯片',
 92: '传感器接口芯片',
 93: 'LIN收发器',
 94: '信号缓冲器中继器分配器',
 95: 'IO扩展器',
 96: '控制器',
 97: '音频视频接口芯片',
 98: '电信',
 99: '接口专用芯片',
 100: '串行接口芯片',
 101: '触摸屏控制器',
 102: 'CAN芯片',
 103: '磁珠',
 104: '有源滤波器',
 105: 'RF滤波器',
 106: '共模扼流圈滤波器',
 107: '馈通式电容器',
 108: 'EMIRFI滤波器',
 109: '铁氧体磁芯与配件',
 110: '信号调节器',
 111: '线对板线对线连接器',
 112: '排针排母',
 113: '板对板连接器',
 114: '背板连接器',
 115: 'USB连接器',
 116: '插拔式连接器',
 117: '螺丝钉接线端子',
 118: '弹簧式接线端子',
 119: '轨道式接线端子',
 120: '压接端子',
 121: 'IO连接器',
 122: 'D-Sub连接器附件',
 123: '音频视频链接器',
 124: 'IEEE1394连接器',
 125: '连接器附件套件',
 126: '照明连接器',
 127: 'RF同轴连接器',
 128: '栅栏式接线端子',
 129: 'FFCFPC连接器',
 130: '圆形连接器',
 131: 'IDC连接器(牛角)',
 132: '汽车连接器',
 133: '电源连接器',
 134: '军工连接器',
 135: '以太网连接器',
 136: 'IC与器件插座',
 137: '内存连接器',
 138: '卡缘连接器',
 139: '鳄鱼夹测试夹',
 140: '贴片电感',
 141: '高频电感',
 142: '工字电感',
 143: '功率电感',
 144: '色环电感',
 145: '固定电感',
 146: '可变电感器套件配件',
 147: '贴片电容',
 148: '贴片电解电容',
 149: '直插电解电容',
 150: '直插瓷片电容',
 151: '云母电容',
 152: '安装型大容量电容',
 153: '固态电解电容',
 154: '钽电容',
 155: '安规电容',
 156: '薄膜电容',
 157: '可调电容',
 159: '超级电容器',
 160: '氧化铌电容',
 161: '电容器阵列与网络',
 162: '电容套件及附件',
 163: '直插独石电容',
 164: '校正电容',
 165: '铝电解电容',
 166: 'NTC热敏电阻',
 167: '贴片电阻',
 169: '采样电阻',
 170: '光敏电阻',
 171: '压敏电阻',
 172: 'MELF晶圆电阻',
 173: '金属膜电阻',
 174: 'PTC热敏电阻',
 175: '金属氧化膜电阻',
 176: '碳膜电阻',
 177: '薄膜电阻-透孔',
 178: '厚膜电阻-透孔',
 179: '铝壳大功率电阻',
 180: '射频高频电阻',
 181: '陶瓷复合电阻器',
 182: '可调电阻电位器',
 183: '精密可调电阻',
 184: '绕线电阻',
 185: '保险电阻',
 186: '水泥电阻',
 187: '金属箔电阻',
 188: '高压电阻',
 189: '排阻',
 190: '直插通孔电阻',
 191: '电阻套件及附件',
 192: '贴片高精密-低温漂电阻',
 193: '无源晶体振荡器',
 194: '有源晶体振荡器',
 195: '圆柱体晶振',
 196: '谐振器',
 197: '可编程振荡器',
 198: '标准时钟振荡器',
 199: '稳压二极管',
 200: '通用二极管',
 201: '功率二极管',
 202: '开关二极管',
 203: '超快快恢复二极管',
 204: '肖特基二极管',
 205: '双向触发二极管DIAC',
 206: 'TVS二极管(瞬态电压抑制二极管)',
 207: 'ESD二极管',
 208: '变容二极管VaractorDiode',
 209: '稳流二极管CRD',
 210: 'PIN二极管',
 211: '整流器',
 212: '整流桥',
 213: '放电管',
 214: '特殊功能放大器',
 215: '视频放大器',
 216: '放大器',
 217: '高速宽带运放',
 218: '仪表运放',
 219: '精密运放',
 220: 'FET输入运放',
 221: '低噪声运放',
 222: '低功耗比较器运放',
 223: '差分运放',
 224: '电压比较器',
 225: '采样保持放大器',
 226: 'LCDGamma缓冲器',
 227: 'LED驱动',
 228: 'LCD驱动',
 229: '电机马达点火驱动器IC',
 230: 'MOS驱动',
 231: '激光驱动器',
 232: '达林顿晶体管阵列驱动',
 233: '驱动芯片',
 234: '门驱动器',
 235: '全桥半桥驱动',
 236: '电子辅料',
 237: '风扇散热片热管理产品',
 238: '焊接脱焊',
 239: '罩类盒类及壳类产品',
 240: '胶带标签',
 241: '容器类',
 242: '螺丝刀镊子扳手工具',
 243: '化学物质',
 244: 'PCB等原型产品',
 245: '配件',
 246: '螺丝紧固件硬件',
 247: '机架机柜',
 248: '机架/机柜',
 249: '线性稳压芯片LDO',
 250: '开关电源芯片',
 251: 'DC-DC芯片',
 252: '电池电源管理芯片PMIC',
 253: '电池保护芯片',
 254: '电压基准芯片',
 255: '电源监控芯片',
 256: '功率开关芯片',
 257: 'DC-DC电源模块',
 258: 'AC-DC电源模块',
 259: '无线充电IC',
 260: 'LEDUPS等其他类型电源模块',
 261: '开发板套件',
 262: 'WiFi物联网模块',
 263: 'WiFi/物联网模块',
 264: '无线模块',
 265: '传感器模块',
 266: '电力线滤波器模块',
 267: '其他模块',
 268: '蜂鸣器',
 269: '扬声器喇叭',
 270: '咪头麦克风',
 272: '电源变压器',
 273: '电流变压器',
 274: '网口变压器',
 275: '工业控制变压器',
 276: '脉冲变压器',
 277: '音频及信号变压器',
 278: '自耦变压器',
 279: '信号继电器',
 280: '继电器插座配件',
 281: '车用继电器',
 282: '固态继电器',
 283: '安全继电器',
 284: '簧片继电器',
 285: '高频射频继电器',
 286: '延时计时继电器',
 287: '工业继电器',
 288: '按键开关',
 289: '船型开关',
 290: '行程开关',
 291: '拨码开关',
 292: '拨动开关',
 293: '五向开关',
 294: '多功能开关',
 295: '锅仔片',
 296: '旋转编码开关',
 297: '开关插座',
 298: '带灯开关',
 299: '旋转波段开关',
 300: '交流接触器',
 301: '压接接触器',
 302: '专用开关',
 303: '开关配件-盖帽',
 304: '轻触开关',
 305: '发光二极管',
 306: '光耦',
 307: '红外发射管',
 308: '红外接收管',
 309: 'LED显示模组',
 310: 'LED数码管',
 311: 'LCD显示模组',
 312: '光可控硅',
 313: 'OLED显示模组',
 314: 'LED灯柱导光管配件',
 315: '真空荧-VFD光显示器',
 316: '等离子体显示器',
 317: '红外收发器',
 318: '光纤收发器',
 319: '光电开关',
 320: '激光器件配件',
 321: '温度传感器',
 322: '超声波传感器',
 323: '气体传感器',
 324: '光学传感器',
 325: '压力传感器',
 326: '颜色传感器',
 327: '图像传感器',
 328: '环境光传感器',
 329: '红外传感器',
 330: '角速度传感器',
 331: '加速度传感器',
 332: '角度传感器',
 333: '位置传感器',
 334: '姿态传感器',
 335: '磁性传感器',
 336: '电流传感器',
 337: '湿度传感器',
 338: '专用传感器',
 339: '模数转换芯片',
 340: '数模转换芯片',
 341: '模拟开关芯片',
 342: '电流监控芯片',
 343: '电量计芯片',
 344: '数字电位器芯片',
 345: '电池',
 346: '电源充电器',
 347: '电池座夹附件',
 348: '线材配件附件',
 349: '数据线信号线',
 350: '电源线',
 351: '多芯电缆',
 352: '同轴电缆',
 353: '电子线材连接线',
 354: '贴片式一次性保险丝',
 355: 'PTC自恢复保险',
 356: '通孔型保险丝',
 357: '保险丝管',
 358: '工业与电气保险丝',
 359: '汽车保险丝',
 360: '特种保险丝',
 361: '温度保险丝',
 362: '保险丝座夹',
 363: '断路器',
 364: 'MOSFET',
 365: '结型场效应晶体管(JFET)',
 366: '可控硅SCR',
 367: '达林顿管',
 368: '数字三极管',
 369: 'IGBT管',
 370: '双极晶体管(三极管)',
 371: 'LAN电信电缆测试',
 372: '万用表与电压表',
 373: '测试与测量',
 374: '仪器设备与配件',
 375: '无线收发芯片',
 376: '射频卡芯片',
 377: '天线',
 378: '射频开关',
 379: 'RF放大器',
 380: 'RF混频器',
 381: 'RF检测器',
 382: 'RF衰减器',
 383: 'RF耦合器',
 385: 'RFFETMOSFET',
 409: '第一次导入',
 411: '11是发广告的的股份',
 412: '机电电气',
 413: '高频继电器',
 414: '通信卫星定位模块',
 415: 'RF双工器',
 416: 'IGBT驱动',
 417: 'FRAM存储器',
 418: '温控开关',
 419: '温湿度传感器',
 420: '电路保护套件',
 422: 'LED管'}


def getClassFromPn(query, trie,class_pn_dic):
    pass
    query = standard(query)
    if len(query)>3:
        pass
        if trie.search(query):
            print('型号在库')
            label = id_name_dict[int(class_pn_dic.setdefault(query, -1))]
            print('完全匹配为： \033[1;31m {} \033[0m '.format(label))
            return label
        else:
            pass
            print('进行型号前缀匹配判断....')
            for i in range(len(query),3,-1):  # 前缀倒着取值
                # print(query[:i])
                tmp = query[:i]
                if trie.startsWith(tmp): # 如果存在最长匹配
                    return_match_li = trie.associate_search(tmp)  # 返回匹配列表
                    print(f'返回匹配列表长度：{len(return_match_li)}, {return_match_li}')

                    # print('=='*20)
                    print("查询量\t匹配量\t类目\t编辑距离")
                    static_pn = {}  # 类目出现次数统计
                    min_levenshtein_dist= float('inf')
                    for match_str in return_match_li:
                        levenshtein_dist = levenshteinDistance(query, match_str)
                        label = id_name_dict[int(class_pn_dic[match_str])]
                        if levenshtein_dist<min_levenshtein_dist: # 更新最小编辑距离
                            min_levenshtein_dist= levenshtein_dist
                            static_pn = {}  # 把之前结果清空
                            static_pn[label] = static_pn.setdefault(label, 0) + 1
                        elif levenshtein_dist==min_levenshtein_dist:
                            static_pn[label] = static_pn.setdefault(label, 0) + 1

                        print(f'{query}\t{match_str}\t{class_pn_dic[match_str]}/{label}\t{levenshtein_dist}')
                    print(static_pn)
                    print('最小编辑距离：',min_levenshtein_dist)

                    print('最终结果判断为： \033[1;31m {} \033[0m '.format(max(static_pn,key=static_pn.get)))
                    # print(f'最终结果判断为\t{max(static_pn,key=static_pn.get)}')
                    # break
                    return max(static_pn,key=static_pn.get)
            else:
                print('判断完毕')
                print('前缀长度<3')  # 结果返NONE
                return None
    else:
        print('输入的字符串长度太短，请重新输入!!!!!!')
        return None

    # print(f'用时：{time.time() - time0}')
    # print('==' * 20)


if __name__ == "__main__":
    trie = Trie()
    # li = ['appo', 'sort', 'sorg', 'sorted', 'boom', 'boote', 'bootm', 'bole', 'bfff']
    # for i in li:
    #     trie.insert(i)

    createTRIE(trie)
    createDic()   # 只需构建一次
    class_pn_dic = load_dic()  # {pn:id}
    print(trie.search('boom'))
    print(getClassFromPn('MIC803-29D4VM3-TR', trie, class_pn_dic))

    while 1:
        query = standard(input('属输入查询型号：'))
        time0 = time.time()
        if len(query)>3:
            pass
            print(trie)
            if trie.search(query):
                print('型号在库')
                label = id_name_dict[int(class_pn_dic.setdefault(query, -1))]
                print('完全匹配为： \033[1;31m {} \033[0m '.format(label))
            else:
                pass
                print('进行型号前缀匹配判断....')
                for i in range(len(query),3,-1):  # 前缀倒着取值
                    # print(query[:i])
                    tmp = query[:i]
                    if trie.startsWith(tmp): # 如果存在最长匹配
                        return_match_li = trie.associate_search(tmp)  # 返回匹配列表
                        print(f'返回匹配列表长度：{len(return_match_li)}, {return_match_li}')

                        # print('=='*20)
                        print("查询量\t匹配量\t类目\t编辑距离")
                        static_pn = {}  # 类目出现次数统计
                        min_levenshtein_dist= float('inf')
                        for match_str in return_match_li:
                            levenshtein_dist = levenshteinDistance(query, match_str)
                            label = id_name_dict[int(class_pn_dic[match_str])]
                            if levenshtein_dist<min_levenshtein_dist: # 更新最小编辑距离
                                min_levenshtein_dist= levenshtein_dist
                                static_pn = {}  # 把之前结果清空
                                static_pn[label] = static_pn.setdefault(label, 0) + 1
                            elif levenshtein_dist==min_levenshtein_dist:
                                static_pn[label] = static_pn.setdefault(label, 0) + 1

                            print(f'{query}\t{match_str}\t{class_pn_dic[match_str]}/{label}\t{levenshtein_dist}')
                        print(static_pn)
                        print('最小编辑距离：',min_levenshtein_dist)

                        print('最终结果判断为： \033[1;31m {} \033[0m '.format(max(static_pn,key=static_pn.get)))
                        # print(f'最终结果判断为\t{max(static_pn,key=static_pn.get)}')
                        break
                else:
                    print('判断完毕')
                    print('前缀长度<3')  # 结果返NONE
        else:
            print('输入的字符串长度太短，请重新输入!!!!!!')

        print(f'用时：{time.time() - time0}')
        print('==' * 20)
