# -*- coding: utf-8 -*-
"""
@Time : 2020/11/23 17:49
@Author : Dufy
@Email : 813540660@qq.com
@File : segment.py
@Software: PyCharm 
Description :
1)
2)
Reference :       
"""
import os
import pandas as pd
import time
import requests, json

def skk(string):
    # canshu = '功率贴片电感-47 uH -环保 5040 bav9,9 1 顺络'
    canshu = ' '
    params = {"bomName": "test", "env": "bom",
              "bomUrl": "test",
              "rowList": [{'row_index': 0,
                           # 'row_content': ["Ferrite Beads Multi-Layer 220Ohm 3% 100MHz 0.7A 0.28Ohm DCR 0402 T/R"],
                           'row_content': [string],
                           'part_list': [""]}]}
    headers = {
        'content-type': 'application/json',
    }
    url = 'http://192.168.0.102/cerebrum/ai/analytic'
    # print(url)
    # time1 = time.time()
    html = requests.post(url, headers=headers, data=json.dumps(params))
    # print("请求参数：" + canshu)
    # print('发送post数据请求成功!')
    # print('返回post结果如下：')
    # print(html.text)
    # print(type(html.text))
    result = json.loads(html.text)
    pn = result['data'][0]['row_data']['part']+result['data'][0]['row_data']['part_correct']
    print(pn)
    return pn


if __name__ == "__main__":
    time0 = time.time()
    pass
    skk("FP301-3/32-100'-BLACK-SPOO")


    print(f'用时：{time.time() - time0}')
