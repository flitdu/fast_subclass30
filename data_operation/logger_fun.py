# -*- coding: utf-8 -*-
"""
Created by Dufy on 2020/3/26  18:32
IDE used: PyCharm 
Description :
1)
2)  
Remark:      
"""

import logging.config
import yaml

# # 日志文件配置
def get_logger():
    log_conf = 'logging.config.yaml'
    with open(log_conf, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    return logging.getLogger()
