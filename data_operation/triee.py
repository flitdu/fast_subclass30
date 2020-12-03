# -*- coding: utf-8 -*-
"""
@Time : 2020/11/16 16:12
@Author : Dufy
@Email : 813540660@qq.com
@File : trie.py
@Software: PyCharm 
Description :
1)
2)
Reference :       
"""
import os
import pandas as pd
import time

# coding=utf-8
#字典嵌套牛逼,别人写的,这样每一层非常多的东西,搜索就快了,树高26.所以整体搜索一个不关多大的单词表,还是O(1).

'''
    Python 字典 setdefault() 函数和get() 方法类似, 如果键不存在于字典中，将会添加键并将值设为默认值。
    说清楚就是:如果这个键存在字典中,那么这句话就不起作用,否则就添加字典里面这个key的取值为后面的默认值.
    简化了字典计数的代码.并且这个函数的返回值是做完这些事情之后这个key的value值.
    dict.setdefault(key, default=None)
    Python 字典 get() 函数返回指定键的值，如果值不在字典中返回默认值。
    dict.get(key, default=None)
'''


class Trie:
    root = {}
    END = '/'  # 加入这个是为了区分单词和前缀,如果这一层node里面没有/他就是前缀.不是我们要找的单词.

    def insert(self, word):
        # 从根节点遍历单词,char by char,如果不存在则新增,最后加上一个单词结束标志
        node = self.root
        for c in word:
            """
            利用嵌套来做,一个trie树的子树也是一个trie树.
            利用setdefault的返回值是value的特性,如果找到了key就进入value
            没找到,就建立一个空字典
            """
            # if c not in node:
            #     node[c]={}
            # node=node[c]
            try:
                node = node.setdefault(c, {})
            except:
                pass
        try:
            node[self.END] = None   # 会添加一个类似 '{'/':None, ...}'的标志位，表明是词，非前缀
        except:
            pass
        # 当word都跑完了,就已经没有字了.那么当前节点也就是最后一个字母的节点
        # 加一个属性标签end.这个end里面随意放一个value即可.因为我们只是判定end这个key是否在字典里面.
        # 考虑insert 同一个单词2次的情况,第二次insert 这个单词的时候,因为用setdefault
        # insert里面的话都不对原字典进行修改.正好是我们需要的效果.
        # 这个self.END很重要,可以作为信息来存储.比如里面可以输入这个单词的
        # 起源,发音,拼写,词组等作为信息存进去.找这个单词然后读出单词的信息.

    def delete(self, word):  # 字典中删除word
        node = self.root
        for c in word:
            if c not in node:
                print('字典中没有不用删')
                return False
            node = node[c]
        # 如果找到了就把'/'删了
        del node['/']
        # 后面还需要检索一遍,找一下是否有前缀的后面没有单词的.把前缀的最后一个字母也去掉.因为没单词了,前缀也没意义存在了.
        # 也就是说最后一个字母这个节点,只有'/',删完如果是空的就把这个节点也删了.
        while node == {}:
            if word == '':
                return
            tmp = word[-1]
            word = word[:-1]
            node = self.root
            for c in word:
                node = node[c]
            del node[tmp]

    def search(self, word):
        node = self.root
        for c in word:
            if c not in node:
                return False
            node = node[c]
        return self.END in node

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        root = self.root
        for i in prefix:
            if i in root:
                root = root[i]
            else:
                return False
        return True

    def associate_search(self, pre):  # 搜索引擎里面的功能是你输入东西,不关是不是单词,他都输出以这个东西为前缀的单词.
        node = self.root
        for c in pre:
            if c not in node:
                return []  # 因为字典里面没有pre这个前缀
            node = node[c]  # 有这个前缀就继续走,这里有个问题就是需要记录走过的路径才行.
        # 运行到这里node就是最后一个字母所表示的字典.
        # 举一个栗子:图形就是{a,b,c}里面a的value是{b,c,d} d的value是{/,e,f} 那么/代表的单词就是ad,看这个形象多了
        # 首先看这个字母所在的字典有没有END,返回a这个list

        # 然后下面就是把前缀是pre的单词都加到a里面.
        # 应该用广度遍历,深度遍历重复计算太多了.好像深度也很方便,并且空间开销很小.
        # 广度不行,每一次存入node,没用的信息存入太多了.需要的信息只是这些key是什么,而不需要存入node.
        # 但是深度遍历,又需要一个flag记录每个字母.字典的key又实现不了.
        # 用函数递归来遍历:只能先用这个效率最慢的先写了
        # 因为你遍历一直到底,到底一定是'/'和None.所以一定travel出来的是单词不是中间结果.
        def travel(node):  # 返回node节点和他子节点拼出的所有单词
            if node == None:
                return ['']
            a = []  # 现在node是/ef

            for i in node:
                tmp = node[i]
                tmp2 = travel(tmp)
                for j in tmp2:
                    a.append(i + j)
            return a

        output = travel(node)
        for i in range(len(output)):
            output[i] = (pre + output[i])[:-1]
        return output


if __name__ == "__main__":
    time0 = time.time()
    pass
    trie = Trie()

    li = ['app', 'sor', 'se', 'bo', 'boo', 'bot', 'bf']
    for i in li:
        trie.insert(i)
    print(trie.search('bo'))

    res = trie.associate_search('s')
    print(res)
    print(trie.startsWith('soaa'))
    print(f'用时：{time.time() - time0}')
