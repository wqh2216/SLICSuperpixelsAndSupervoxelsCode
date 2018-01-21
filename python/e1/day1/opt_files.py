#Author:wqh
# -*- coding: utf-8 -*-
#打开一个文件名，如果没有就创建这个文件
user_name = input("please input your id:")
# fo = open("foo.txt","w")
# fo.writelines(user_name)
# print("文件名:",fo.name)
# 关闭文件
# fo.close()
with open("foo.txt",mode='w',encoding='utf-8') as f:
    f.writelines(user_name)