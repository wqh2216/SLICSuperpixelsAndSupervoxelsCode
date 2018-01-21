#Author:wqh
people_list = [('张三',1),('李四',2),('王五',3),('马六',4),('张大',5)]
flag = 0
while flag is not True:
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for item in  enumerate(people_list,start = 1):
        index = item[0]
        p_name = item[1][0]
        p_class = item[1][1]
        print(index,p_name,p_class)
    break