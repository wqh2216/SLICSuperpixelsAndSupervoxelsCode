#Author:wqh
import copy
#切记，列表的位置是从0开始计的
names = "张三 李四 王五"


#定义一个列表
names = ["张三","李四","王五","马六"]


names.append("张二")#追加，将张二追加到names列表最后的位置中
names.insert(1,"张大")#使用insert将张大插入至names列表中自己想插入的位置，在此处，我想插入的位置是第一个位置，也就是李四的前面
names.insert(3,"李二")#随意插入位置，插入第三个位置

#列表的扩展
names1 = [1,2,3,4]
names.extend(names1)
print(names)

#将其中的一个值进行修改，直接赋值即可
names[2] = "李留"

#将列表中的一个值给删除
names.remove("李留")#删除方法一
del names[1]#删除方法二
names.pop()#删除方法三，这个方法是把最后一个位置的元素删除（在没有任何下表的时候），如果想删除任意位置的值，输入下表即可

#反转整个列表的值
names.reverse()
print(names)

#排序,特殊符号在最前面，然后大写，然后小写
# names.sort()
# print(names)

#查询某个元素的位置
print(names[names.index("李二")])

#打印列表
print(names)


#取出列表中的第一个值，计算机中位置从零开始算
print(names[0],names[2])
#从第二个位置开始取，取到第三个位置，切片
print(names[1:2])#切片
print(names[-1])#切片，这种方式是从最后一个位置往前开始取，如果要取倒数第二个值，则将这里的值赋值为-2
print(names[-2:])#取最后的值把后面的-1省略

#统计某个元素的个数
print(names.count("李二"))
name2 = names.copy()
print(names)
print(name2)
names[0] = "liliu"

print(names)
print(name2)

#列表里面包含列表
names3 = ["1","2","3",["zhangsan","lisi"]]
#更改列表里面列表中的值
names3[3][0] = "张三"

print(names3)



