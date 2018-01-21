#Author:wqh
#列表的循环
names = ["wqh","wqh1","wqh2","1","2","3"]
pwd = ["123","456"]
#隔两个值进行打印
print(names[0:-1:2])
#这句话的打印效果与上一句话打印效果一样
print(names[::2])

#从1到10每隔2打印一次数字
# range(1,10,2)

#对于列表中的值进行循环打印
for i in names:
    print(i)