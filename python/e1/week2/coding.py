#Author:wqh
#定义一个变量
msg = "某某人"
#输出变量
print(msg)
#打印msg的类型
print(type(msg))
#打印变量，并且把str类型转换成bytes类型
print(msg.encode(encoding="utf-8"))
#此处将bytes类型在转换为str的类型
print(msg.encode(encoding="utf-8").decode(encoding="utf-8"))