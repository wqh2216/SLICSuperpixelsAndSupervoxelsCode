#Author:wqh
#深拷贝，浅拷贝
import copy

person = ['name',['saving',100]]
'''
p1 = copy.copy(person)#做浅拷贝的一种方式
p2 = person[:]#浅拷贝的另一种方式
p3 = list(person)#工厂方式
'''

p1 = person[:]
p2 = person[:]

p1[0] = 'lisan'
p2[0] = 'wangwu'
print(p1,p2)