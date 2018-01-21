#Author:wqh
# -*- coding: utf-8 -*-
#上面的内容python2中会用到，主要用来处理乱码的问题
name = "wqh"
name2 = name
print("my name is ",name,name2)

name = "1122"
name3 = "天地"

print(name,name2)
print(name3)
username = input("username:")
password = input("password:")
print(username,password,)

#首先要先定义几个输入变量，然后再将这个变量格式化输出
name = input("name:")
age = int(input("age:")) #把字符型转成整型
print(type(age))
job = input("job:")
salary = input("salary:")

#格式化输出，info
info = '''
-------------- info of  %s  -------
Name:%s
Age:%d
Job:%s
Salary:%s
''' % (name,name,age,job,salary)
#格式化输出第二种方法
info2 = '''
    -------------- info of  {_name}  -------
Name:{_name}
Age:{_age}
Job:{_job}
Salary:{_salary}
'''.format(_name=name,
            _age=age,
            _job=job,
            _salary=salary)
print(info2)
#格式化输出第三种方法
info3  =  '''
    -------------- info of {0}   -------
Name:{0}
Age:{1}
Job:{2}
Salary:{3}
'''.format(name,age,job,salary)
print(info3)

