#Author:wqh
#简单登录，其中的验证以及用户的读取应该在进行优化
import getpass
count = 0
while count < 3:
    # 提示用户输入用户名与密码
    login_name = input("please input your id:")
    login_password = input("please input your password:")
    #打开login_user.txt文件读取其中的用户名，并将用户名赋值给变量login_user
    with open("login_user.txt","r") as f4:
        login_user = f4.readline()
    #判断被锁定用户与输入用户是否一致，一致则程序退出
    if login_name == login_user:
        print("this id is locked,program exit")
        break
    else:
        #打开user.txt文件进行数据的读取
        with open("user.txt", "r") as f1:
            data_user = f1.readline()
        #打开password.txt文件，将其中的密码读取出来
        with open("password.txt","r") as f2:
            data_password = f2.readline()
        #如果登录名与密码相同则登录成功，否则登录失败
        if login_name == data_user and login_password == data_password:
            print("welcome!")
            break
        else:
            print("Verification error, please re-enter")
        #对于登录次数进行计数，也就是个计数器
        count = count+1
        #如果用户登录错误数达到三次，则将该id记下，写入文件中，以便下次登录时进行检验
        if count == 3:
            with open("login_user.txt", "w") as f3:
                 f3.writelines(login_name)
                 print("you have tired 3 times,this id is locked")
                 break





