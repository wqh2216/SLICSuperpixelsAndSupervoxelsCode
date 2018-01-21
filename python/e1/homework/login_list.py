#Author:wqh
user_name = ["wqh","wqh1","wqh2"]
user_password = ["123","456","789"]


count = 0
while count < 3:
    login_name = input("please input your id:")
    login_password = input("please input your password:")
    #打开被锁定用户的文件，读取里面的用户名，如果是被锁定用户则告之用户该账号已被锁定，退出程序
    with open("locked_user.txt", "r") as f2:
        locked_user = f2.readline()
        if login_name == locked_user:
            print("this id is locke,program exit")
            break
    #对用户的登录进行验证
    if login_name == user_name[0] and login_password == user_password[0]:
        print("welcome",user_name[0])
        break
    if login_name == user_name[1] and login_password == user_password[1]:
        print("welcome",user_name[1])
        break
    if login_name == user_name[2] and login_password == user_password[2]:
        print("welcome",user_name[2])
        break
    else:
        print("validation error")
    count = count + 1
    #如果尝试并且错误，则将该用户id输入至用户锁定文件中，结束程序
    if count == 3:
        with open("locked_user.txt","w") as f1:
            f1.writelines(login_name)
        print("you hava tried 3 times,this id is locked")
        break