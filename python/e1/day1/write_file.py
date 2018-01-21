#Author:wqh
login_name = input("please input your id:")
login_password = input("please input your password:")
# 应该使用with打开文件，因为使用with打开文件后，文件会关闭
with open("user.txt", "a") as f1:
    write = (login_name)
    f1.write(write)
with open("password.txt", "a") as f2:
    write = (login_password)
    f2.write(write)
