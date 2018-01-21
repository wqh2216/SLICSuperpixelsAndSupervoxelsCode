#Author:wqh
import getpass


_username = 'wqh'
_password = '123'


#暗文密码
# username = input("usernme:")
# password = getpass.getpass("password:")
#
# print(username,password)
#
# if _username == username and _password == password:
#     print("Welcome user {name} login...".format(name=username))
# else:
#     print("Invalid username or password")

#明文密码
username = input("usernme:")
password = input("password:")

print(username,password)

if _username == username and _password == password:
    print("Welcome user {name} login...".format(name=username))
else:
    print("Invalid username or password")