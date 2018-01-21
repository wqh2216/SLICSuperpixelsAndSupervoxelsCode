#Author:wqh
#返回是0，则说明命令执行成功
import os

# cmd_res = os.system("dir")#执行命令不保存结果
cmd_res = os.popen("dir").read()#执行命令保存结果,打印出内存对象地址,读出结果
print("-->",cmd_res)
os.mkdir("new_dir")#创建一个新的目录
