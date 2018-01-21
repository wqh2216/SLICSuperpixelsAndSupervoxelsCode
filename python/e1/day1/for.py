#Author:wqh
#i是一个临时变量，循环10次
'''
for i in range(10):
    print("loop",i)
#加入步长为2
for m in range(0,10):
    if m < 5:
        print("loop",m)
    else:
    #continue的作用是跳出本次循环，执行下次循环。break的作用是结束本次循环
        continue
    print("go on......")
    '''
for i in range(10):

#外层循环每当循环1次，内层循环循环10次
    print('--------------',i)
    for j in range(10):
        print(j)
        if j > 5:
            break
