#Author:wqh
age_of_oldboy = 56
count = 0



while count < 3:
    guess_age = int(input("guess age:"))
    if guess_age == age_of_oldboy :
        print("yes,you got it.")
        break
    elif guess_age > age_of_oldboy:
        print("think smaller...")
    else:
        print("think bigger!")
    count = count + 1
    #此处到了三次以后加了一个判断，判断用户是否想继续执行
    if count == 3:
        countine_confirm = input("do you want to keep guessing?")
        #如果继续执行输入n，再将计数器清零，返回上一个while循环中继续运行
        if countine_confirm != 'n':
            count = 0
else:
    print("you have tired 3 times,game exit")



