#Author:wqh

#简单的猜数字游戏

game_number = 30
count = 0
while count < 3:
    # 当猜测次数小于三次时执行下面的语句
    guess_numebr = int(input("guess a number:"))
    #判断猜测数字与所给数字之间关系
    if guess_numebr == game_number:
        print("you are right!")
        break
    elif guess_numebr > game_number:
        print("bigger")
    else:
        print("smaller")
    count = count +1
    #如果猜测次数到达三次，那么进行询问是否继续进行游戏
    if count == 3:
        play = input("Do you want to play:(y/n)")
        if play == 'y':
            count = 0
        elif play =='n':
            print("game over")
else:
    print("you have tired 3 times,game exit")
