#Author:wqh
number = 30
for i in range(3):
    guess_number = int(input("guess_number:"))
    if guess_number > 30:
        print("bigger")
    elif guess_number <30:
        print("smaller")
    else:
        print("you are right!")
        break
else:
    print("you have tired 3 times,game exit")