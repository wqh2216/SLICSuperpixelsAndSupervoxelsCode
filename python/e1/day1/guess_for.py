#Author:wqh
age_of_oldboy = 56
for i in range(3):
    guess_age = int(input("guess age:"))
    if guess_age < age_of_oldboy:
        print("smaller")
    elif guess_age > age_of_oldboy:
        print("bigger")
    else:
        print("you are right")
        break
else:
    print("you have tired 3 times,game exit")
