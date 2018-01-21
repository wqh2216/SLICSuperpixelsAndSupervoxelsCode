#Author:wqh
#定义一个商品的列表
goods = [('iphone',8000),('imac book',10000),('python book',100),('tv',5000),('others',1000)]
#定义一个空列表将用户购买的商品添加进去
user_choose = []
left_money_list = []

#输入用户的工资情况,
user_salary = int(input("please input your salary:"))#如果不把类型转换为int类型，那么类型就会是str程序运行如果不输入数值就会报错


#提醒用户是否想要购买商品
enter_entry = input("do you want to buy a commodity ? y/n:")


#进入循环，判断是否进行购买，以及输入的钱数是否够
while enter_entry == 'y':
    choose_number = int(input("please chose your commodity's number:"))
    # 把所有商品打印出来
    for item in enumerate(goods, start=1):
        index = item[0]
        g_name = item[1][0]
        g_price = item[1][1]
        print(index, g_name, g_price)
    #选择列表中商品进行购买,第一个选择的商品是商品1
    if choose_number == goods.index(('iphone',8000)) + 1:
        #用户输入的工资钱数与商品价格进行比较
        if user_salary >= goods[0][1]:
            print("added success!")
            #计算剩余的钱数
            left_money = user_salary - goods[0][1]
            #购买成功则将商品添加至空列表中,同时也将剩余的钱数添加至列表中
            user_choose.append(goods[0][0])
            #将剩余钱数放入剩余钱数空列表
            left_money_list = left_money
            #询问用户是否继续购买
            enter_quit = input("do you want to eixt? y/n:")
            if enter_quit == 'y':
                print("program exit")
                break
            else:
                print("please shopping")
                #将用户剩余的钱数赋值给用户的原来输入
                user_salary = left_money
                continue
    #以下的代码与上面的代码意思相同，故不写注释
    if choose_number == goods.index(('imac book',10000)) + 1:
        if user_salary >= goods[1][1]:
            print("added success!")
            left_money = user_salary - goods[1][1]
            user_choose.append(goods[1][0])
            left_money_list = left_money
            enter_quit = input("do you want to eixt? y/n:")
            if enter_quit == 'y':
                print("program exit")
                break
            else:
                print("please shopping")
                user_salary = left_money
                continue
    if choose_number == goods.index(('python book',100)) + 1:
        if user_salary >= goods[2][1]:
            print("added success!")
            left_money = user_salary - goods[2][1]
            user_choose.append(goods[2][0])
            left_money_list = left_money
            enter_quit = input("do you want to eixt? y/n:")
            if enter_quit == 'y':
                print("program exit")
                break
            else:
                print("please shopping")
                user_salary = left_money
                continue
    if choose_number == goods.index(('tv',5000)) + 1:
        if user_salary >= goods[3][1]:
            print("added success!")
            left_money = user_salary - goods[3][1]
            user_choose.append(goods[3][0])
            left_money_list = left_money
            enter_quit = input("do you want to eixt? y/n:")
            if enter_quit == 'y':
                print("program exit")
                break
            else:
                print("please shopping")
                user_salary = left_money
                continue
    if choose_number == goods.index(('others',1000)) + 1:
        if user_salary >= goods[4][1]:
            print("added success!")
            left_money = user_salary - goods[4][1]
            user_choose.append(goods[4][0])
            left_money_list = left_money
            enter_quit = input("do you want to eixt? y/n:")
            if enter_quit == 'y':
                print("program exit")
                break
            else:
                print("please shopping")
                user_salary = left_money
                continue
    else:
        print("Not enough money")
        enter_quit1 = input("do you want to eixt? y/n:")
        #询问用户是否想退出程序，如果想就退出，否则继续购物
        if enter_quit1 == 'y':
            print("program exit")
            break
        else:
            print("please shopping")
            continue

print("your commodity is",user_choose)
print("your left money is",left_money_list)









