#Author:wqh
#定义一个商品的列表
goods = [('iphone',8000),
         ('imac book',10000),
         ('python book',100),
         ('tv',5000),
         ('others',1000)]
#定义一个空列表将用户购买的商品添加进去
user_choose = []
left_money_list = []
#询问用户是否想购买商品
enter_entry = input("do you want to buy somethings? y/n:")
if enter_entry == 'y':
    # 输入用户的工资情况,
    user_salary = int(input("please input your salary:"))  # 输入必须是数字，否则会报错
    flag = 0
    # 进入循环判断用户购买与
    while flag is not True:
        print("--------------------------------------------------")
        # 打印商品列表
        for item in enumerate(goods, start=1):
            index = item[0]
            g_name = item[1][0]
            g_price = item[1][1]
            print(index, g_name, g_price)
        print("--------------------------------------------------")
        user_choice = int(input("please input your choose(1-5):")) - 1
        print("--------------------------------------------------")
        # 将商品的列表形式转化为元组形式,以便下面直接使用元组中的元素
        items = goods[user_choice]
        # 判断用户的输入的工资与商品的价钱关系
        if user_salary >= items[1]:
            print("added success!")
            # 将用户所选商品添加至用户空列表中
            user_choose.append(items[0])
            # 计算用户所剩的工资钱数
            left_money = user_salary - items[1]
            # 将用户所剩钱数添加至用户剩余钱数空列表中
            left_money_list.append(left_money)
            print("your items is", user_choose[user_choice], ",your left money is", left_money_list[user_choice])
            # 询问用户是否想继续购物，然后进行判断
            enter_continue = input("do you want to continue? y/n:")
            if enter_continue == 'y':
                print("please shopping")
                # 将用户剩余的钱数赋值给用户的原来输入，继续进行循环
                user_salary = left_money
                continue
            if enter_continue == 'n':
                print("program exit")
                break
            else:
                print("please input correct words")
        else:
            print("you have no enough money!")
            break
    print("your items is ", user_choose, ",your left money is", left_money_list[user_choice])
if enter_entry == 'n':
    print("thanks for coming!see you soon!")













