#Author:wqh
salary = input('input your salary:')#让用户输入自己的工资
#判断用户的输入是否为数字，如果输入的是数字进行下一步，这个地方是我写程序的时候没有想到的
if salary.isdigit:
    salary = int(salary)
else:
    exit('salary is not digit!!')
#定义一个变量，欢迎用户进入
welcome_msg = 'welcome to our shoping mall'
print(welcome_msg.center(50,'-'))  #.center（距离，字符）还挺好玩

#定义一个列表，列表中套元组
product_list = [
    ('Iphone',5888),
    ('Mac Air',8000),
    ('XiaoMi',19.9),
    ('coffee',30),
    ('Tesla',820000),
    ('Bike',700),
    ('Cloth',200)
]
#定义购物车一个空列表
shop_car = []

#标志位
exit_flag = 0#定义标志位的意义是要进入循环
while exit_flag is not True:
    print('product list :'.center(50,'-'))
    #实现将物品更好的展现出来，进行一个循环
    for item in enumerate(product_list):  #实现更加优美的索引
        #print(item) item实际上是(0, ('Iphone', 5888))这个模样了，下面把它拆分了
        index = item[0]     #获得商品序号
        p_name = item[1][0]   #获得商品名称
        p_price= item[1][1]   #获得商品价格
        print(index,p_name,p_price)
    user_choice = input('[q=quit,c=check]  what do you want to buy?:')
    if user_choice.isdigit():
        user_choice = int(user_choice)
        if user_choice < len(product_list):  #输入的商品序号要在范围之内
            p_item = product_list[user_choice] #选择的商品
            if p_item[1] <= salary:  #买得起
                shop_car.append(p_item)  #放入购物车
                salary -= p_item[1]  #减钱
                print('purchased products are:'.center(40, '-'))
                for item in shop_car:
                    print(item)
                print('Your balance is [%s]' % salary)
            else:                   #买不起
                print('Your balance is [%s],cannot buy the product!'%salary)
        else:
            print('we do not have the product,please input again!')
    elif user_choice == 'q' or user_choice == 'quit':
        print('purchased products are:'.center(40,'-'))
        for item in shop_car:
            print(item)
        print('END '.center(40,'-'))
        print('Your balance is [%s]'%salary)
        exit_flag =True
    elif user_choice == 'c' or user_choice == 'check':
        print('purchased products are:'.center(40, '-'))
        for item in shop_car:
            print(item)
        print('Your balance is %d' % salary)
        print('Check '.center(40, '-'))
    else:
        print('Incorrect input ,please input again!!')