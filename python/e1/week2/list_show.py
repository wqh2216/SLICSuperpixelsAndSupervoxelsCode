#Author:wqh
product = [('iphone',1000),('xiaomi',2000),('meizu',3000),('other',10000)]
flag = 0
while flag is not True:
    for item in enumerate(product,start = 1):
        index = item[0]
        p_name = item[1][0]
        p_price = item[1][1]
        print(index,p_name,p_price)
    break



