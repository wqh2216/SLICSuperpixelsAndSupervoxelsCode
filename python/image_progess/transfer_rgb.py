#Author:wqh
#对于rgb图像进行灰度化
from PIL import Image
I = Image.open('C:\\Users\\wqh\\Desktop\\gt\\sam.jpg')
I.show()
L = I.convert('L')
L.show()