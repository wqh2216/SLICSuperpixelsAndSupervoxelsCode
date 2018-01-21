#Author:wqh
from PIL import Image
import glob,os
# im = Image.open("C:\\Users\\wqh\\Desktop\\test1\\2.jpg")
for files in glob.glob("C:\\Users\\wqh\\Desktop\\test1*.jpg"):
    filepath, filename = os.path.split(files)
    filterame, exts = os.path.splitext(filename)
    # opfile = r'C:\\Users\\wqh\\Desktop\\test1'
    im = Image.open(files)
    im.show()
