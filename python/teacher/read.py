f = open("F:\\code\\python\\teacher\\profiles.txt")
line = f.readline()
while line:
   print(line)
   line = f.readline()
f.close()
