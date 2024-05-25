from pylab import *
import sys

content = open(sys.argv[1],'r').readlines()

X = []
Y = []
Z = []

for line in content:
	line = line.rstrip()
	if line=="":continue
	x,y,z = line.split(" ")
	x = float(x)
	y = float(y)
	z = float(z)
	X.append(x)
	Y.append(y)
	Z.append(z)
plot(X,Y,'ro')
plot(X,Z,'go')
legend()
show()
