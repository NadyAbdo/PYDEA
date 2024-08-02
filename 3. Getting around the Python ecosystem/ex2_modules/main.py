import file_fcts as op
import math as m

def func1(x,y) :
    return m.pow(op.sum(op.divide(op.multiply(x,y),op.sum(x,y)),op.divide(x,op.multiply(y,op.subtract(x,y)))),2)

def func2(x,y) :
    return m.sqrt(1+func1(x,y))

print("Function 1 output is",func1(6,2))
print("Function 2 output is",func2(6,2))