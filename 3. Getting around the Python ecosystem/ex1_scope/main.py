total = 5; # This is global variable.
# Outer function definition is here
def sum (arg1, arg2):
    # Add both the parameters and return them."
    total = arg1 * arg2; # Here total is outer local variable.
    print ("Inside the outer function local total is:" , total)
    def innersum (Arg1,Arg2):
        Arg1=Arg1*Arg2
        return Arg1
    # Here total is inner local variable.
    print ("Inside the inner function local total is:" , innersum(total,arg1))
    return total
# Now you can call sum function
sum (7,3)
print ("Outside the function global total is : ", total)
print()
total=sum(5,6)
print ("Outside the function global total : ",total)