operators = ['+','-','*','/']
num_1 = int(input("Please enter a first number: "))
num_2 = int(input("Please enter a second number: "))
operation = input("Please enter an operation: ")

if operation in operators :
    if operation == '+' :
        result = float(num_1+num_2)
    if operation == '-' :
        result = float(num_1-num_2)
    if operation == '*' :
        result = float(num_1*num_2)
    if operation == '/' :
        result = float(num_1/num_2)
    print(num_1,operation,num_2,"=",result)
    
else :
    print("NOT A VALID OPERATOR!")
