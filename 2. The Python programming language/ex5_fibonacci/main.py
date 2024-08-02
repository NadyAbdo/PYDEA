def fibonacci(index) :
    if index <= 0 :
        return 0
    elif index == 1 :
        return 1
    else :
        return fibonacci(index - 1) + fibonacci(index - 2)


index = int(input("Please enter an index: "))

result = fibonacci(index)
print("The Fibonacci number with index",index,"is",result,".")