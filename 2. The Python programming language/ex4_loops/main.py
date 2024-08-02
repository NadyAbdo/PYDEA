num_1 = int(input("Please enter a first number:"))
num_2 = int(input("Please enter a second number:"))


if num_1 < num_2 :  
    diff = num_2 - num_1 
    for i in range(diff-1) :
        print(num_1+i+1,end="")
        if (num_1+i+2 < num_2) :
            print(", ",end="")
    print()

elif num_1 > num_2 : 
    diff = num_1 - num_2  
    for i in range(diff-1) :
        print(num_2+i+1,end="")
        if (num_2+i+2 < num_1) :
            print(", ",end="")
    print()

'''
str_ = ''
for i in range(num_1+1, num_2):
    str_ = str_ + str(i) + ','
print(str_)
'''