def col(n):
    if n%2 == 0 :
        return float(n/2)
    else :
        return float((3*n)+1)

iterations = 21

n = int(input("Enter a starting number: "))

print("The Collatz sequence is: ",end="")

for i in range(iterations) :
    print(n,end="")
    if i < iterations :
        print(", ", end="")
    n = col(n)
print("...")