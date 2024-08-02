temperature = float(input("Please enter a temperature in degree Celsius: "))

if temperature <= 0 :
    print ("Water is solid.")

elif temperature > 0 and temperature <100 :
    print ("Water is liquid.")
    
elif temperature >=100 :
    print("Water is gaseous.")