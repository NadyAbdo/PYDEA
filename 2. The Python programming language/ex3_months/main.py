months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month = input("Please enter a month: ")

if month in months :
    match month :
        case 'January' :
            print(month,"has 31 days.")
        case 'February' :
            print(month,"has 28 or 29 days.")
        case 'March' :
            print(month,"has 31 days.")
        case 'April' :
            print(month,"has 30 days.")
        case 'May' :
            print(month,"has 31 days.")
        case 'June' :
            print(month,"has 30 days.")
        case 'July' :
            print(month,"has 31 days.")
        case 'August' :
            print(month,"has 31 days.")
        case 'September' :
            print(month,"has 30 days.")
        case 'October' :
            print(month,"has 31 days.")
        case 'November' :
            print(month,"has 30 days.")
        case 'December' :
            print(month,"has 31 days.")
            
else :
    print("Month not valid.")