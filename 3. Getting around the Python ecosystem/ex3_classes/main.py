total_employees = 0

class Employee:

    def __init__(self, name, salary, age):

        # Initialize the instance variables
        self.name = name
        self.salary = salary
        self.age = age

        # Increment the total number of employees when a new object is create
        Employee.count()

    def count() :

        # Fuction to increment the total number of employees when a new object is create
        global total_employees
        total_employees += 1

    def display_employee_info(self):
        
        # Display the attributes of the employee
        print("Name:",self.name,", Salary:",self.salary,", Age:",self.age)

# Creating employee objects
employee1 = Employee("John", 2000, 28)
employee2 = Employee("Stuart", 3000, 32)
employee3 = Employee("Jane", 4000, 37)

# Display the information for each employee
employee1.display_employee_info()
employee2.display_employee_info()
employee3.display_employee_info()

# Display the total number of employees
print("Total Employee:",total_employees)