# Defining the function
def calculate_cost(cost,discount): 
    #calculating cost after discount

    price = cost * (100-discount)/100
    #printing volume of sphere
    print('price of book with initial cost: ',cost,' and discount: ',discount,"% :",price) 


# Function calls with 3 different costs and discounts.
calculate_cost(100,10) # cost = 100, discount = 10%
calculate_cost(350,20) # cost = 350, discount = 20%
calculate_cost(24.95,40) # cost = 24.95, discount = 40%