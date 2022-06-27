# Defining the function
def print_volume(r): 
    # initializing value of pi
    pi = 3.1415926535897932
    #calculating volume of sphere
    volume = 4/3 * pi * r**3
    #printing volume of sphere
    print('Volume of sphere with radius',r,'is: ',volume) 

# Function calls with 3 different radius.
print_volume(2) #radius = 2
print_volume(10) #radius = 10
print_volume(15) #radius = 15
 