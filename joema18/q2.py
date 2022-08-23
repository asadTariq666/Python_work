from GreatestCommonDivider import *
numbers = [8,16,24,36]
  
num1=numbers[0]
num2=numbers[1]
gcd=calculate_gcd(num1,num2)
  
if len(numbers)<=2:
    gcd_of_input=calculate_gcd(numbers[0],numbers[1])
else:
    for i in range(2,len(numbers)):
        gcd_of_input=calculate_gcd(gcd,numbers[i])
      
print(gcd_of_input)