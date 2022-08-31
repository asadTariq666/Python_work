# GreatestCommonDivisor.py

def calculate_gcd(num1,num2):
    while(num2):
        num1, num2 = num2, num1 % num2
