import random
class SleepingBag:
    def __init__(self, priority):
        self.name = "Sleeping Bag"
        self.priority = int(priority) 
        self.weight = 10
class Rope:
    def __init__(self, priority):
        self.name = "Rope"
        self.priority = int(priority)
        self.weight = 3

class PocketKnife:
    def __init__(self, priority):
        self.name = "Pocket Knife"
        self.priority = int(priority)
        self.weight = 2

class Torch:
    def __init__(self, priority):
        self.name = "Torch"
        self.priority = int(priority)
        self.weight = 5

class WaterBottle:
    def __init__(self, priority):
        self.name = "Water Bottle"
        self.priority = int(priority)
        self.weight = 9

class Glucose:
    def __init__(self, priority):
        self.name = "Glucose"
        self.priority = int(priority)
        self.weight = 8

class FirstAidSupplies:
    def __init__(self, priority):
        self.name = "First Aid Supplies"
        self.priority = int(priority)
        self.weight = 6

class RainJacket:
    def __init__(self, priority):
        self.name = "Rain Jacket"
        self.priority = int(priority)
        self.weight = 3

class PersonalLocatorBeacon:
    def __init__(self, priority):
        self.name = "Personal Locator Beacon"
        self.priority = int(priority)
        self.weight = 2

flag = True
generations = []
All_Items = []


def Create_Initial_Population():

    max_weight = 30 
    global generations
    global All_Items
    global flag 

    if flag: 
        userName = input("Welcome to Hiking! What is your name: ")
        print("Hi "+userName+"! Please rate the items as your nessecity:-")
        SB = SleepingBag(input("Enter the Priority of Sleeping Bag(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(SB)
        R = Rope(input("Enter the Priority of Rope(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(R)
        PK = PocketKnife(input("Enter the Priority of Pocket Knife(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(PK)
        T = Torch(input("Enter the Priority of Torch(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(T)
        WB = WaterBottle(input("Enter the Priority of Water Bottle(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(WB)
        G = Glucose(input("Enter the Priority of Glucose(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(G)
        FAS = FirstAidSupplies(input("Enter the Priority of First Aid Supplies(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(FAS)
        RJ = RainJacket(input("Enter the Priority of Rain Jacket(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(RJ)
        PLB = PersonalLocatorBeacon(input("Enter the Priority of Personal Locator Beacon(15 for High priority, 10 for medium priority, 5 for low priority): "))
        All_Items.append(PLB) 
        flag = False 
    
    weight=0
    All_Items_Copy = All_Items.copy()
    this_generation = []
    while(1):
        x = random.choice(All_Items_Copy) 
        All_Items_Copy.pop( All_Items_Copy.index(x)) 
        if(int(weight+x.weight)<=max_weight): 
                weight = weight+x.weight
                this_generation.append(x)
        if len(All_Items_Copy)<1: 
                break
    efficiency = fitnessFunction(this_generation)
    this_generation.append(int(efficiency))
    generations.append(this_generation)
    return this_generation

def fitnessFunction(this_generation):
    total_priority = 0
    for o in this_generation:
        total_priority = total_priority + o.priority
    return total_priority

Create_Initial_Population()