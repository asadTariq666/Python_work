class Stock:
    def __init__(self, symbol, name, previous_price, current_price):
        self.__symbol = symbol
        self.__name = name
        self.__previous_price = previous_price
        self.__current_price = current_price

    #getters
    def get_symbol(self):
        return self.__symbol

    def get_name(self):
        return self.__name

    def get_previous_price(self):
        return self.__previous_price

    def get_current_price(self):
        return self.__current_price
    
    #setters
    def set_symbol(self, symbol):
        self.__symbol = symbol

    def set_name(self, name):
        self.__name = name

    def set_previous_price(self, previous_price):
        self.__previous_price = previous_price

    def set_current_price(self, current_price):
        self.__current_price = current_price

    #constructor
    def getChangePercent(self):
        return ((self.__current_price - self.__previous_price) / self.get_previous_price()*100)
