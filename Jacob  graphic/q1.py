from graphics import *
from random import *
def main():
    win = GraphWin('Shapes',500,500)
    position = win.getMouse()
    x =position.getX()
    y =position.getY()
    sq = Rectangle(Point(x-15,y-15),Point(x+15,y+15))
    sq.setFill('green')
    sq.draw(win)

    position = win.getMouse()
    sq.undraw()
    x =position.getX()
    y =position.getY()
    sq = Rectangle(Point(x-15,y-15),Point(x+15,y+15))
    sq.setFill('blue')
    sq.draw(win)

    position = win.getMouse()
    sq.undraw()
    x =position.getX()
    y =position.getY()
    sq = Rectangle(Point(x-15,y-15),Point(x+15,y+15))
    sq.setFill('green')
    sq.draw(win)

    position = win.getMouse()
    sq.undraw()
    x =position.getX()
    y =position.getY()
    sq = Rectangle(Point(x-15,y-15),Point(x+15,y+15))
    sq.setFill('green')
    sq.draw(win)

    position = win.getMouse()
    sq.undraw()
    x =position.getX()
    y =position.getY()
    sq = Rectangle(Point(x-15,y-15),Point(x+15,y+15))
    sq.setFill('green')
    sq.draw(win)


    # print(x)
    # print(y)
    # win.getMouse()
    win.getMouse()
main()