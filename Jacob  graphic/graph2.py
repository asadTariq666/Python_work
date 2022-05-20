

from graphics import *
from random import *
def main():
 win = GraphWin('Jacobo',400,400)
 position = win.getMouse() 
 x =position.getX()
 y =position.getY()
 point =Point(x,y)
 r=20
 circ = Circle(point,r)
 circ.setFill('green')
 circ.draw(win)

 position = win.getMouse() 
 circ.undraw()
 x =position.getX()
 y =position.getY()
 point =Point(x,y)
 r=50
 circ = Circle(point,r)
 circ.setFill('green')
 circ.draw(win)
 
 position = win.getMouse() 
 circ.undraw()
 x =position.getX()
 y =position.getY()
 point =Point(x,y)
 r=35
 circ = Circle(point,r)
 circ.setFill('blue')
 circ.draw(win)

 position = win.getMouse() 
 circ.undraw()
 x =position.getX()
 y =position.getY()
 point =Point(x,y)
 r=50
 circ = Circle(point,r)
 circ.setFill('blue')
 circ.draw(win)

 position = win.getMouse() 
 circ.undraw()
 x =position.getX()
 y =position.getY()
 point =Point(x,y)
 r=20
 circ = Circle(point,r)
 circ.setFill('green')
 circ.draw(win)

 win.getMouse()
 circ.undraw()

 label = Text(Point(200, 200), 'Press any key to continue')
 label.draw(win)
#  print(x)
#  print(y)
#  win.getMouse()
 win.getMouse()
main()