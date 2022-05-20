from graphics import *
def main():
 win = GraphWin('Tic Tac Toe',1300,1300)
 # set coordinates for window. This helps in identifying the coordinates of the window when placing obj win.setCoords(0,0,12,12)
 # create vertical red lines at 1/3rd and 2/3rd of the window
 l1 = Line(Point(40,0),Point(40,120))
 l1.setFill('red')
 l1.draw(win)
 l2 = Line(Point(80,0),Point(80,120))
 l2.setFill('red')
 l2.draw(win)
 # create horizontal red lines at 1/3rd and 2/3rd of the window
 l3 = Line(Point(0,40),Point(120,40))
 l3.setFill('red')
 l3.draw(win)
 l4 = Line(Point(0,80),Point(120,80))
 l4.setFill('red')
 l4.draw(win)
 # draw a blue circle
 p1 = Point(20,100)
 r = 20
 circ = Circle(p1,r)
 circ.setFill('blue')
 circ.draw(win)
 # draw a green triangle
 tri = Polygon(Point(80,0),Point(120,0),Point(100,40))
 tri.setFill('green')
 tri.draw(win)

 # wait for user to click mouse
 win.getMouse()
 # after user clicks on window, move circle to the right and triangle to the left
 circ.move(40,0)
 tri.move(-40,0)
 # wait for user to click mouse
 win.getMouse()
 # after user clicks, remove the circle
 circ.undraw()
 win.getMouse()
main()