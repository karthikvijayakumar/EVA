# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1273')
Config.set('graphics', 'height', '1049')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(10,3,0.9)
action2rotation = [10,0,-10]
last_reward = 0
scores = []
im = CoreImage("./images/lalbagh_mask.png")

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/lalbagh_mask_rot_90.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1096
    goal_y = 1049-558
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    sensor4_x = NumericProperty(0)
    sensor4_y = NumericProperty(0)
    sensor4 = ReferenceListProperty(sensor4_x, sensor4_y)
    sensor5_x = NumericProperty(0)
    sensor5_y = NumericProperty(0)
    sensor5 = ReferenceListProperty(sensor5_x, sensor5_y)
    sensor6_x = NumericProperty(0)
    sensor6_y = NumericProperty(0)
    sensor6 = ReferenceListProperty(sensor6_x, sensor6_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    signal4 = NumericProperty(0)
    signal5 = NumericProperty(0)
    signal6 = NumericProperty(0)
    signal7 = NumericProperty(0)
    signal8 = NumericProperty(0)
    signal9 = NumericProperty(0)

    # size -> width of the square
    def sensor_value(self, center_x, center_y, size_x, size_y):
        global longueur
        global largeur
        ret_val = 1
        
        if( center_x+size_x>longueur-50 or center_x-size_x<50 or center_y+size_y>largeur-50 or center_y-size_y<50):
            ret_val = size_y*size_y
        else:
            ret_val = np.nanmean( sand[ int(center_x) - int(size_x) : int(center_x) + int(size_x), int(center_y) - int(size_y) : int(center_y) + int(size_y)] )*100
            ret_val = 1 if np.isnan(ret_val) else ret_val
        
        return ret_val

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(10, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(10, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(10, 0).rotate((self.angle-30)%360) + self.pos
        self.sensor4 = Vector(20, 0).rotate(self.angle) + self.pos
        self.sensor5 = Vector(20, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor6 = Vector(20, 0).rotate((self.angle - 30) % 360) + self.pos
        self.sensor7 = Vector(40, 0).rotate(self.angle) + self.pos
        self.sensor8 = Vector(40, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor9 = Vector(40, 0).rotate((self.angle - 30) % 360) + self.pos
        self.signal1 = int(self.sensor_value(self.sensor1_x, self.sensor1_y, 5, 5))
        self.signal2 = int(self.sensor_value(self.sensor2_x, self.sensor2_y, 5, 5))
        self.signal3 = int(self.sensor_value(self.sensor3_x, self.sensor3_y, 5, 5))
        self.signal4 = int(self.sensor_value(self.sensor4_x, self.sensor4_y, 10, 10))
        self.signal5 = int(self.sensor_value(self.sensor5_x, self.sensor5_y, 10, 10))
        self.signal6 = int(self.sensor_value(self.sensor6_x, self.sensor6_y, 10, 10))
        self.signal7 = int(self.sensor_value(self.sensor4_x, self.sensor4_y, 20, 20))
        self.signal8 = int(self.sensor_value(self.sensor5_x, self.sensor5_y, 20, 20))
        self.signal9 = int(self.sensor_value(self.sensor6_x, self.sensor6_y, 20, 20))

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Ball4(Widget):
    pass
class Ball5(Widget):
    pass
class Ball6(Widget):
    pass
class Ball7(Widget):
    pass
class Ball8(Widget):
    pass
class Ball9(Widget):
    pass


class Goal(Widget):
    pass


# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    ball4 = ObjectProperty(None)
    ball5 = ObjectProperty(None)
    ball6 = ObjectProperty(None)
    ball7 = ObjectProperty(None)
    ball8 = ObjectProperty(None)
    ball9 = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(2, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, self.car.signal4, self.car.signal5, self.car.signal6, self.car.signal7, self.car.signal8, self.car.signal9, orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        self.ball4.pos = self.car.sensor4
        self.ball5.pos = self.car.sensor5
        self.ball6.pos = self.car.sensor6
        self.ball7.pos = self.car.sensor7
        self.ball8.pos = self.car.sensor8
        self.ball9.pos = self.car.sensor9
        self.goal.pos = Vector(goal_x, goal_y)

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            last_reward = -100 * ((distance+100)/1650) # 1650 is the length of the diagonal of the image
            if last_distance < distance:
                last_reward = -120 * (distance/1650)
            print( 1, goal_x, goal_y, distance, int(self.car.x), int(self.car.y), last_signal, last_reward, last_distance)
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -10.2 * (distance/1650)
            if distance < last_distance:
                last_reward = 0.2 * ((1650-distance)/1650)
            print(0, goal_x, goal_y, distance, int(self.car.x), int(self.car.y), last_signal, last_reward, last_distance)
        if self.car.x < 50:
            self.car.x = 50
            last_reward = -500
        if self.car.x > self.width - 50:
            self.car.x = self.width - 50
            last_reward = -500
        if self.car.y < 50:
            self.car.y = 50
            last_reward = -500
        if self.car.y > self.height - 50:
            self.car.y = self.height - 50
            last_reward = -500

        if distance < 25:
            last_reward = 100
            if swap == 0:
                goal_x = 501
                goal_y = 1049 - 331
                swap = 1
            elif swap == 1:
                goal_x = 245
                goal_y = 1049-732
                swap = 0
            else:
                goal_x = 1096
                goal_y = 1049-558                
                swap = 2

        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        largeur = 1273
        longueur = 1049

        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        print(parent.width)
        clearbtn = Button(text = 'clear', pos = (longueur - parent.width,0))
        savebtn = Button(text = 'save', pos = (longueur - (2*parent.width), 0))
        loadbtn = Button(text = 'load', pos = (longueur - (3 * parent.width), 0) )
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
