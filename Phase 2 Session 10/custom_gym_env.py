import gym
from gym import error, spaces, utils, wrappers
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from gym.envs.classic_control import rendering
from PIL import Image, ImageOps, ImageDraw
import time
import os

class CityMap(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-float('inf'), float('inf'))
    spec = None
    
    observation_window_size = 80
    # observation_window_size is the side length of the square surrounding the car
    # The car would see (observation_window_size/2) ahead and behind and (observation_window_size/2) to the left and right
    
    max_turn_radians = np.pi/6.0
    # pi/6 radians = 180/6 = 30 degrees
    # The car can turn 30 degrees to the left or right    
    
    distance_threshold_done = 30
    # Distance to target to reach before considering the episode done
    
    goal_circle_radius = int(distance_threshold_done/2)

    #Max steps before we call the episode done
    max_episode_steps = 1000
    
    def __init__(self, citymap, roadmask, car_image):
        self.action_space = spaces.Box(low = np.float64(-self.max_turn_radians), high = np.float64(self.max_turn_radians), shape = (1,) ) 
        # Action space is how many degrees to turn the car in radians
        
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (self.observation_window_size, self.observation_window_size) ) 
        # All combinations of white and black pixels of observation_window_size x observation_window_size
        
        self.state = None
        
        self.citymap = citymap.copy()
        self.roadmask = roadmask.copy()
        self.car_image = car_image.copy()
        
        #Find size of the roadmask for reference later
        self.roadmask_size_x, self.roadmask_size_y = self.roadmask.getbbox()[2:4]
        
        # Pad the road mask image to allow for rotations
        # Amount of padding required = ( diagonal length of the observation window )/2
        self.padding_size = int(self.observation_window_size/np.sqrt(2))
        padding = ( self.padding_size, self.padding_size, self.padding_size, self.padding_size )
        self.roadmaskpadded = ImageOps.expand( self.roadmask, padding, fill = 255 ) # Pad and fill with sand
        
        #Set goal point
        self.goal_x = 1154
        self.goal_y = 158
        
        self.car_pos_x = 0
        self.car_pos_y = 0
        
        self.reset()

    """
        Parameters:
        
        Returns:
            ( next_state, reward, done, info )
        
    """
    def step(self, action):

        # Things to compute
        # 1. Next position
        # 2. Screen grab from next position ( Next state )
        # 3. Reward on moving to next position
        # 4. Is the episode done
        # 5. Any info to pass on to the agent


        # 1. Next position
        # From (pos_x, pos_y) we move forward with 'speed' steps in the direction 'angle+action'
        # New angle of car
        self.car_angle = self.car_angle + action
        if(self.car_angle < 0):
            self.car_angle = (2*np.pi) + self.car_angle
        elif(self.car_angle > (2*np.pi)):
            self.car_angle = self.car_angle - (2*np.pi)
        
        # Car speed depends on whether we are riding on sand or not
        speed = 5 if self.roadmask.getpixel(( self.car_pos_x, self.car_pos_y )) == 0 else 2
        
        displacement_x = speed * np.sin( self.car_angle )
        displacement_y = -1 * speed * np.cos( self.car_angle )
        # Displacement y is negative since the top of the frame is y=0
        # Hence if the car is pointing upwards ( oriented at 0 degrees ) then the y values would decrease
        
        self.car_pos_x = self.car_pos_x + displacement_x
        self.car_pos_y = self.car_pos_y + displacement_y
        
        # Clip position to boundaries of the image
        self.car_pos_x = np.clip(self.car_pos_x, 0, self.roadmask_size_x-1) 
        self.car_pos_y = np.clip(self.car_pos_y, 0, self.roadmask_size_y-1)
        
        # 2. Screen grab from next position ( Next state )
        next_state = self._extract_current_frame()
        
        # 3. Reward on moving to next position
        
        new_distance_from_goal = np.sqrt( (self.car_pos_x - self.goal_x)**2 + (self.car_pos_y - self.goal_y)**2 )
        
        pixel_value_at_car_pos = self.roadmask.getpixel((self.car_pos_x, self.car_pos_y))
#         assert pixel_value_at_car_pos in [0,1], "Pixel values are not exactly 0 or 1")
        
        if( pixel_value_at_car_pos == 1 ):
            #Currently on sand
            reward = -1
        elif( new_distance_from_goal < self.distance_from_goal ):
            reward = -0.2
        else:
            reward = 0.1
        
        self.distance_from_goal = new_distance_from_goal
        
        # 4. Is the episode done?
        
        if(new_distance_from_goal < self.distance_threshold_done):
            done = 1
            self.reset()
            next_state = np.zeros( self.observation_window_size**2 ).reshape((self.observation_window_size,self.observation_window_size) )
        else:
            done = 0
        
        # 5. Any info to pass on to the agent

        print(
            "x: "+ str(self.car_pos_x) + 
            "; y: " + str(self.car_pos_y) + 
            "; angle(deg): " + str(self.car_angle*180/np.pi) + 
            "; action: " + str(action*180/np.pi) +
            "; reward: " + str(reward)
            )
        
        return next_state, reward, done, {}

    """
        Extracts the frame that the agent/car currently sees
        With respect to the frame extracted the car is always pointing upward
        Keeping the orientation fixed is key since else for the same scene( screen grab ), the car can be in different orientations 
        and hence should take different actions
        
        For example take the following case 
            Environment: A single straight road with the car in the middle on the road
            Goal: Left end of the road and outside the visibility of the agent
        
            The car can be oriented left or right and the goal can be to the left or right
            <<< NEED TO THINK OF A BETTER EXAMPLE >>>
            
        Parameters:
            None
        
        Returns:
            img - Numpy array of shape ( observation_window_size, observation_window_size )
    """
    def _extract_current_frame(self):
        # We know the current position of the car
        # Step 1: Extract a square of size observation_window_size*sqrt(2) surrounding the car ( Call this rough cut )
        # Step 2: Rotate the rough cut image around the center by angle of the car
        # Step 3: Extract a square of size observation_window_size around the center
        
        
        # Step 1: Extract a square of size observation_window_size*sqrt(2) surrounding the car ( Call this rough cut )
        # We need to use the padded version of the road mask here
        # Hence we add self.padding_size to the x,y position of the car
        bounding_box_rough_cut = ( self.car_pos_x, self.car_pos_y, self.car_pos_x+(2*self.padding_size), self.car_pos_y+(2*self.padding_size) )
        # print("Bounding box of rough cut: " +str(bounding_box_rough_cut))

        rough_cut = self.roadmaskpadded.crop(bounding_box_rough_cut)
        
        # Step 2: Rotate the rough cut image around the center by angle of the car
        
        rough_cut_rotated = rough_cut.rotate( self.car_angle * (180/np.pi) )
        # PIL's rotate function:
        #  - takes input in degrees ( 180 degrees = pi radians; x radians = x*(180/pi) degrees )
        #  - by default rotates around the center of the image
        #  - rotates anti-clockwise
        
        # Step 3: Extract a square of size observation_window_size around the center
        # Center of the rough cut image is ( rough_cut_size/2, rough_cut_size/2 )
        
        bounding_box_current_frame = ( 
            self.padding_size - (self.observation_window_size/2), 
            self.padding_size - (self.observation_window_size/2), 
            self.padding_size + (self.observation_window_size/2), 
            self.padding_size + (self.observation_window_size/2)
        )
        
        current_frame = rough_cut_rotated.crop(bounding_box_current_frame)

        return np.expand_dims( np.expand_dims( np.asarray(current_frame)/255, axis = 0 ), axis = 0 )
    
    def reset(self):
        #Randomly initialise the starting position and set velocity
        self.car_pos_x = np.random.randint( 0, self.roadmask_size_x )
        self.car_pos_y = np.random.randint( 0, self.roadmask_size_y )
        # Car position is measured with respect to the road mask ( without padding ). (0,0) is top left
        self.car_angle = np.random.default_rng().random() * np.pi * 2.0
        # Initial angle ranges from 0 to 2*pi
        # Angle measures rotation from vertical axis (i.e angle = 0 when car is heading upwards in the map)
        
        #Distance from goal
        self.distance_from_goal = np.sqrt( (self.car_pos_x - self.goal_x)**2 + (self.car_pos_y - self.goal_y)**2 )
        
        return self._extract_current_frame()


    def render(self, mode='human', close=False):
        self.viewer = rendering.SimpleImageViewer()
        #Build image of map with goal and car overlaid
        
        #Create a copy of the map
        map_copy = self.citymap.copy()
        
        #Draw a circle over the goal
        draw = ImageDraw.Draw(map_copy)
        draw.ellipse( 
            (self.goal_x - self.goal_circle_radius, 
             self.goal_y-self.goal_circle_radius, 
             self.goal_x+self.goal_circle_radius, 
             self.goal_y+self.goal_circle_radius
            ), 
            fill = 'red', 
            outline = 'red', 
            width = 1 
        )
        del(draw)
        
        # Create a copy of the car and rotate it to the currrent orientation according to the env state
        # Using 90 - curr_angle since the car image oriented horizontally while our angles are from the vertical
        car_image_copy = self.car_image.copy().rotate( 360 - (self.car_angle*180/np.pi), expand = True )
        car_size_x, car_size_y = car_image_copy.getbbox()[2:4] # The last 2 coordinates represent the size of the car
        
        #Overlay the car on the map ( copy )
        map_copy.paste( car_image_copy, box = ( int(self.car_pos_x - (car_size_x/2)), int(self.car_pos_y - (car_size_y/2)) ) )
        del(car_image_copy)
        del(car_size_x)
        del(car_size_y)        
        
        current_frame = Image.fromarray( self._extract_current_frame().squeeze(0).squeeze(0)*255 ).convert('RGB')
        
        if mode == 'rgb_array':
            return np.asarray(current_frame)
            # return np.asarray(map_copy)
#         elif mode == 'human':
#             if self.viewer is None:
#                 self.viewer = rendering.SimpleImageViewer()
# #             self.viewer.imshow(np.asarray(current_frame)*255)
#             self.viewer.imshow(np.asarray(map_copy))
#             return self.viewer.isopen
    
    def close(self):
        pass
        # if self.viewer is not None:
        #     self.viewer.close()
        #     print(self.viewer.isopen)
        #     self.viewer = None