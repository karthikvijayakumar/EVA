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
    metadata = {'render.modes': ['rgb_array']}
    reward_range = (-float('inf'), float('inf'))
    spec = None
    
    observation_window_size = 40
    # observation_window_size is the side length of the square surrounding the car
    # The car would see (observation_window_size/2) ahead and behind and (observation_window_size/2) to the left and right
    
    max_action = np.float64(1.0)

    max_turn_radians = np.pi/6.0
    # pi/6 radians = 180/6 = 30 degrees
    # The car can turn 30 degrees to the left or right    
    
    distance_threshold_done = 30
    # Distance to target to reach before considering the episode done
    
    goal_circle_radius = int(distance_threshold_done/2)

    #Max steps before we call the episode done
    max_episode_steps = 2500

    #State image size
    state_image_size = 20
    
    def __init__(self, citymap, roadmask, car_image, render_pov = 'map'):
        self.action_space = spaces.Box(low = np.float64(-self.max_action), high = np.float64(self.max_action), shape = (1,) ) 
        # Action space is how many degrees to turn the car in radians
        
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (self.observation_window_size, self.observation_window_size) ) 
        # All combinations of white and black pixels of observation_window_size x observation_window_size
        
        self.state = None
        
        self.citymap = citymap.copy()
        self.roadmask = ImageOps.invert( roadmask.copy() )
        self.car_image = car_image.copy()
        self.render_pov = render_pov
        
        #Find size of the roadmask for reference later
        self.roadmask_size_x, self.roadmask_size_y = self.roadmask.size

        # Find length of diagonal of the road mask image >= Max distance from goal
        self.road_mask_diagonal = np.sqrt( self.roadmask_size_x**2 + self.roadmask_size_y**2 )
        
        # Pad the road mask image to allow for rotations
        # Amount of padding required = ( diagonal length of the observation window )/2
        self.padding_size = int(self.observation_window_size/np.sqrt(2))
        padding = ( self.padding_size, self.padding_size, self.padding_size, self.padding_size )
        self.roadmaskpadded = ImageOps.expand( self.roadmask, padding, fill = 0 ) # Pad and fill with sand
        
        #Set goal point
        self.goal_x = 1154
        self.goal_y = 158
        # self.goal_x = 408
        # self.goal_y = 144
        
        self.car_pos_x = 0
        self.car_pos_y = 0

        #Set number of steps in this episode
        self.num_steps = 0

        # Variable to track split between various steps taken in this episode
        self.steps_split = {
            'road' : 0,
            'road_towards_goal' : 0,
            'road_away_goal' : 0,
            'sand' : 0,
            'sand_towards_goal' : 0,
            'sand_away_goal' : 0
        }
        
        self.reset()

    """
        Parameters:
        
        Returns:
            ( next_state, reward, done, info )
        
    """
    def step(self, action_array):

        # Type check to ensure we get a array of shape 1 where the element is a float32
        assert type(action_array) == np.ndarray, "Input action should be an nd array"
        assert action_array.shape == (1,), "Input action should be of shape (1,)"
        assert type(action_array[0]) == np.float32, "Input action arrays element must be a numpy float32"
        
        action = action_array[0]
        
        # Setting info to a empty dict
        info = {}

        # Things to compute
        # 1. Next position         
        # 2. Reward on moving to next position
        # 3. Update number of steps taken
        # 4. Update steps_split
        # 5. Combine Screen grab from next position and orientation to produce the next state
        # 6. Is the episode done?
        # 7. Any info to pass on to the agent

        # 1. Next position
        # From (pos_x, pos_y) we move forward with 'speed' steps in the direction 'angle+action*max_turn_radians'
        # The action given by the agent is from -1 to 1. The env maps the action to degrees of turn
        # New angle of car
        # Angle of the car lies in [-pi, pi)
        self.car_angle = self.car_angle + (action*self.max_turn_radians)
        if(self.car_angle < -np.pi):
            self.car_angle = self.car_angle + (2*np.pi) 
        elif(self.car_angle >= np.pi):
            self.car_angle = self.car_angle - (2*np.pi)
        
        # Car speed depends on whether we are riding on sand or not
        speed = 5 if self.roadmask.getpixel(( self.car_pos_x, self.car_pos_y )) == 255 else 2
        
        displacement_x = speed * np.sin( self.car_angle )
        displacement_y = -1 * speed * np.cos( self.car_angle )
        # Displacement y is negative since the top of the frame is y=0
        # Hence if the car is pointing upwards ( oriented at 0 degrees ) then the y values would decrease
        
        old_car_pos_x = self.car_pos_x
        old_car_pos_y = self.car_pos_y

        self.car_pos_x = self.car_pos_x + displacement_x
        self.car_pos_y = self.car_pos_y + displacement_y
        
        # Clip position to boundaries of the image
        self.car_pos_x = np.clip(self.car_pos_x, 0, self.roadmask_size_x-1)
        self.car_pos_y = np.clip(self.car_pos_y, 0, self.roadmask_size_y-1)
        
        # 2. Reward on moving to next position
        # Reward is computed as a sum of different components
        # Component 1: Living penalty
        # Component 2: If the car is in sand or not
        # Component 3: Distance covered towards the goal
        # Component 4: Termination conditions - Reached the goal / Not moving
        # Component 5: Cost of turning
        
        new_distance_from_goal = np.sqrt( (self.car_pos_x - self.goal_x)**2 + (self.car_pos_y - self.goal_y)**2 )
        
        pixel_value_at_car_pos = self.roadmask.getpixel((self.car_pos_x, self.car_pos_y))

        reward = 0

        if(pixel_value_at_car_pos == 0):
            #On sand

            if(
                ( (old_car_pos_x-self.car_pos_x) == 0 and (self.car_pos_x == 0 or self.car_pos_x >= self.roadmask_size_x-1) ) or 
                ( (old_car_pos_y-self.car_pos_y) == 0 and (self.car_pos_y == 0 or self.car_pos_y >= self.roadmask_size_y-1) )
            ):
                #Handle boundary cases
                reward = 0.5*np.abs(action)
                # Incentivise large turns when at the boundary
            elif(new_distance_from_goal < self.distance_from_goal):
                #Handle non boundary cases
                reward = 0.1
        else:
            #On road
            if(new_distance_from_goal < self.distance_from_goal):
                reward = 1
            else:
                reward = 0.3
        assert reward <= 1, "Reward for a single step pre-termination bonus is greater than 1. Reward : " +str(reward)
        
        #Component 1: Living penalty
        # reward -= 1

        # Component 2
        # reward += 0.5 if( pixel_value_at_car_pos == 0 ) else 2
        #Currently on sand
        # reward += -1
        #reward = -1 * ((new_distance_from_goal)/self.road_mask_diagonal) # 1650 is the length of the diagonal of the image

        # Component 3
        # reward +=  1 if ( self.distance_from_goal - new_distance_from_goal > 0 ) else -0.5

        # reward = -0.2 * (new_distance_from_goal/self.road_mask_diagonal)
        # reward = 0.2 * (1 - (new_distance_from_goal/self.road_mask_diagonal) )
        
        # Component 4: Reward on termination conditions
        if( new_distance_from_goal < self.distance_threshold_done ):
            # Give high +ve reward when it has reached the goal
            reward += 50
        # elif( self.num_steps == self.max_episode_steps ):
            # Give high -ve reward when the num steps has crossed max steps
            # pass
            # reward = -1000
        elif(
            old_car_pos_x-self.car_pos_x == 0 and old_car_pos_y-self.car_pos_y == 0 or
            ( (old_car_pos_x-self.car_pos_x) == 0 and (self.car_pos_x == 0 or self.car_pos_x >= self.roadmask_size_x-1) ) or 
            ( (old_car_pos_y-self.car_pos_y) == 0 and (self.car_pos_y == 0 or self.car_pos_y >= self.roadmask_size_y-1) ) 
            ):
            # Give high -ve reward when hitting a wall or moving into a corner
            reward -= 20

        # Component 5: Add cost of turning action*max_radians
        # reward += (-1)*np.abs(action)*0.1

        # 3. Update number of steps taken
        self.num_steps += 1
 
        # 4. Update steps_split
        if(pixel_value_at_car_pos == 0):
            self.steps_split['sand'] +=  1
            if( new_distance_from_goal < self.distance_from_goal ):
                self.steps_split['sand_towards_goal'] += 1
            else:
                self.steps_split['sand_away_goal'] += 1
        else:
            self.steps_split['road'] += 1
            if( new_distance_from_goal < self.distance_from_goal ):
                self.steps_split['road_towards_goal'] += 1
            else:
                self.steps_split['road_away_goal'] += 1

        assert self.steps_split['sand']+self.steps_split['road'] == self.num_steps, "Steps splits doesnt match with num steps"

        info.update( self.steps_split )
        
        self.distance_from_goal = new_distance_from_goal

        # 5. Combine screen grab from current position with orientation and distance to goal to form next state
        next_state = ( self._extract_current_frame(), self._compute_orientation_towards_goal()/np.pi , self.distance_from_goal/self.road_mask_diagonal)
        # We scale the orientation and distance by their max values to ensure their absolute values dont cross one

       
        # 6. Is the episode done? and compute info to pass to agent
        
        if( 
            new_distance_from_goal < self.distance_threshold_done  or 
            self.num_steps == self.max_episode_steps or
            ( old_car_pos_x-self.car_pos_x == 0 and old_car_pos_y - self.car_pos_y == 0)
            ):
            # Either we have reached the target position or we have exceed the max steps for this episode or the car is not moving
            done = True

            # Info to pass to agent
            if( new_distance_from_goal < self.distance_threshold_done):
                info['reached_goal'] = True
                info['termination_reason'] = 'reached goal'
            elif( self.num_steps == self.max_episode_steps ):
                info['reached_goal'] = False
                info['termination_reason'] = 'max steps'
            elif( old_car_pos_x-self.car_pos_x == 0 and old_car_pos_y - self.car_pos_y == 0 ):
                info['reached_goal'] = False
                info['termination_reason'] = 'car not moving'
            else:
                info['reached_goal'] = False
                info['termination_reason'] = 'not terminated'

            # self.reset()
            next_state = (self._zero_screen_grab(),0,0)
            # Return a zero screen grab, zero orientation and zero distance in case of termination
            
        else:
            done = False

        # 7. Any info to pass on to the agent
        assert done or reward <= 1, "Reward for a non-terminating step is greater than 1. Reward : " +str(reward)
        return next_state, reward, done, info

    """
        Zero screen grab for episode termination conditions
    """
    def _zero_screen_grab(self):
        screen_grab = np.expand_dims( 
                np.expand_dims( 
                    np.zeros( self.state_image_size **2 ).reshape(( self.state_image_size , self.state_image_size )),
                    axis = 0 
                ),
                axis = 0 )
        return screen_grab

    """
        Definition of orientation:
            With respect to the axes of car ( car's forward pointing upwards ), at how many degrees is the goal
            orientation lies in the range [-pi,pi)

        We compute this in two steps:
        Step 1: At what angle is the goal with respect to the vertical
            Angle of goal wrt horizontal is tan_inverse( distance in y axis / distance in x axis )
            Angle of goal wrt vertical is 90 + the above = 90 + tan_inverse( distance in y axis / distance in x axis )        
        Step 2: Subtract the angle of the car from the above quantity to get angle relative to the car axes
            Angle of goal wrt car = 90 + tan_inverse( distance in y axis / distance in x axis ) - car angle wrt vertical
    """
    def _compute_orientation_towards_goal(self):
        orientation = np.arctan2( self.goal_x - self.car_pos_x, self.car_pos_y - self.goal_y ) - self.car_angle
        # orientation = np.pi/2.0 + np.arctan2( self.goal_y - self.car_pos_y, self.goal_x - self.car_pos_x ) - self.car_angle

        if(orientation >= np.pi):
            orientation = orientation - (2*np.pi)
        elif(orientation < -np.pi):
            orientation = orientation + (2*np.pi)
        
        return orientation
    
    """
        Simple euclidean distance computation
        Abstracted to a function to avoid rewriting in multiple places
    """
    def _compute_distance_from_goal(self):
        return np.sqrt( (self.car_pos_x - self.goal_x)**2 + (self.car_pos_y - self.goal_y)**2 ) 

    """
        Extracts the frame that the agent/car currently sees
        With respect to the frame extracted the car is always pointing upward
        Keeping the orientation fixed is key since else for the same scene( screen grab ), the car can be in different orientations 
        and hence should take different actions
           
        Parameters:
            None
        
        Returns:
            img - Numpy array of shape ( observation_window_size/2, observation_window_size/2 )
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

        # Step 4: Cropping out to only what is visible in front of the car
        bounding_box_forward_view = (
            int(self.observation_window_size/4),
            0,
            int((3*self.observation_window_size)/4),
            int(self.observation_window_size/2)
        )

        forward_view = current_frame.crop(bounding_box_forward_view)
        
        # Scaling down the image to half the dimensions for optimising memory and simplifying input to agent
        forward_view = forward_view.resize((self.state_image_size,self.state_image_size), resample = Image.NEAREST )
        # current_frame = current_frame.resize((int(self.observation_window_size/8), int(self.observation_window_size/8)), resample = Image.NEAREST )

        return np.expand_dims( np.expand_dims( np.asarray(forward_view)/255, axis = 0 ), axis = 0 )
    
    def reset(self):
        #Randomly initialise the starting position and set velocity
        self.car_pos_x = np.random.randint( 0, self.roadmask_size_x )
        self.car_pos_y = np.random.randint( 0, self.roadmask_size_y )
        # self.car_pos_x = 100
        # self.car_pos_y = 445
        # Car position is measured with respect to the road mask ( without padding ). (0,0) is top left
        self.car_angle = np.random.uniform(-1,1) * np.pi
        # self.car_angle = 0
        # Initial angle ranges from 0 to 2*pi
        # Angle measures rotation from vertical axis (i.e angle = 0 when car is heading upwards in the map)
        
        #Distance from goal
        self.distance_from_goal = self._compute_distance_from_goal()

        #Set num_steps to 0
        self.num_steps = 0        

        #Set steps splits to zero
        self.steps_split = {
            'road' : 0,
            'road_towards_goal' : 0,
            'road_away_goal' : 0,
            'sand' : 0,
            'sand_towards_goal' : 0,
            'sand_away_goal' : 0
        }
        
        
        return (self._extract_current_frame(), self._compute_orientation_towards_goal()/np.pi, self.distance_from_goal/self.road_mask_diagonal )
        # We scale the orientation and distance by their max values to ensure their absolute values dont cross one


    def render(self, mode='rgb_array', close=False):        
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
        car_image_copy = self.car_image.copy().rotate( 360 - (self.car_angle*180/np.pi), expand = True )
        car_size_x, car_size_y = car_image_copy.getbbox()[2:4] # The last 2 coordinates represent the size of the car
        
        #Overlay the car on the map ( copy )
        map_copy.paste( car_image_copy, box = ( int(self.car_pos_x - (car_size_x/2)), int(self.car_pos_y - (car_size_y/2)) ) )
        del(car_image_copy)
        del(car_size_x)
        del(car_size_y)       
                
        if mode == 'rgb_array':
            if(self.render_pov == 'map'):            
                return np.asarray(map_copy)
            elif(self.render_pov == 'car'):
                current_frame = Image.fromarray( self._extract_current_frame().squeeze(0).squeeze(0)*255 ).convert('RGB')
                current_frame = current_frame.resize((self.observation_window_size, self.observation_window_size))
                return np.asarray(current_frame)
    
    def close(self):
        pass