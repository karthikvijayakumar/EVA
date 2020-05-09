# Code base

There are 3 things at a high level:

1. Python files
    I initially wrote the code into multiple files and ran them locally for easier debugging. For running locally this would still be the preferred approach

    Ensure that all the python files are in the same folder and the images folder is present there too.

    1. custom_gym_env.py

        Defines the custom gym environment designed to simulate the track
    2. models.py

        Defines the T3D class and network classes for Actor and Critic networks
    3. model_helpers.py
        
        Defines the replay buffer to store and sample experiences from and the policy evaluation function
    4. main.py

        Main script that runs the training and inference

        Just run 

            python main.py

        Make sure you have the images folder copied in the same directory as the main.py script.
    
    5. model_inference.py

        Script to run inference from saved model and record video of the runs

2. End game.ipynb

    Contains all of the above code copy pasted into a single notebook for convenient running on colab.

    Ensure you have copied the MASK1.png and car_upright.png images to the working directory in colab

3. images

    Contains two images - MASK1.png and car_upright.png which are required by the Gym environment

    The remaining images are for illustrations in this document    

4. requirements.txt

    Python packages required for recreating the environment using pip

# High level components:

1. Custom Gym environment
2. TD3 model
3. Training and inference process

# Custom Gym environment

I created a simple bare bones gym environment which simulates the track for assignment 7

In general a gym interface should define the following 5 apis ( [Ref](https://github.com/openai/gym/blob/f2c9793eb762c26bb0b5524f5f69cdd536688f2e/gym/core.py#L8) )

1. step

    Given an action, return the next state of the system, reward for the action, is the episode done and any extra info
2. reset

    Reset the state of the environment to an initial/random state
3. render

    How a human would see the environment or how to record it etc.

4. close

    Close the environment. Close any open file handles etc.

5. seed

    Set random seed of the environment


and set the following attributes:

1. action_space
2. observation_space
3. reward_range

I have so far only defined step,reset,render, action_space and observation_space

## Initialisation of the environment

The environment is initialised with the following params:

1. city map image
2. road mask image
3. car image
4. rendering point of view

    This takes two values - map(default) and car

    This represents at what point of view the video recorded is
    In case of map(default), the video would show the whole map with the car overlayed on it
    In case of car, the video would be a first person view of what the agent sees ( the square crop around the car )

Max steps in the environment is set to 2500. The reasoning is as follows:

- The length of the diagonal of the image is 1650
- Our map doesnt have a lot of non-convex regions, hence the agent shouldnt have to cover more than 1650 pixels to reach the target
- The car moves atleast one pixel per timestep ( assuming its not hitting the wall )
- With one pixel movement per step and giving a little bit of room, the agent should be able to reach the goal in 2500 steps ( Currently assuming only one goal. Will upgrade to multiple goals later )


## Definition of state

The state of the env for the external world comprises of 3 parts:
    1. pixels around the car
    2. orientation of the car towards the goal
    3. distance from the goal

### Pixels around the car

Say the car is at (x,y), then we want to capture a fixed size square around the car to give as input to the agent. In some sense this is what the car/agent sees.

The one caveat here is that the car can be at an angle( with respect to the vertical ). Depending on the current angle of the car the action required of the agent can be different at the exact same spot in the map. Hence I chose to rotate the crop of the map by the angle of the car to ensure with the respect to the car the top of the image is always where it's front is pointing to.

#### Calculation


*Notation*

- x,y - Position of the car
- theta - Angle of the car wrt vertical
- d - length of side of square around the car for agent

Say car position is x,y and is at angle theta wrt to the vertical, then what we want is the following crop

![Diagram 1](/Phase%202%20Session%2010/images/diagram_1.png)

We cant however get a rotated crop, hence we take a larger square crop without rotation first, rotate the crop and then crop it further to the size required

![Diagram 2](/Phase%202%20Session%2010/images/diagram_2.png)

Lets call the larger square crop "rough cut". Rough cut is largest when the car is at an angle of 45 degrees to the vertical. To simplify code and calculation we take the largest rough cut that can accommodate all angles of the car.

What is the max size of "rough cut"?

Say the car is at 45 degrees ( theta = 45 ) then the side of the rough cut square is d*sqrt(2)

![Diagram 3](/Phase%202%20Session%2010/images/diagram_3.jpg)

However the rough cut can go outside the bounds of the image itself. Hence we pad the image with d/sqrt(2) pixels of white on all sides. We maintain this padded image for calculation purposes in the environment object.

The code in the _get_current_screen function uses the padded image and corrects for the padding in the coordinate system

I will write up the exact math in this doc at a later point of time.

#### How to obtain the screen grab for the agent

1. Obtain rough cut - Square of side length d*sqrt(2) centered around the car
2. Rotate the rough cut by the car angle - This ensures what the car sees is towards the top of the image
3. Crop a square of side length d around the center
4. Crop a square of side length d/2 in the upper middle - This is what would be in front of the car

Step 1
![Diagram 4](/Phase%202%20Session%2010/images/diagram_4.jpg)

Step 2 and 3
![Diagram 5](/Phase%202%20Session%2010/images/diagram_5.jpg)

Step 4
![Diagram 5_2](/Phase%202%20Session%2010/images/diagram_5_2.jpg)

### Orientation of the goal wrt to the car

This represents at the how many degrees wrt the axes of the car is the goal at.

Say one drew a line between the car and the goal, it would make an angle wrt to the horizontal, lets call this psi

psi+90 would be the angle this line would make to the vertical.
theta is the angle of the car wrt to the vertical

Hence psi+90-theta is the angle of the goal wrt to the axes of the car

![Diagram 6](/Phase%202%20Session%2010/images/diagram_6.jpg)

### Distance to the goal

Fairly self explanatory. At each point the euclidean distance between the car and the goal is fed to the agent

## Action space

I have used an action space of 1. The only action the agent can take is to turn the car by +/- 5 degrees at any given point.

Making the network predict a large range of numbers seemed unnecessarily complex. Also in mujoco environments like AntWalker I noticed they restrict action to -1,1 and internally map it to degrees of rotation.

Hence I limited the action space to -1,1 and converted it +/- 5 degrees within the step function

Note: Since I use radians the max angle of turn would be pi/36 radians ( pi radians = 180 degrees, 180/36 degrees = 5 degrees )

# T3D model

## Network architecture

- I have used a simple average pooling layer for the image
- We obtain a 20x20 image from the environment. Average pooling with a 4x4 kernel with stride 4 gives us 25 values
- Flatten the output from the avg pooling and combine with orientation and distance to goal
- Pass these values through a FC network with 2 layers and one output layer

Currently the network seems to use a large number of parameters for what seems to be a simple task.
In the future I plan to experiment with smaller networks

# Training and inference process

After every call to the train function in the T3D class, I print out how much the weights have changed in the first 2 conv layers and in the head FC layer. This gives us a sense if the backprop is effective and if the model is learning anything new at all.

In the inference process I create two sets of videos with different point of view of recording. One records the whole map with the car overlayed on it, the second records what the car sees ( only the screen capture part, orientation and distance are not rendered )

There are 2 things to note during the training process:

1. Epsilon greedy
    To increase the exploration done by the agent I implemented epsilon greedy
    In the training loop the agent uses random actions for the first 10k iterations
    Post that it uses epsilon greedy to randomly choose actions with epsilon probability
    epsilon starts out at 1 after 10k timesteps and eventually converges to 0.05 ( 5% )
    
    The graph of epsilon over timesteps is below
    ![Diagram epsilon greedy](/Phase%202%20Session%2010/images/diagram_epsilon_greedy.png)

2. LR scheduler
    I noticed that the change to the networks was of constant magnitude after a while
    I believe this is because the optimiser is moving back and forth around a minima
    Taking cue from this I put in a LR scheduler to drop the LR by half after every certain number of iterations
    The halving point is 5k for the critic and 2.5k for the actor. Both their LRs get halved at the same time