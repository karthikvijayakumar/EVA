15 steps in the T3D code:

# Step 1
Defining the experience replay buffer

This class needs to have 2 methods:
1. Append

    Given the replay buffer is of fixed size, after the reqd size is reached
2. Sample

![Step 1](https://drive.google.com/uc?export=view&id=1BaCmiR6zl23zcjlUJB24uIWxCWTZz5Sp)

# Step 2
Define the class for the Actor and Actor Target
    Define a simple DNN with 2 hidden layers.
    Both the Actor and Actor Target have the same DNN structure and hence dont require different classes. 

![Step 2](https://drive.google.com/uc?export=view&id=1TFKY4jzUnKywxD7_PUVP3YCwx99bVdTZ)

# Step 3
Define the class for the Critic and Critic Target
    There are 2 critic models in T3D. Hence we need to define 2 DNNs in this class
    The forward function returns a tuple of both DNNs outputs
    We also define a function to return the Q value from just the first critic DNN for updating the Actor weights.

![Step 3](https://drive.google.com/uc?export=view&id=1BS4_lneg7EwK10_2cC76aBPpx3TDLHzg)

# Intermediate step
Declare the T3D class
This houses 4 objects ( 6 DNN models )
    1. Actor
    2. Critic ( contains two critic DNNs internally )
    3. Actor target
    4. Critic target ( contains 2 critic DNNs internally )

![Step 3.5](https://drive.google.com/uc?export=view&id=1jQa0iMXgcPL7fyBfK-wR3ow98T05SEC3)

# Training begins
Do the following steps for each iteration ( num iterations = 100k )

# Step 4
Extract a sample from the replay memory
Convert each part of the sample into tensors

What we have now is a set of tuples of the form:

    <current state, current action, reward, next state>

![Step 4](https://drive.google.com/uc?export=view&id=1hOZ1ue7leRKRddoHOuM2ZnlHceT0ILLO)

# Step 5
Pass the next state through the actor target to obtain the right action to do at this point.

![Step 5](https://drive.google.com/uc?export=view&id=11tHKROsvyXLRLQ9P0_gcwNsDn9nppzYr)

# Step 6
Add gaussian noise to the action suggested by the actor target and clip the values to be within acceptable levels

![Step 6](https://drive.google.com/uc?export=view&id=1a-iiqCVMr4pI6HHkObAUIFPaeT6FFiF5)

# Step 7
Pass the next state and next action ( obtained in step 6 ) into the critic target to obtain two Q values from the two critic target DNNs

![Step 7](https://drive.google.com/uc?export=view&id=1cfz7l57qeSswhsq-qRXzww3LJMibZtgq)

# Step 8
Compute minimum of the 2 Q values obtained from the 2 critic target DNNs

![Step 8](https://drive.google.com/uc?export=view&id=1TX-yUF0_sZ9G4sEeiIKyiFey8Ws4LwXf)

# Step 9
Set target values for Q(current state, current action)

target Q := reward + ((1-done)*discount*target_Q).detach()

![Step 9](https://drive.google.com/uc?export=view&id=1mqFAXeuFVKIq7bOCIIQ3oTn_wMdnYySF)

# Step 10
Compute the outputs from the two critics for (current state, current action)

![Step 10](https://drive.google.com/uc?export=view&id=1CE5MyQYOdI94EdKVVg0QRXytT-nY0Jwp)

# Step 11
Compute critic loss and 

For each of the critic models it is the mean squared error between target Q value ( from step 9 ) and what that critic model provides

We sum the the losses across the two critic models

![Step 11](https://drive.google.com/uc?export=view&id=186vlMjSOtRet4J2Wk1xUkz8j42n7Ucuy)

# Step 12
Backpropogate the critic loss obtained in step 11

![Step 12](https://drive.google.com/uc?export=view&id=1CYolxou-lChAicsKDTZGeINbHmBvCCtr)

# Step 13
Every 2 iterations we compute loss on the actor and backpropogate the same

Actor loss is computed as the average ( across the batch sampled from replay memory ) of the critic 1's output value for (current state, current action)

![Step 13](https://drive.google.com/uc?export=view&id=1et19h7cAHZLeKO7TUgdWhOCLfXWlGuyj)

# Step 14 and 15
Every two iterations update the actor target and critic target with polyak averaging

![Step 14 15](https://drive.google.com/uc?export=view&id=1zOZAeZbRoMA8zxN_h5MO5NDsLFrPmDzJ)
