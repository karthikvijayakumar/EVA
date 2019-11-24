# Objective

The objective of this assignment is to look at two examples where people have trained a model on CIFAR-10 in under 120 seconds, observe their process and write down the key steps/improvements in their process

# mc.ai ( fenwicks ) website

Obtaining 94% on CIFAR-10 is quite hard

Steps in the fenwicks code:

1. Download the fenwicks library in the local file system
2. Set various hyperparameters as used by David Page ( with the exception of weight decay )
3. Setup GCS account configuration
The author seems to use GCS buckets to store intermediate results and the CIFAR dataset
4. Learn mean and std dev from training images and use them to normalise the train and test images
5. Compute and store the tf record version of the training and testing images in data_dir/train.tfrec and data_dir/test.tfrec respectively
6. Setting train and test image parsers/generators:
    a. Train parser: Comprises of the following image augmentations:
        i. padding 4px on all sides and taking a 32x32px crop
        ii. Random horizontal flip with 50% probability of flip
        iii. Cutout of 8x8
    b. Test parser: Simple parser with no transformation
7. Build the network layer by layer
8. Setup the learning rate scheduler
    Uses a triangular function to compute the learning rate at each epoch
9. Train the model
10. Evaluate results on the test dataset and report accuracy

# David Page's website

## Section 1

Main innovations in the 341 seconds from fast.ai student's submission for the DAWN Bench

    1. Mixed precision training
    2. Smaller network with sufficient capacity
    3. Employing higher learning rate

If one were to compute the total number of operations required to train a model and compare it with the number of operations a GPU can do each second, one could see that at 100% efficiency  the entire training process should take about 40 seconds ( under certain assumptions like parameter updates take no time etc. ). This means there is still a wide gap between the 341 seconds submission and what would be possible at 100% efficiency, hence one could potentially speed up the process considerably further.

Improvements made:

1. Removed one batch norm-relu group after the first convolution
2. Removed the kink in the learning rate at the 15th epoch
3. Moving one time image pre-processing tasks ( padding, normalisation and transposition ) to the start of the training 
    For heavier processes one could keep a running dataloader process for image pre-processing steps that are required at each step ( random cropping and flipping )
    Some image pre-processing steps are required at each pass through the training set and are redone at each iteration. PyTorch launches individual processes each time for image pre-processing. 
4. Make bulk calls to random generator at the start of each epoch instead of calling the random number generator at each epoch/iteration

The above improvements bring down the training time to 297 seconds

## Section2

Improvements:

1. Increase batch size to 512 and increase learning rate by 10%
    These improvements reduced the training time to 256s
2. To train a network at higher learning rates there are 2 regimes:
    a. Curvature effects dominate
    b. Catastrophic forgetting dominates

    For CIFAR-10 and the current model at batch size 128, we are in the latter. If we increase the batch size to 512, then we would be in the former.

## Section 3

1. On profiling the code for different batchsizes it was observed that convolutions and batch norm dominate the time consumed per epoch.
2. PyTorch by default converts a model to half precision which uses a slow code path and doesnt use the CuDNN methods. On converting ther batch norm weights to single precision the CuDNN libraries are triggered and there is almost a 2 second drop in time per epoch. This brings down the time to 186 seconds.
3. Further optimizations of the GPU code could be done but are left out at this point
4. Implementing cutout with 8x8 square leads to achievening 94% in 35 epohcs. With further tweaks to the learning rate schedule improves the median to 94.5% from 94.3%
5. If the learning rate schedule is accelerated to 30 epochs ( from 35  ) then we achieve a median accuracy of 94.13%. With a higher batch size of 768 the median drops to 94.06% but still achieves the target


## Section 4

This section focusses on changing the network architecture and making it simpler

Changes implemented:

1. Removing residual elements from the network and only retain the backbone
2. Replacing the downsampling convolutions with 3x3 with stride 2
3. Replacing the downsampling convolutions with 3x3 with stride 1 followed by a max pooling layer with window size 2x2
4. Replacing the final poooling layer before the classifier with a std global max pooling layer and doubling the output dimension to final convolution layer
5. Set constant initialization of the batch norm layer to 1.
6. Rescale the final classifier by a multiplicative factor of 0.125

The above changes bring the test accuracy to 91.1% in 20 epocsh and 47s. Clearly after the above changes the network is not complex enough to reach 94% accuracy in 20 epochs.

At this point one can introduce capacity into the network through various means. One could add additional conv layers or add back residual blocks. With some brute force search it is found that adding residual blocks to the first and third layer performs the best. This best model has a test accuracy of 93.8% after 20 epochs in 66s. When extended to 24 epochs, it improves to 94.08% in 79s.

## Section 5

This section focusses on tuning hyperparameters

1. Hyper params to tune are maximal learning rate(lambda), momentum(p), batch size(N) and weight decay(alpha)
2. Notice flat directions in which lambda/N, lambda/(1-p) and lambda*alpha are constant
3. To find the ideal set of hyperparams one can perform a cyclical coordinate descent where we tune one parameter at a time. It would be better to perform the search in the space of ( ((lambda*alpha)/(1-p)), p and alpha ) to align with the flat directions in the space
4. Theoretical observations/results:
    1. Changing momentum, keeping lambda/(1-p) constant does not have any effect on training
    2. Weight decay in the presence of batch normalisation acts as a stable control mechanism on the effective step size. If gradient updates get too small, weight decay shrinks the weights and boosts gradient step sizes until equilibrium is restored. The reverse happens when gradient updates grow too large.
    3. For weights with a scaling symmetry, gradients are orthogonal to weights. Hence gradient updates lead to an increase in weight norm whilst weight decay leads to a decrease
    4. For small weights the growth term dominates and vice versa for large weights

## Section 6

1. In SGD with momentum, if we rescale batch size by a factor n and lambda by the same amount, then the effective step size gros by a factor of sqrt(n).
2. Gradients between different time steps are largely orthogonal
3. The rest of this section focussed on the similarities of LARS with SGD with momentum

## Section 7

This section goes into the detailed mechanism by which batch norm enables the high learning rates crucial for rapid training.


## Section 8

Changes implemented:

1. Moved pre-processing of images to the GPU. Cutout is applied in batches to the images with prior shuffling of the images
2. Moving max pool layers to after the RELUs
3. Label smoothing -> Blending one-hot target probabilities with a uniform distribution over class labels
4. Using CELU instead of RELU as an activation function
5. Ghost batch norm -> Making batch norm updates happen in smaller chunks than the batch size
6. Frozen batch norm scales -> We observe that the scale of batch norm ( learned parameter ) converges to about 0.25 over the course of training. Hence it makes sense to fix instead of trying to learn it. Operationally it would be better to fix it at 1 and instead change the alpha in CELU by a factor of 4 and also change lambda and weight decay by a factor of 4^2 and (1/4)^2 respectively
7. Further increasing the learning rate by a factor of 4 and dividing weight decay by 4 improves results further.
8. Input patch whitening -> PENDING
9. Exponential moving average -> Parameters of the model are exponential moving averages of past values. These simulate annealing of learning rates in a certain way.
10. Test time augmentation -> Both the original image and its horizontal flip being given to the model and we pick the best output. Removing cutout further decreases the training time to 26s.

Without test time augmentation, David was able to reach an accuracy of 94.1% in 13 epocsh with a total training time of 34s.

With test augmentation, things further improved and he was able to reach an accuracy of 94.1% in just 10 epochs with a training time of just 26 seconds.