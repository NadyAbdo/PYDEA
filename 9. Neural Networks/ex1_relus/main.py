'''

1. Vanishing Gradients:

In the context of deep neural networks, during backpropagation, gradients are calculated with respect to the weights of the network.
These gradients are then used to update the weights in the opposite direction of the gradient to minimize the loss function. 
The vanishing gradient problem occurs when the gradients become extremely small as they are propagated backward through the layers of the network.

This problem is particularly pronounced in deep networks with many layers. 
As the gradients diminish exponentially with each layer, the updates to the weights become negligible in the early layers. 
As a result, these early layers learn very slowly or not at all, essentially hindering the training process.

One of the reasons for vanishing gradients is the use of activation functions with small derivatives, such as the sigmoid or hyperbolic tangent (tanh) functions. 
These functions squash their input into a small range, causing their gradients to be small in certain regions. 
This can make it difficult for the gradients to propagate effectively through the network.

2. Dying ReLUs:
Rectified Linear Units (ReLUs) are popular activation functions that replace all negative values in the input with zero. 
While ReLUs are computationally efficient and work well in many cases, they can suffer from the "dying ReLU" problem.

The dying ReLU problem occurs when a large gradient flows through a ReLU unit, causing the weights to be updated in such a way that the unit always outputs zero. 
Once a ReLU unit becomes inactive (always outputting zero), it effectively becomes a "dead" unit, and no gradient can flow through it during backpropagation. 
As a result, the weights associated with this unit stop learning, and the unit remains inactive throughout the training process.

The dying ReLU problem is often observed when the learning rate is too high or when a large number of ReLU units receive large negative inputs during training.

'''