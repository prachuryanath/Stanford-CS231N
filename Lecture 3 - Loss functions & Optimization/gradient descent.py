# This simple loop is at the core of all Neural Network libraries. 
# There are other ways of performing the optimization (e.g. LBFGS), 
# but Gradient Descent is currently by far the most common and 
# established way of optimizing Neural Network loss functions. 

# Vanilla Gradient Descent

while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update


# Vanilla Minibatch Gradient Descent


while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update

# In large-scale applications (such as the ILSVRC challenge), the training data can have on order 
# of millions of examples. Hence, it seems wasteful to compute the full loss function 
# over the entire training set in order to perform only a single parameter update. A very common approach 
# to addressing this challenge is to compute the gradient over batches of the training data. For example, 
# in current state of the art ConvNets, a typical batch contains 256 examples from the entire training 
# set of 1.2 million. This batch is then used to perform a parameter update:

