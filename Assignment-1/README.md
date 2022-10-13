# Assignment-1

The dataset used in the training of the model is ***MNIST dataset*** with image size of $28\times 28$. Therefore, input layer has ***784 nodes***, the hidden layer has ***256 nodes*** and the output layer has ***10*** layers. This simplified model has only 1 hidden layer implemented.

## Model Components

### Activation Functions

***Non-Linearity***: This model uses ***ReLu*** non-linearity. ReLu non-linearity keeps the positive numbers and replaces negative numbers by 0.

***Loss Calculation***: This model uses ***softmax*** as loss function.

The two common ways to calculate loss are SVM and Softmax.
In SVM, we mark loss is given by:
$$L_i = \sum_{j=/y_i}{max(0,s_j-s_{y_i}+1)}$$
The problem with this approach is that it doesn't differentiate between a difference of 1 or 100 in the scores. 

A better way to calculate loss is using **Softmax** Classifier. In this, we assume the score to be the unnormalized log probabilities. We calculate the probability using this and consider negative logarithm of probability as score.

### Layer Parameters

$$ H = W_0 X + b_0 $$
$$ O = W_1 H + b_1 $$

Where $X$ is input image, $H$ is hidden layer and $O$ is the output layer. 
The parameters $W_0, W_1, b_0, b_1$ are the parameters of the model. 
These parameters are optimized using backward propagation for prediction.
It is important that these parameters are initialized properly for good results.

#### Initialization of Parameters

The model uses ***Xavier Initialization***.
The biases are taken to be zero arrays whereas weights are taken as:

```
W = numpy.random.randn(input_dimension, output_dimension) / sqrt(input_dimension / 2)
```
The half in `sqrt` makes it work well with relu

### Cost Function

This function calculates the loss.
The model parameters are changed in epochs by propagation functions to reduce this loss.
The loss is considered to be the negative logarithm of probability.
If this loss value is less, that means that the model is predicting the correct label with more probability for that image.

## Propagation

### Forward Propagation

It just takes input `X` and returns `output_layer` given by:
```
hidden_layer = relu(X.dot(W0)+b0)
output_layer = relu(hidden_layer.dot(W1)+b1)
```
Where `W0`, `W1`, `b0`, `b1` are model parameters.

### Backward Propagation

Backward propagation updates the weights in the direction of gradient. 
The change in any value $A$ is given by 

$$\mathit{dA} = -\frac{\mathit{d}}{\mathit{dA}}O \times l$$

where $l$ represents ***learning rate*** and $O$ represents the Output.

The learning rate is also a crucial hyperparameter. A small hyperparameter will lead to a very high training time whereas a large hyperparameter won't let the accuracy increase.
The function takes learning rate as a parameter.

There are various techniques for backward propagation for achieving smaller train time. The model implemented uses a simple linear update.

## `train` function

This function uses the functions made above. It initializes the parameters of model and does forward and backward propagation.
The loss is printed every 100 epochs to ensure that the loss is decreasing.

This function returns the trained model.

## Prediction

Prediction of label for image is done using the trained model. 
Model here refers to the parameters of model and the computations to be done to get probabilities of various labels.
The following functions are used for prediction

### Predict

This function takes the model parameters and image as arguments.
The predicted label for the image is returned by the function.

### Accuracy

This function is used to calculate the accuracy for test set, given the predictions with correct labels.

## Results

While training, learning rate is taken to be $0.001$.
The model used 256 as the size of hidden layer.
The number of epochs is 1000
The following is the output showing how the loss value changed during training.

```
0 th iteration:
Loss: 2.302928390075917 	 lr: 0.001
100 th iteration:
Loss: 0.49789760225743873 	 lr: 0.001
200 th iteration:
Loss: 0.2938763791239336 	 lr: 0.001
300 th iteration:
Loss: 0.2768128279501788 	 lr: 0.001
400 th iteration:
Loss: 0.2797758323987911 	 lr: 0.001
500 th iteration:
Loss: 0.19697690354826541 	 lr: 0.001
600 th iteration:
Loss: 0.19647771168430328 	 lr: 0.001
700 th iteration:
Loss: 0.21803188757120498 	 lr: 0.001
800 th iteration:
Loss: 0.23709193874010234 	 lr: 0.001
900 th iteration:
Loss: 0.16253238433562608 	 lr: 0.001
```

The accuracy of model on test-set was 95.066 %.

The final model is saved as pickle file. The pickle file has the size of hidden layer and the values of model parameters.
