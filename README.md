# SoC-CNN-lytical

The following are the brief details of the models implemented in this repository
- Assignment 1 has implementation of image classification for ***MNIST*** dataset from scratch. It uses the following python Libraries:
	1. `numpy` arrays are used for simpler code and faster computations
	2. `matplotlib.pyplot` is used for plotting images and visualizing data and results
	3. `sklearn.model_selection.train_test_split` is used for splitting the dataset into train and test dataset
	
	The final accuracy of the trained model has an accuracy of ***95.067 %***.\
	The above libraries are used in all the model implementations.
- Assignment 2 has implementation of image classification for ***MNIST*** dataset. It uses `torch` library for easier and faster implementation of Neural networks.

	The final accuracy of the trained model has an accuracy of ***97.723 %***.
- Assignment 3 has implementation of image classification for ***CIFAR-10*** dataset using Convolutional Neural Networks. It uses `torch` for `Dataloader` class as well as for `torch.nn.Module` class for neural network. `torchvision` is used for transforming images.\
	The final accuracy of the trained model is ***66.242 %***.
- Assignment 4 is about image segmentation on ***Caravana*** dataset. It implements ***U-Net*** for the same. It also used `torch` and `torchvision` libraries.\
	The final accuracy of the trained model has an accuracy of ***89.714 %***.
