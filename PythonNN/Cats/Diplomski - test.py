# -*- coding: utf-8 -*-
import scipy
from scipy import misc
from skimage import io
from skimage.transform import resize
from PIL import Image
from scipy import ndimage
from app_functions import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

"""
index = 1
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
"""

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]


x=' ';
print(5*x)
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
print(5*x)


# Reshape from (64x64x3,1) to (nx,m) 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T


print(5*x)
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print(5*x)


# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.



"""
#My picture tewst
my_image = "my_car.jpg" 
my_label_y = [0] # 


fname = "images/" + my_image
image = np.array(io.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
"""


layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    
    np.random.seed(1)
    costs = []                         # keeping track of cost
    

    parameters = initialize_parameters_deep(layers_dims)
 
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward prop
        AL, caches = L_model_forward(X, parameters)
        
        cost = compute_cost(AL, Y)
    
        # Back prop
        grads = L_model_backward(AL, Y, caches)
 
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # cost plot
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate= 0.0075, num_iterations = 2500, print_cost = True)

pred_train = predict(train_x, train_y, parameters)

pred_test = predict(test_x, test_y, parameters)

