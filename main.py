import Neutral_Network as nn
import numpy as np
import os

"""

Array Appending


>>> import numpy as np
>>> np_arr1 = np.array([[1, 2], [3, 4]])
>>> np_arr2 = np.array([[10, 20], [30, 40]])
>>> 
>>> np.append(np_arr1, np_arr2)
array([ 1,  2,  3,  4, 10, 20, 30, 40])
>>>
>>> np.append(np_arr1, np_arr2, axis=0)
array([[ 1,  2],
       [ 3,  4],
       [10, 20],
       [30, 40]])
>>>
>>> np.append(np_arr1, np_arr2, axis=1)
array([[ 1,  2, 10, 20],
       [ 3,  4, 30, 40]])
>>> 
>>> np.insert(np_arr1, 1, np_arr2, axis=0)
array([[ 1,  2],
       [10, 20],
       [30, 40],
       [ 3,  4]])
>>> 
>>> np.insert(np_arr1, 1, np_arr2, axis=1)
array([[ 1, 10, 30,  2],
       [ 3, 20, 40,  4]])
>>> 

"""

"""

numpy.sum


>>> a=np.sum([[0,1,2],[2,1,3]])
>>> a
9
>>> a.shape
()
>>> a=np.sum([[0,1,2],[2,1,3]],axis=0)
>>> a
array([2, 2, 5])
>>> a.shape
(3,)
>>> a=np.sum([[0,1,2],[2,1,3]],axis=1)
>>> a
array([3, 6])
>>> a.shape
(2,)


"""

"""

np.array--> copy


Case 1:

a1 = np.array([1, 2, 3])
a2 = np.array(a1, copy=False)
a1[0] = 50
print(a1)
# [50, 2, 3]
print(a2)
# [50  2  3]

When you pass another numpy array to np.array you can either copy the contents it
to a new object in the memory, or not do that. Normally, you want to copy because
you don't want to modify the original array, but there are circumstances where
that's not a good thing.


Case2:

orig = np.array([1, 2, 3])
modified = np.array(orig, dtype=float, copy=False)
modified[0] = 50
print(modified)
# [50.  2.  3.]
print(orig)
# [1, 2, 3]

In above example, you are asking numpy to convert all the data to float.
It cannot possibly do that without copying the data to a new object.
So "a copy is needed to satisfy this requirement". numpy will ignore copy=False.
"""


#Testing---------------------------

test = nn.Neutral_Network()
nums_1 = np.array([1,-2,3])
nums_2 = np.array([-4,5,-6])

a = test.relu(nums_1)
k = nums_1[nums_1>0]

nums_1[nums_2<=0] = 0
nums_1.reshape((1,1,3))

"""
initialize parameter

np.random.seed(3)
test.initialize_parameters([5,4,3])
print("W1 = " + str(test.parameters["W1"]))
print("b1 = " + str(test.parameters["b1"]))
print("W2 = " + str(test.parameters["W2"]))
print("b2 = " + str(test.parameters["b2"]))

"""

"""

forward Propagation Testing

"""

#Test 1
def linear_forward_test_case():
    np.random.seed(1)
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = np.array([[1]])
    """
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    
    return A, W, b

"""

A, W, b = linear_forward_test_case()

Z, linear_cache = test.linear_forward(A, W, b)
print("Z = " + str(Z))

"""

#Test 2
def linear_activation_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = 5
    """
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

"""

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = test.neutron_forward(A_prev, W, b,"sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = test.neutron_forward(A_prev, W, b, "relu")
print("With ReLU: A = " + str(A))

"""

#Test 3
def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return X, parameters

"""

test.data_X, test.parameters = L_model_forward_test_case_2hidden()
AL, caches = test.forward_propagation()
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

"""

#Test 4

def compute_cost_test_case():
    Y = np.asarray([[1, 1, 0]])
    aL = np.array([[.8,.9,0.4]])
    
    return Y, aL

"""
           
Y, AL = compute_cost_test_case()
test.data_Y = Y

print("cost = " + str(test.cost_function(AL)))

"""

#test 5

def linear_activation_backward_test_case():
    """
    aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
    """
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)
    
    return dA, linear_activation_cache

"""
dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = test.neutron_backward(dAL, linear_activation_cache,"sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = test.neutron_backward(dAL, linear_activation_cache,"relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
"""

#test 6
def L_model_backward_test_case():

    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}
    AL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
    
    """
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)


    
    return AL, Y, caches


def print_grads(grads):
    
    print ("dW1 = "+ str(test.grads["dW1"]))
    print ("db1 = "+ str(test.grads["db1"]))
    print ("dA1 = "+ str(test.grads["dA1"]))

"""
AL, Y_assess, caches = L_model_backward_test_case()
test.data_Y = Y_assess
grads = test.backward_propagation(AL, caches)
print_grads(grads)

"""

#test 7

def update_parameters_test_case():
    
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return parameters, grads

"""
test.parameters, test.grads = update_parameters_test_case()
test.update_parameter(0.1)

print ("W1 = "+ str(test.parameters["W1"]))
print ("b1 = "+ str(test.parameters["b1"]))
print ("W2 = "+ str(test.parameters["W2"]))
print ("b2 = "+ str(test.parameters["b2"]))

"""

#end test--------------

#Main

"""
np.random.seed(1)
trial = nn.Neutral_Network()
trial.Modelling(alpha = 0.1,iteration = 4000,print_cost=True)
res = trial.Accuracy(trial.data_Y)

print ("Accuracy: %f" %(res))

path = os.getcwd()
np.savetxt(path+"/"+"Data_X.csv",trial.data_X,delimiter = ',',fmt='%f')
np.savetxt(path+"/"+"Data_Y.csv",trial.data_Y,delimiter = ',',fmt='%f')
"""
