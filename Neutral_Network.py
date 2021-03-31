import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

class Neutral_Network:

    def __init__(self,dim1 = 64,dim2 = 64,dim3 = 3,esc = 0.01):

        self.data_X = None
        self.data_Y = None
        self.curr_file_path = os.getcwd()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.initial_esc = esc

        self.parameters = {}
        self.grads = {}

    def Filtering_Data(self):

        Source_Path = self.curr_file_path + "/" + "Source"
        Data_Path = self.curr_file_path + "/" + "Data"

        files_list = os.listdir(Source_Path)

        for Data in files_list:

            if Data == ".DS_Store":

                continue

            #Filtering
            Dname = Source_Path + "/" + Data          
            Image_k = Image.open(Dname)

            if Image_k.size[0] != self.dim1 or Image_k.size[1] != self.dim2:

                Image_k = Image_k.resize((self.dim1,self.dim2))

            Image_k.save( (Data_Path+"/"+Data) )

    def Constructing_DataSet(self):

        """
        Data_X : (n_x,number of samples)
        Data_Y : (1,number of samples)

        """

        Data_Path = self.curr_file_path + "/" + "Data"
        files_list = os.listdir(Data_Path)

        tmp_Y = []
        self.data_X = np.zeros((1,self.dim1,self.dim2,self.dim3))
        
        for Data in files_list:

            if Data == ".DS_Store":

                continue

            #Bulid Y
            tmp_Y.append(self.classify_label(Data))

            #Bulid X
            DPath = Data_Path + "/" + Data
            Image_k = plt.imread(DPath).reshape(1,self.dim1,self.dim2,self.dim3)
            self.data_X = np.append(self.data_X,Image_k,axis = 0)

        self.data_X = np.delete(self.data_X,(0),axis = 0)
        self.data_X = (self.data_X.reshape(self.data_X.shape[0],-1).T)/255
        self.data_Y = np.array(tmp_Y).reshape(1,len(tmp_Y))
            
            


    def sigmoid(self,Z):

        """
        Z : shape -> (n_l,number_of_samples)
        
        """

        res = 1/(1+np.exp(-Z))

        return res,Z

    def relu(self,Z):

        """
        Z : shape -> (n_l,number_of_samples)
        
        """

        return np.maximum(0,Z),Z

    def sigmoid_dZ(self,dA,Z):

        """
        Z : shape -> (n_l,number_of_samples)
        dA: shape -> (n_l,number_of_samples)
        
        """

        sig = 1/(1+np.exp(-Z))

        dG = sig*(1-sig)

        dZ = dA*dG

        return dZ

    def relu_dZ(self,dA,Z):

        """
        Z : shape -> (n_l,number_of_samples)
        dA: shape -> (n_l,number_of_samples)
        
        """

        """
        Method 1:
        
        dG = Z[:] not work!!!
        dG[dG<=0] = 0

        dZ = dG*dA

        """

        #Method 2:

        dZ = np.array(dA,copy=True)
        dZ[Z<=0] = 0

        return dZ

    def classify_label(self,name):

        if name[0] == "y":

            return 1

        else:

            return 0

    def initialize_parameters(self,layer_dims):

        L = len(layer_dims)

        for l in range(1,L):

            self.parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*self.initial_esc
            self.parameters["b"+str(l)] = np.zeros((layer_dims[l],1))


    def linear_forward(self,A_prev,W,b):

        """
        W : shape --> (l,l-1)
        b : shape --> (l,1)
        A_prev : shape --> (l-1,number of samples)
        """

        Z = np.dot(W,A_prev) + b
        linear_cache = (A_prev,W,b)

        return Z , linear_cache

    def neutron_forward(self,A_prev,W,b,A_func ="relu" ):

        """
        W : shape --> (l,l-1)
        b : shape --> (l,1)
        A_prev : shape --> (l-1,number of samples)
        """

        Z,linear_cache = self.linear_forward(A_prev,W,b)
        
        if A_func == "relu":

            A,activation_cache = self.relu(Z)

        elif A_func == "sigmoid":

            A,activation_cache = self.sigmoid(Z)


        cache = (linear_cache,activation_cache)
        
        return A,cache

    def forward_propagation(self):

        A = self.data_X
        L = int(len(self.parameters)/2)
        NN_cache = []

        for l in range(1,L):

            A_prev = A
            A , cache = self.neutron_forward(A_prev,self.parameters["W"+str(l)],self.parameters["b"+str(l)],A_func = "relu")
            NN_cache.append(cache)

        A_prev = A
        AL,cache = self.neutron_forward(A_prev,self.parameters["W"+str(L)],self.parameters["b"+str(L)],A_func = "sigmoid")
        NN_cache.append(cache)

        return AL,NN_cache

    def cost_function(self,AL):

        """
        Y : shape --> (1,number of samples)
        AL : shape --> (1, number of samples)
        """

        m = self.data_Y.shape[1]

        a1 = self.data_Y*np.log(AL)
        a2 = (1-self.data_Y)*np.log(1-AL)

        cost = -np.sum(a1+a2)/m
        cost = np.squeeze(cost)

        return cost
    

    def neutron_backward(self,dA,cache,A_func = "relu"):
            

        """
        cache --> (linear_cache,activation_cache) :

            linear_cache --> (A_prev,W,b)
            activation_cache --> Z

        Z: (l,number of samples)
        W : (l,l-1)
        b : (l,1)
        dA_prev: (l-1,number of samples)

        dZ : (l,number_of_samples)
        dW : (l,l-1)
        db : (l,1)
        dA: (l,number_of_samples)
        
        """

        m = dA.shape[1]

        if A_func == "relu":

            dZ = self.relu_dZ(dA,cache[1])
            dW = np.dot(dZ,cache[0][0].T)/m
            db = np.sum(dZ,axis=1,keepdims=True)/m
            dA_prev = np.dot(cache[0][1].T,dZ)

        elif A_func == "sigmoid":

            dZ = self.sigmoid_dZ(dA,cache[1])
            dW = np.dot(dZ,cache[0][0].T)/m
            db = np.sum(dZ,axis=1,keepdims=True)/m
            dA_prev = np.dot(cache[0][1].T,dZ)


        return dA_prev,dW,db

    def backward_propagation(self,AL,NN_cache):

        """

        NN_cache --> (cache0,cache1,......,cache(l-1))
        cache --> (linear_cache,activation_cache) :

            linear_cache --> (A_prev,W,b)
            activation_cache --> Z

        Z: (l,number of samples)
        W : (l,l-1)
        b : (l,1)
        A_prev: (l-1,number of samples)

        dZ : (l,number_of_samples)
        dW : (l,l-1)
        db : (l,1)

        """

        L = len(NN_cache)

        #Sigmoid backward
        dAL = -np.divide(self.data_Y,AL)+np.divide(1-self.data_Y,1-AL)
        dA_prev,dW,db = self.neutron_backward(dAL,NN_cache[L-1],"sigmoid")

        self.grads["dW"+str(L)] = dW
        self.grads["db"+str(L)] = db
        self.grads["dA"+str(L-1)] = dA_prev
                                              
        #Relu backward

        for l in range(L-2,-1,-1):

            dA_prev,dW,db = self.neutron_backward(self.grads["dA"+str(l+1)],NN_cache[l],"relu")

            self.grads["dW"+str(l+1)] = dW
            self.grads["db"+str(l+1)] = db
            self.grads["dA"+str(l)] = dA_prev
            


    def update_parameter(self,alpha=0.1):

        L = int(len(self.parameters)/2)

        for l in range(1,L+1):

            self.parameters["W"+str(l)] = self.parameters["W"+str(l)] - alpha*self.grads["dW"+str(l)]
            self.parameters["b"+str(l)] = self.parameters["b"+str(l)] - alpha*self.grads["db"+str(l)]



    def Modelling(self,alpha = 0.1,iteration = 1500,print_cost = False):

        self.Filtering_Data()
        self.Constructing_DataSet()

        self.initialize_parameters([self.data_X.shape[0],20, 7, 5, 1])

        for i in range(1,iteration+1):

            #Forward propagation
            AL,NN_cache = self.forward_propagation()

            #Cost
            cost = self.cost_function(AL)

            #Backward propagation
            self.backward_propagation(AL,NN_cache)

            self.update_parameter(alpha)

            if print_cost and i%100 == 0:

                print ("Cost after iteration %i: %f" %(i, cost))
                

    def predict(self,Y_hat):

        result = []
        
        for i in range(Y_hat.shape[1]):

            if Y_hat[0][i] > 0.5:

                result.append(1)

            else:
                
                result.append(0)

        return np.array(result).reshape((1,Y_hat.shape[1]))
                                        

    def Accuracy(self,Y_set):

       #predict
       AL,NN_cache = self.forward_propagation()

       #accuracy
       predict = self.predict(AL)

       res = np.mean(np.abs(predict-Y_set),axis = 1)
       res = np.squeeze(res)

       res = (1 - res)*100
       
       return res

        

        
