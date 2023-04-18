import numpy      as np
import tensorflow as tf

from tensorflow       import keras
from tensorflow.keras import layers

import sys
sys.path.append("../../")

from toolkit5G.SymbolMapping import Demapper
from toolkit5G.SymbolMapping import Mapper

class LLRNet:
    
    def __init__(self, modOrder, nodesPerLayer = np.array([32, 64, 32]), numTrainingSamples = 10000000, 
                 numTestSamples = 10000, activationfunctions = np.array(['tanh', 'tanh', 'tanh', 'linear']), 
                 loss = 'mse', metrics=['accuracy'], epochs=100, batch_size=100):
        self.modOrder      = modOrder
        self.nodesPerLayer = nodesPerLayer.flatten()
        self.numLayers     = self.nodesPerLayer.size
        self.loss          = loss
        self.metrics       = metrics
        self.epochs        = epochs
        self.batch_size    = batch_size
        
        self.numTrainingSamples  = numTrainingSamples*self.modOrder
        self.numTestSamples      = numTestSamples*self.modOrder
        self.activationfunctions = activationfunctions
        

        
    def __call__(self, SNR):

        # Create mapping and Demapping for Generating Training Dataset
        mapper   = Mapper("qam", self.modOrder)
        demapper = Demapper("app", "qam", self.modOrder, None, False)

        # Generate Training Dataset
        ## Generate Training inputs
        symbs    = mapper(np.random.randint(2, size = (1,self.numTrainingSamples))) 
        symbs    = symbs + np.sqrt(0.5/SNR)*(np.random.standard_normal(size=symbs.shape)+1j*np.random.standard_normal(size=symbs.shape)).astype(np.complex64)

        train_input        = np.zeros((symbs.size, 2))
        train_input[...,0] = np.real(symbs.flatten())
        train_input[...,1] = np.imag(symbs.flatten())
        
        ## Generate Training Labels
        train_label        = demapper([symbs, 1/SNR]).reshape(-1,self.modOrder, order = "C")
        
        # Keras Model for training
        ## Defining the model
        model = tf.keras.Sequential()
        
        ## Add Input layer to the model
        model.add(tf.keras.layers.Dense(self.nodesPerLayer[0], input_shape=(2,), activation=self.activationfunctions[0]))
        
        ## Add Hidden layer to the model
        for n in range(1, self.numLayers):
            model.add(keras.layers.Dense(self.nodesPerLayer[n], activation = self.activationfunctions[n]))
           
        ## Add Output layer to the model
        model.add(keras.layers.Dense(self.modOrder, activation = self.activationfunctions[-1]))

        ## Compilation of model
        model.compile(optimizer='adam', loss=self.loss, metrics=self.metrics)

        print()
        print()
        print("************************************ Training the Model ***********************************")
        print("...........................................................................................")
        # fitting the model
        model.fit(train_input, train_label, epochs=self.epochs, batch_size=self.batch_size)
        
        print()
        print()
        print("*********************************** Evaluating the Model **********************************")
        print("...........................................................................................")
        ## Model Evaluation
        _, accuracy = model.evaluate(train_input, train_label)
        print('Training Accuracy: %.2f' % (accuracy*100))
        
        
        # Generate Test Dataset
        ## Generate Test inputs
        symbs    = mapper(np.random.randint(2, size = (1,self.numTestSamples))) 
        symbs    = symbs + np.sqrt(0.5/SNR)*(np.random.standard_normal(size=symbs.shape)+1j*np.random.standard_normal(size=symbs.shape)).astype(np.complex64)

        test_input        = np.zeros((symbs.size, 2))
        test_input[...,0] = np.real(symbs.flatten())
        test_input[...,1] = np.imag(symbs.flatten())
        
        print()
        print()
        print("************************************ Testing the Model ************************************")
        print("...........................................................................................")
        ## Generate Test Labels
        test_label        = demapper([symbs, 1/SNR]).reshape(-1,self.modOrder, order = "C")

        pred_labels       = model.predict(test_input)

        print("Mean Square Error: "+str(np.mean((test_label-pred_labels)**2)))
        
        return model
        