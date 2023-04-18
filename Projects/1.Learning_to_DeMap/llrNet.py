import numpy      as np
import tensorflow as tf

from tensorflow       import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")

from toolkit5G.SymbolMapping import Demapper
from toolkit5G.SymbolMapping import Mapper

class LLRNet:
    
    def __init__(self, modOrder, nodesPerLayer = np.array([32, 64, 32]), numTrainingSamples = 10000000, 
                 numTestSamples = 10000, activationfunctions = np.array(['tanh', 'tanh', 'tanh', 'linear']), 
                 loss = 'mse', metrics=['accuracy'], epochs=4, batch_size=32):
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
        

        
    def __call__(self, SNR, inputs = None, returnModel = False):

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
        train_label        = demapper([symbs, np.float32(1/SNR)]).reshape(-1,self.modOrder, order = "C")
        
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
        
        
        if returnModel and (inputs is None):
            return model
        elif(returnModel and (inputs is not None)):
            test_input        = np.zeros((inputs.size, 2))
            test_input[...,0] = np.real(inputs.flatten())
            test_input[...,1] = np.imag(inputs.flatten())
            return model, model.predict(test_input).reshape(inputs.shape[0:-1]+(inputs.shape[-1]*self.modOrder,))
        elif(not returnModel and (inputs is None)):
            return None, None
        else:
            test_input        = np.zeros((inputs.size, 2))
            test_input[...,0] = np.real(inputs.flatten())
            test_input[...,1] = np.imag(inputs.flatten())
            return model.predict(test_input).reshape(inputs.shape[0:-1]+(inputs.shape[-1]*self.modOrder,))
        
         
    
    def displayMapping(self, modOrder = 2, bitLocations = [0,1], SNRdB = [-5,0,5], numSamples = 1000, displayRealPart = True):
        
        i = 0 if displayRealPart else 1
        xlabelLLR = "$\Re\{$Symbol$\}$" if displayRealPart else "$\Im\{$Symbol$\}$"
        fig, ax   = plt.subplots(len(SNRdB), len(bitLocations))
        
        # Create mapping and Demapping for Generating Training Dataset
        mapper    = Mapper("qam", modOrder)
        demapper  = Demapper("app",    "qam", modOrder, None, False)
        demapper2 = Demapper("maxlog", "qam", modOrder, None, False)

        # Generate Training Dataset
        ## Generate Training inputs
        symbols    = mapper(np.random.randint(2, size = (1, numSamples))) 
        
        train_input= np.zeros((symbols.size, 2))
        snrIndex = 0
        xmax = 0
        xmin = 0
        for snrdb in SNRdB:
            SNR      = 10**(snrdb/10)
            symbs    = symbols + np.sqrt(0.5/SNR)*(np.random.standard_normal(size=symbols.shape) + 1j*np.random.standard_normal( size = symbols.shape)).astype(np.complex64)

            train_input[...,0] = np.real(symbs.flatten())
            train_input[...,1] = np.imag(symbs.flatten())

            ## Generate Training Labels
            train_label        = demapper([symbs,  np.float32(1/SNR)]).reshape(-1, modOrder, order = "C")
            train_label2       = demapper2([symbs, np.float32(1/SNR)]).reshape(-1, modOrder, order = "C")
            
            ################################### LLRNet Model ################################
            # Keras Model for training
            ## Defining the model
            model = tf.keras.Sequential()

            ## Add Input layer to the model
            model.add(tf.keras.layers.Dense(self.nodesPerLayer[0], input_shape=(2,), activation=self.activationfunctions[0]))

            ## Add Hidden layer to the model
            for n in range(1, self.numLayers):
                model.add(keras.layers.Dense(self.nodesPerLayer[n], activation = self.activationfunctions[n]))

            ## Add Output layer to the model
            model.add(keras.layers.Dense(modOrder, activation = self.activationfunctions[-1]))

            ## Compilation of model
            model.compile(optimizer='adam', loss=self.loss, metrics=self.metrics)

            print()
            print()
            print("************************************ Training the Model ***********************************")
            print("...........................................................................................")
            # fitting the model
            model.fit(train_input, train_label, epochs=self.epochs, batch_size=self.batch_size, verbose = 0)

            print()
            print()
            print("*********************************** Evaluating the Model **********************************")
            print("...........................................................................................")
            ## Model Evaluation
            _, accuracy = model.evaluate(train_input, train_label)
            print('Training Accuracy: %.2f' % (accuracy*100))

            pred_labels = model.predict(train_input)
            
            bi = 0
            for bitIdx in bitLocations:
                ax[snrIndex, bi].plot(train_input[...,i], -train_label[...,bitIdx], 'o', mfc='none', 
                                      mec='royalblue', markersize = 6, label = "Log-MAP")
                ax[snrIndex, bi].plot(train_input[...,i], -train_label2[...,bitIdx], ".", mec = 'crimson', 
                                      markersize = 3, label = "Max-Log-MAP")
                ax[snrIndex, bi].plot(train_input[...,i], -pred_labels[...,bitIdx], ".", mec='gold', 
                                      markersize = 3, label = "LLRNet")
                bi += 1
            snrIndex += 1
            maxNew = np.max(train_input[...,i])
            minNew = np.min(train_input[...,i])
            xmax = xmax if maxNew < xmax else maxNew
            xmin = xmin if minNew > xmin else minNew
                

        snrIndex = 0
        for snrdb in SNRdB:
            bi   = 0
            for bitIdx in bitLocations:
                ax[snrIndex, bi].set_xlim([1.1*xmin, 1.1*xmax])
                ax[snrIndex, bi].grid()
                ax[snrIndex, bi].set_xlabel(xlabelLLR)
                ax[snrIndex, bi].set_ylabel("LLR for bit-"+str(bitIdx))
                ax[snrIndex, bi].set_title("SNR:"+str(snrdb)+"dB")
                
                
                bi += 1
            snrIndex += 1
        
        ax[0,0].legend(loc ='upper left')
        plt.show()
        
        return plt, fig, ax