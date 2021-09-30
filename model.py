import tensorflow as tf


class ActorCritic(tf.keras.models.Model):
    '''
    len(convolution_layers) = number of convolution layers, 
    convolution_layers[i][0] = kernel shape (square)
    covolution_layers[i][1] = # of kernels per layer,
    covolution_layers[i][2] = stride,
    covolution_layers[i][3] = max_pool stride and kernel size,

    len(fully_connected_layers) = number of hidden neural layers, 
    fully_connected_layers[i] = # of neurons per layer
    '''

    def __init__(self, name, numberOutputs, convolutionLayers, denseLayers, lstmUnits, training, **kwargs):
        super().__init__(name=name, **kwargs)
        self.training = training

        # Convolutional Layers
        self.convolutionLayers = []
        self.poolingLayers = []
        i = 0
        for convLayer in convolutionLayers:
            self.convolutionLayers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        convLayer[1], convLayer[0], padding='same', activation=None),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.MaxPool2D(padding='same')
                ])
            )
            i += 1

        # Flattening Layer
        self.flatten = tf.keras.layers.Flatten()

        # LSTM Layer
        self.lstmCell = tf.keras.layers.LSTMCell(lstmUnits)
        self.lstmRNN = tf.keras.layers.RNN(self.lstmCell, return_state=True)

        # Dense Layers
        self.denseLayers = []
        i = 0
        for denseLayer in denseLayers:
            self.denseLayers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(
                        denseLayer, activation=None),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                ])
            )
            i += 1

        # Actor Layers
        self.actor = tf.keras.layers.Dense(
            numberOutputs, activation=None)

        # Critic Layer
        self.critic = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputAndState, training):
        inputTensor, state = inputAndState
        x = self.convolutionalLayerCall(inputTensor)
        x, state = self.lstmLayerCall(x, state)
        actorValues, criticValues = self.outputLayerCall(x)
        return actorValues, criticValues, state

    def warmupCall(self, inputTensor):
        x = self.convolutionalLayerCall(inputTensor)
        x = tf.reshape(x, [1, x.shape[0], x.shape[1]])  # REVER!!!!
        '''
        # x.shape => (batch, time, features)
        '''
        x, *state = self.lstmRNN(x)
        prediction = self.outputLayerCall(x)
        return prediction, state

    def convolutionalLayerCall(self, inputTensor):
        x = inputTensor
        for conv in self.convolutionLayers:
            x = conv(x)
        x = self.flatten(x)
        return x

    def lstmLayerCall(self, inputTensor, state):
        x, state = self.lstmCell(
            inputTensor, states=state, training=self.training)
        return x, state

    def outputLayerCall(self, inputTensor):
        x = inputTensor
        for dense in self.denseLayers:
            x = dense(x)
        actorValues = self.actor(x)
        criticValues = self.critic(x)
        return actorValues, criticValues
