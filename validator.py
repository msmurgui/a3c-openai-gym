from model import ActorCritic
from envs import create_atari_env
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class Validator():
    def __init__(self, validatorId, coordinator, globalParams, globalModel, globalManager):
        self.validatorId = validatorId
        self.name = 'validator-' + str(self.validatorId)
        self.coordinator = coordinator
        self.globalParams = globalParams
        self.globalModel = globalModel
        self.globalManager = globalManager

        self.env = create_atari_env(self.globalParams.envName)
        self.env.seed(self.validatorId + self.globalParams.seed)
        tf.random.set_seed(self.validatorId + self.globalParams.seed)

        self.localModel = ActorCritic(name=self.name,
                                      numberOutputs=self.env.action_space.n,
                                      convolutionLayers=self.globalParams.convolutionLayers,
                                      denseLayers=self.globalParams.denseLayers,
                                      lstmUnits=self.globalParams.lstmUnits,
                                      training=False)
        state = self.env.reset()
        state = tf.convert_to_tensor(state, dtype=np.float32)
        _, self.initialModelState = self.localModel.warmupCall(
            tf.expand_dims(state, 0))

    def validate(self):
        state = self.env.reset()
        state = tf.convert_to_tensor(state, dtype=np.float32)

        modelState = 0

        startTime = time.time()
        rewardSum = 0
        episodeCount = 0
        done = True

        
        while not self.coordinator.should_stop():
            self.env.render()
            episodeCount += 1

            self.localModel.set_weights(self.globalModel.get_weights())
            if done:
                cx = tf.zeros( [1, self.globalParams.lstmUnits])
                hx = tf.zeros( [1, self.globalParams.lstmUnits])
                modelState = (hx, cx)

            actorValue, _, modelState = self.localModel(
               [ tf.expand_dims(state, 0), modelState])

            probabilityDistro = tf.nn.softmax(actorValue)
            nextAction = tf.math.argmax(probabilityDistro, 1 )
            nextAction = tf.reshape(nextAction, (1, 1))
            #nextAction = tf.random.categorical(probabilityDistro, 1)
           
            state, reward, done, _ = self.env.step(nextAction.numpy())
            rewardSum += reward

            done = (done or episodeCount >= self.globalParams.maxEpisodeLength)
            if done:
                print("Time {}, episode reward {}, episode length {}".format(time.strftime(
                    "%Hh %Mm %Ss", time.gmtime(time.time() - startTime)), rewardSum, episodeCount))
                episodeCount = 0
                rewardSum = 0
                state = self.env.reset()
                self.globalManager.save()
                time.sleep(0)
            
            state = tf.convert_to_tensor( state, dtype= np.float32 )

            
