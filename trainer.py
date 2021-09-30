from model import ActorCritic
from envs import create_atari_env
from utils.livePlotter import livePlotter
import tensorflow as tf
import numpy as np
import time
from loss import computeLoss


class Trainer():
    def __init__(self, trainerId, coordinator, globalParams, globalModel, optimizer):
        self.trainerId = trainerId
        self.name = 'trainer-' + str(trainerId)
        self.coordinator = coordinator
        self.globalModel = globalModel
        self.globalParams = globalParams
        self.optimizer = optimizer

        self.env = create_atari_env(self.globalParams.envName)
        self.env.seed(self.trainerId + self.globalParams.seed)
        tf.random.set_seed(self.trainerId + self.globalParams.seed)

        self.localModel = ActorCritic(name=self.name,
                                      numberOutputs=self.env.action_space.n,
                                      convolutionLayers=self.globalParams.convolutionLayers,
                                      denseLayers=self.globalParams.denseLayers,
                                      lstmUnits=self.globalParams.lstmUnits,
                                      training=True)

        state = self.env.reset()
        state = tf.convert_to_tensor(state, dtype=np.float32)
        _, self.initialModelState = self.localModel.warmupCall(
            tf.expand_dims(state, 0))

    def train(self):
        state = self.env.reset()
        episodeCount = 0
        done = True
        modelState = self.initialModelState

        avgCount = 0
        movingAverageLoss = []
        for _ in range(100):
            movingAverageLoss.append(0)

        while not self.coordinator.should_stop():
            with tf.GradientTape() as tape:
                episodeCount += 1
                avgCount += 1
                self.localModel.set_weights(self.globalModel.get_weights())
                if done:
                    cx = tf.zeros([1, self.globalParams.lstmUnits])
                    hx = tf.zeros([1, self.globalParams.lstmUnits])
                    modelState = (hx, cx)

                values = []
                logProbabilities = []
                rewards = []
                entropies = []

                for _ in range(self.globalParams.numberSteps):

                    actorValue, criticValue, modelState = self.localModel(
                        [tf.expand_dims(state, 0), modelState])
                    values.append(criticValue)

                    probabilityDistro = tf.nn.softmax(actorValue, axis=1)
                    nextAction = tf.random.categorical(probabilityDistro, 1)

                    logProbabilityDistro = tf.nn.log_softmax(
                        actorValue, axis=1)
                    logProbability = tf.gather(
                        params=logProbabilityDistro, axis=1, indices=nextAction[0][0])
                    logProbabilities.append(logProbability)

                    entropy = - tf.reduce_sum(
                        (logProbabilityDistro * probabilityDistro))
                    entropies.append(entropy)

                    state, reward, done, _ = self.env.step(nextAction.numpy())

                    reward = max(min(reward, 1), -1)
                    rewards.append(reward)

                    done = (done or episodeCount >=
                            self.globalParams.maxEpisodeLength)
                    if done:
                        episodeCount = 0
                        state = self.env.reset()
                        break

                cumulativeReward = tf.zeros((1, 1))

                if not done:
                    actorValue, criticValue, modelState = self.localModel(
                        [tf.expand_dims(state, 0), modelState])
                    cumulativeReward = criticValue
                else:
                    _, modelState = self.localModel.warmupCall(
                        tf.expand_dims(state, 0))

                values.append(cumulativeReward)

                loss = computeLoss(rewards, cumulativeReward, values,
                                   logProbabilities, entropies, self.globalParams)
                
                if(self.trainerId == 0):
                    print(self.name, " loss: ", loss)
                    movingAverageLoss.pop(0)
                    movingAverageLoss.append(loss)
                    total = 0
                    for x in movingAverageLoss:
                        total += x
                    print(self.name, ' moving average Loss: ', total /
                          len(movingAverageLoss), ' iter ', avgCount)

                gradients = tape.gradient(
                    loss, self.localModel.trainable_weights)
                #Why?
                gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                self.optimizer.apply_gradients(
                    zip(gradients, self.globalModel.trainable_weights))
