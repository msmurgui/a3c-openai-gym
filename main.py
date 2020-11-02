from __future__ import print_function
import os
import threading
import multiprocessing
from envs import create_atari_env
from model import ActorCritic
from trainer import Trainer
from validator import Validator
import tensorflow as tf
import numpy as np

# Gather all the parameters


class Parameters():
    def __init__(self):
        self.learningRate = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        # Set processes to number of available CPU threads
        self.numberProcesses = multiprocessing.cpu_count()
        self.numberSteps = 40
        self.maxEpisodeLength = 1000
        self.envName = 'Breakout-v0'
        self.convolutionLayers = np.array(
            [[(3, 3), 32, 1, 2], [(3, 3), 32, 1, 2], [(3,3), 32, 1, 2], [(3, 3), 32, 1, 2]])
        self.denseLayers = np.array([])
        self.lstmUnits = 256


os.environ['OMP_NUM_THREADS'] = '1'  # 1 thread per core
parameters = Parameters()
# we create an optimized environment thanks to universe
env = create_atari_env(parameters.envName)

globalModel = ActorCritic(
    name='global',
    numberOutputs=env.action_space.n,
    convolutionLayers=parameters.convolutionLayers,
    denseLayers=parameters.denseLayers,
    lstmUnits=parameters.lstmUnits,
    training=True
)
optimizer = tf.keras.optimizers.Adam(parameters.learningRate)

checkpoint = tf.train.Checkpoint(model=globalModel, optimizer=optimizer)
checkpointManager = tf.train.CheckpointManager(
    checkpoint, './modelCheckpoint', max_to_keep=3)

checkpoint.restore(checkpointManager.latest_checkpoint)
if checkpointManager.latest_checkpoint:
    print("Restored from {}".format(checkpointManager.latest_checkpoint))
else:
    print("Initializing from scratch.")


coordinator = tf.train.Coordinator()
trainerThreads = []

initState = env.reset()
initState = tf.convert_to_tensor(initState, dtype=np.float32)

print('Shape: ', initState.shape )
print( 'Expanded shape: ', tf.expand_dims(initState, 1).shape)

pred, initialModelState = globalModel.warmupCall(tf.expand_dims(initState, 0))
globalModel(tf.expand_dims(initState, 0), initialModelState)
globalModel.summary()

# Trainer Threads created for model training. Number of cpu threads available
# minus 1 for validator
for threadId in range(parameters.numberProcesses):
    print('Trainer ', threadId, ' process created.')
    trainer = Trainer(trainerId=threadId,
                      coordinator=coordinator,
                      globalParams=parameters,
                      globalModel=globalModel,
                      optimizer=optimizer,
                      )
    thread = threading.Thread(target=trainer.train)
    thread.start()
    trainerThreads.append(thread)

# Validator thread created for model validation.
#print('Validator ', parameters.numberProcesses-1, ' proceess created.')
#validator = Validator(validatorId=parameters.numberProcesses-1,
#                      coordinator=coordinator,
#                      globalParams=parameters,
#                      globalModel=globalModel,
#                      globalManager=checkpointManager
#                      )
#thread = threading.Thread(target=validator.validate)
#thread.start()
#trainerThreads.append(thread)

coordinator.join(trainerThreads)
