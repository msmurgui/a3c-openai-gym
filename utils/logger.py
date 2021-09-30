import tensorflow as tf
import datetime
from config.directories import LOGS_DIR

def createLogger(type='train'):
    currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logDir = LOGS_DIR + '/' + currentTime + '/' + type
    return tf.summary.create_file_writer(logDir)
    
