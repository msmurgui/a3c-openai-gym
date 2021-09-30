import os

if(os.path.exists('/home/sebamurgui')):
    PROJECT_DIR = '/home/sebamurgui/Documents/HouseOfEngineering/CodeRoom/AI/a3c-openai-gym'
else:
    PROJECT_DIR = '/content/gdrive/MyDrive/Projects/a3c-openai-gym'

ASSETS_DIR = PROJECT_DIR + '/assets'
LOGS_DIR = ASSETS_DIR + '/logs'