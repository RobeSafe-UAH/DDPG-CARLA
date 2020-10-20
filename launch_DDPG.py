import os
import carla_config as settings
import time

#Clear Carla Environment
print('### Reseting Carla Map ###')
os.system('python3 ' + settings.path2CARLA + 'PythonAPI/util/config.py -m ' + str(settings.CARLA_MAP))
time.sleep(5)

print('####### RUNNING DDPG ', settings.WORKING_MODE, ' IN ', settings.TRAIN_PLAY[settings.TRAIN_PLAY_MODE], ' MODE #######')
os.system('python3 DDPG/ddpg_carla.py --train ' + str(settings.TRAIN_PLAY_MODE))