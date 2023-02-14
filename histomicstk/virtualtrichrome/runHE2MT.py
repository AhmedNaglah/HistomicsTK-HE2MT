import tensorflow as tf
import argparse
import importlib
from utils.deeptrain_util import *

""" BATCH_SIZE = 1
PATH = "D:/liver256_cond_tiny/"
MODEL = 'condGAN256'
LAMDA = 100
LR = 2e-4
EPOCHS = 4
OPTIMIZER = 'Adam'
EXPERIMENT_ID = 'condGAN256_tiny'
MONITORFREQ = '5epoch'
CHECKPOINTFREQ = 2
MODELSAVEFREQ = 2

BATCH_SIZE = 1
PATH = "F:/N002-Research/liver-pathology/liver256_cycle_tiny/"
MODEL = 'cycleGAN256'
LAMDA = 100
LR = 2e-4
EPOCHS = 4
OPTIMIZER = 'Adam'
EXPERIMENT_ID = 'condGAN256_tiny'
MONITORFREQ = 'batch'
CHECKPOINTFREQ = 2
MODELSAVEFREQ = 2 """

parser = argparse.ArgumentParser(description='Naglah Deep Learning - Model Training API')
parser.add_argument("--model", type= str, default= 'condGAN256', help="Transformation Model")
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--batchsize", type= int, default= 1, help="batch size for NN training")
parser.add_argument("--lamda", type= int, default= 100, help="Lambda for condGAN or cycleGAN")
parser.add_argument("--lr", type= float, default= 2e-4, help="Lambda for condGAN or cycleGAN")
parser.add_argument("--epochs", type= int, default= 25, help="Lambda for condGAN or cycleGAN")
parser.add_argument("--optimizer", type= str, default= 'Adam', help="Optimizer, Available Optimizers: Adam , Adagrad, SGD, RMSprop")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )
parser.add_argument("--monitor_freq", type= str, default= '1epoch', help="Monitoring Frequency")
parser.add_argument("--checkpoint_freq", type= int, default= 1, help="checkpoint_freq Frequency")
parser.add_argument("--modelsave_freq", type= int, default= 10, help="modelsave_freq Frequency")

try:
    params = parser.parse_args()

    BATCH_SIZE = params.batchsize
    PATH = params.dataroot
    MODEL = params.model
    LAMDA = params.lamda
    LR = params.lr
    EPOCHS = params.epochs
    OPTIMIZER = params.optimizer
    EXPERIMENT_ID = params.experiment_id
    MONITORFREQ = params.monitor_freq
    CHECKPOINTFREQ = params.checkpoint_freq
    MODELSAVEFREQ = params.modelsave_freq

    try:
        myClass = getattr(importlib.import_module(f'models.{MODEL}'), MODEL)
        try:
            if MODEL.startswith('condGAN'):
                train_dataset = tf.data.Dataset.list_files(f'{PATH}train/*.jpg')
                val_dataset = tf.data.Dataset.list_files(f'{PATH}val/*.jpg')
                test_dataset = tf.data.Dataset.list_files(f'{PATH}test/*.jpg')
                monitor_dataset = tf.data.Dataset.list_files(f'{PATH}monitor/*.jpg')

                train_dataset = train_dataset.map(load_image)
                train_dataset = train_dataset.batch(BATCH_SIZE)
                val_dataset = val_dataset.map(load_image)
                val_dataset = val_dataset.batch(BATCH_SIZE)
                test_dataset = test_dataset.map(load_image_test)
                test_dataset = test_dataset.batch(BATCH_SIZE)
                monitor_dataset = monitor_dataset.map(load_image)
                monitor_dataset = monitor_dataset.batch(BATCH_SIZE)

            if MODEL.startswith('cycleGAN'):
                trainA = tf.data.Dataset.list_files(f'{PATH}trainA/*.png')
                trainB = tf.data.Dataset.list_files(f'{PATH}trainB/*.png')

                trainA = trainA.map(loadCycleGAN)
                trainA = trainA.batch(BATCH_SIZE)

                trainB = trainB.map(loadCycleGAN)
                trainB = trainB.batch(BATCH_SIZE)

                train_dataset = trainA, trainB

                val_dataset = tf.data.Dataset.list_files(f'{PATH}val/*.jpg')
                test_dataset = tf.data.Dataset.list_files(f'{PATH}test/*.jpg')
                monitor_dataset = tf.data.Dataset.list_files(f'{PATH}monitor/*.jpg')
                val_dataset = val_dataset.map(load_image)
                val_dataset = val_dataset.batch(BATCH_SIZE)
                test_dataset = test_dataset.map(load_image_test)
                test_dataset = test_dataset.batch(BATCH_SIZE)
                monitor_dataset = monitor_dataset.map(load_image)
                monitor_dataset = monitor_dataset.batch(BATCH_SIZE)

            try:
                model = myClass()
                model.compile(optimizer=OPTIMIZER, lamda=LAMDA, learning_rate=LR)
                try:
                    model.fit(train_dataset, val_dataset, monitor_dataset, epochs=EPOCHS, experiment_id=EXPERIMENT_ID, dataroot=PATH, monitor_freq=MONITORFREQ, checkpointfreq=CHECKPOINTFREQ, modelsavefreq = MODELSAVEFREQ)
                    try:
                        model.test(test_dataset, experiment_id=EXPERIMENT_ID, dataroot=PATH)
                    except:
                        print(f'Error Code NAGLAH005 - Testing Error --->  Experiment Code: {EXPERIMENT_ID}')
                except:
                    print(f'Error Code NAGLAH004 - Training Error --->  Experiment Code: {EXPERIMENT_ID}')
            except:
                print(f'Error Code NAGLAH003 - Cannot Initialize or Compile Model --->  Experiment Code: {EXPERIMENT_ID}')
        except:
            print(f'Error Code NAGLAH002 - Cannot Load Dataset --->  Experiment Code: {EXPERIMENT_ID}')
    except:
        print(f'Error Code NAGLAH001 - Model is not found --->  Experiment Code: {EXPERIMENT_ID}')
except:
    print('Error Code NAGLAH000 - Script Error - Check Parameters')