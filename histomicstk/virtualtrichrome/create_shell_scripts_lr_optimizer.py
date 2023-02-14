import os
from datetime import datetime

OUTDIR = 'D:/codes/media2'

def createDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False 

dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%b%d%H%M")

createDir(f"{OUTDIR}/{timestampStr}")

PATH = f"{OUTDIR}/{timestampStr}"

counter = 0

def saveShellScript(l5, job):
    global PATH
    l1 = f"#!/bin/bash \n"
    l2 = f"#SBATCH --job-name={job}.job \n"
    l3 = f"#SBATCH --mail-user=$USER@louisville.edu \n"
    l4 = f"#SBATCH --output={job}.out  \n"
    fname_ = f"{job}.sh"
    fname = f"{PATH}/{job}.sh"
    f = open(fname, "x")
    f.write(l1)
    f.write(l2)
    f.write(l3)
    f.write(l4)
    f.write(l5)
    f.close()
    print(f'dos2unix {fname_}')
    print(f'sbatch {fname_}')

command = 'python -u $HOME/projects/media2/src/runHE2MT.py'
dataroot = '$HOME/projects/media2/data/ensemble/liver256/'
model = 'condGAN256'
lamdas = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325]
learning_rates = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
epochs = 2
l = 100

for lr in learning_rates:
    job = f'{timestampStr}_{counter}'
    counter+=1
    l5 = f"{command} --dataroot {dataroot} --model {model} --epochs {epochs} --lamda {l} --lr {lr} --experiment_id {job}"
    saveShellScript(l5, job)

dataroot = '$HOME/projects/media2/data/ensemble/liver_cycleGAN256/'
model = 'cycleGAN256'
lamdas = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325]
learning_rates = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

epochs = 2
l = 100
for lr in learning_rates:
    job = f'{timestampStr}_{counter}'
    counter+=1
    l5 = f"{command} --dataroot {dataroot} --model {model} --epochs {epochs} --lamda {l} --lr {lr} --experiment_id {job}"
    saveShellScript(l5, job)

