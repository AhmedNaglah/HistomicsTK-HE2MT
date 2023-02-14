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

counter = 1

def saveShellScript(l5, job):
    global PATH
    l1 = f"#!/bin/bash \n"
    l2 = f"#SBATCH --job-name={job}.job \n"
    l3 = f"#SBATCH --mail-user=$USER@louisville.edu \n"
    l4 = f"#SBATCH --output={job}.out  \n"
    l4_ = f"#SBATCH -p gpu \n"

    fname_ = f"{job}.sh"
    fname = f"{PATH}/{job}.sh"
    f = open(fname, "x")
    f.write(l1)
    f.write(l2)
    f.write(l3)
    f.write(l4)
    f.write(l4_)
    f.write(l5)
    f.close()
    print(f'dos2unix {fname_}')
    print(f'sbatch {fname_}')

command = 'python -u $HOME/projects/media2/src/runHE2MT.py'
dataroot = '$HOME/projects/media2/data/ensemble/liver256/'
model_ = 'condGAN256'
layers = [1,2,3,4,5,6,7,8]
epochs = 10
l = 100
lr = 2e-4

for layer in layers:
    job = f'{timestampStr}_{counter}'
    model = f"{model_}_{layer}"
    counter+=1
    l5 = f"{command} --dataroot {dataroot} --model {model} --epochs {epochs} --lamda {l} --lr {lr} --experiment_id {job}"
    saveShellScript(l5, job)