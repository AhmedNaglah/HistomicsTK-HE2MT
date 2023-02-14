import os
from datetime import datetime

OUTDIR = 'D:/codes/media2'

dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%b%d%H%M")

counter = 0

def saveShellScript(l5, job):
    l1 = f"#!/bin/bash \n"
    l2 = f"#SBATCH --job-name={job}.job \n"
    l3 = f"#SBATCH --mail-user=$USER@louisville.edu \n"
    l4 = f"#SBATCH --output={job}.out  \n"
    fname_ = f"{job}.sh"
    fname = f"{OUTDIR}/{job}.sh"
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
epochs = 2

for l in lamdas:
    job = f'{timestampStr}_{counter}'
    counter+=1
    l5 = f"{command} --dataroot {dataroot} --model {model} --epochs {epochs} --lamda {l} --experiment_id {job}"
    saveShellScript(l5, job)

dataroot = '$HOME/projects/media2/data/ensemble/liver_cycleGAN256/'
model = 'cycleGAN256'
lamdas = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325]
epochs = 2

for l in lamdas:
    job = f'{timestampStr}_{counter}'
    counter+=1
    l5 = f"{command} --dataroot {dataroot} --model {model} --epochs {epochs} --lamda {l} --experiment_id {job}"
    saveShellScript(l5, job)
