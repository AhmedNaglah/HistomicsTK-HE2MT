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

command = 'python -u $HOME/projects/media2/src/evaluate_similarity.py'

input_path = "D:/codes/media2/process_similarity.txt"

f = open(input_path, 'r')
for line in f:
    subroot, model, exper_old = line.replace("\n", "").split(',')

    dataroot = f'$HOME/projects/media2/data/ensemble/{subroot}/'

    job = f'{timestampStr}_{counter}'
    counter+=1
    l5 = f"{command} --dataroot {dataroot} --model {model} --experiment_id {job} --experiment_id_old {exper_old} "
    saveShellScript(l5, job)
