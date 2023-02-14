import os
import pandas as pd

def createDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

PATH = "D:/codes/media2/output"
EXPERIMENT_ID = 'ENSEMBLE-LOO2'

loos = [ ("cGAN", 0, "N11280610"), ("cGAN", 1, "N11280611"), ("cGAN", 2, "N11280612"), ("cGAN", 3, "N11280613"), ("cGAN", 4, "N11280614")]
ensembles = [("cGAN", 64, "N11281839"), ("cGAN", 128, "N11281840"), ("cGAN", 256, "N11281841"), ("cGAN", 512, "N11281842"), ("cGAN", 1024, "N11281843")]
prefix = "Processing Testing... "

output = []

for loo in loos:
    model, i, fname = loo
    print(f"Processing: {model}, {i}, {fname}")
    try:
        f = open(f"{PATH}/{fname}.out", "r")
        for l in f:
            if l.startswith(prefix):
                content = {}
                metrics = l.replace(prefix, "").replace(" ", "").replace("#","").replace('\n','').split(",")
                for metric in metrics:
                    m, val = metric.split(":")
                    content[m] = val
                content['model'] = model
                content['param'] = i
                content['exper'] ="loo"
                output.append(content)
    except:
        print(f"Error: {fname}")

for ensemble in ensembles:
    model, i, fname = ensemble
    print(f"Processing: {model}, {i}, {fname}")
    try:
        f = open(f"{PATH}/{fname}.out", "r")
        for l in f:
            if l.startswith(prefix):
                content = {}
                metrics = l.replace(prefix, "").replace(" ", "").replace("#","").replace('\n','').split(",")
                for metric in metrics:
                    m, val = metric.split(":")
                    content[m] = val
                content['model'] = model
                content['param'] = i
                content['exper'] ="ensemble"
                output.append(content)
    except:
        print(f"Error: {fname}")

loos = [ ("cycleGAN", 0, "N11281000"), ("cycleGAN", 1, "N11281100"), ("cycleGAN", 2, "N11281200"), ("cycleGAN", 3, "N11281300"), ("cycleGAN", 4, "N11281400")]
ensembles = [("cycleGAN", 64, "N11281844"), ("cycleGAN", 128, "N11281845"), ("cycleGAN", 256, "N11281846"), ("cycleGAN", 512, "N11281847"), ("cycleGAN", 1024, "N11281848")]

for loo in loos:
    model, i, fname = loo
    print(f"Processing: {model}, {i}, {fname}")
    try:
        f = open(f"{PATH}/{fname}.out", "r")
        for l in f:
            if l.startswith(prefix):
                content = {}
                metrics = l.replace(prefix, "").replace(" ", "").replace("#","").replace('\n','').split(",")
                for metric in metrics:
                    m, val = metric.split(":")
                    content[m] = val
                content['model'] = model
                content['param'] = i
                content['exper'] ="loo"
                output.append(content)
    except:
        print(f"Error: {fname}")

for ensemble in ensembles:
    model, i, fname = ensemble
    print(f"Processing: {model}, {i}, {fname}")
    try:
        f = open(f"{PATH}/{fname}.out", "r")
        for l in f:
            if l.startswith(prefix):
                content = {}
                metrics = l.replace(prefix, "").replace(" ", "").replace("#","").replace('\n','').split(",")
                for metric in metrics:
                    m, val = metric.split(":")
                    content[m] = val
                content['model'] = model
                content['param'] = i
                content['exper'] ="ensemble"
                output.append(content)
    except:
        print(f"Error: {fname}")

df = pd.DataFrame(output)
df.to_csv(f"{PATH}/{EXPERIMENT_ID}_results.csv", index=False)

df2 = df.groupby(['model', 'param'], as_index=False).mean()
df2.to_csv(f"{PATH}/{EXPERIMENT_ID}_results_summary.csv", index=False)

print('HERE')
