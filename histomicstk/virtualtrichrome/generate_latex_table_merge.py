import pandas as pd
import sys

PATH = "F:/N002-Research/liver-pathology/segmentation"
TABLE = "Validation Semantic"
CAPTION = "Quantitative results of the semantic segmentation experiments - validation"
SMALL = True
LABEL = 'tab-01'

DATA = [('Bile Duct Branch',[('cGAN + MT2F-CLR',[(0.81,0.03),(0.74,0.07)]),
('cGAN + MT2F-UNET',[(0.80,0.03),(0.73,0.07)]),
('cycleGAN + MT2F-CLR',[(0.70,0.06),(0.63,0.07)]),
('cycleGAN + MT2F-UNET',[(0.70,0.06),(0.63,0.07)]),
('HE2F',[(0.78,0.06),(0.76,0.05)])]),
('Hepatic Artery Branch',[('cGAN+MT2F-CLR',[(0.79,0.03),(0.69,0.09)]),
('cGAN+MT2F-UNET',[(0.79,0.04),(0.68,0.08)]),
('cycleGAN+MT2F-CLR',[(0.73,0.05),(0.57,0.19)]),
('cycleGAN+MT2F-UNET',[(0.73,0.05),(0.56,0.18)]),
('HE2F',[(0.80,0.06),(0.76,0.08)])]),
('Portal Vein Branch',[('cGAN+MT2F-CLR',[(0.87,0.04),(0.71,0.10)]),
('cGAN+MT2F-UNET',[(0.86,0.04),(0.70,0.10)]),
('cycleGAN+MT2F-CLR',[(0.80,0.09),(0.58,0.16)]),
('cycleGAN+MT2F-UNET',[(0.79,0.09),(0.57,0.15)]),
('HE2F',[(0.86,0.08),(0.75,0.09)])]),
('All Images',[('cGAN + MT2F-CLR',[(0.89,0.11),(0.43,0.24)]),
('cGAN + MT2F-UNET',[(0.88,0.11),(0.43,0.23)]),
('cycleGAN + MT2F-CLR',[(0.69,0.13),(0.14,0.14)]),
('cycleGAN + MT2F-UNET',[(0.70,0.13),(0.14,0.13)]),
('HE2F',[(0.85,0.13),(0.36,0.31)])]),
]

DATA = [
('All Images',[('cGAN + MT2F-CLR',[(0.89,0.11),(0.43,0.24)]),
('cGAN + MT2F-UNET',[(0.88,0.11),(0.43,0.23)]),
('cycleGAN + MT2F-CLR',[(0.69,0.13),(0.14,0.14)]),
('cycleGAN + MT2F-UNET',[(0.70,0.13),(0.14,0.13)]),
('HE2F',[(0.85,0.13),(0.36,0.31)])]),
('validation_samples',[('cGAN_MT2F_CLR',[(0.75,0.10),(0.08,0.10)]),
('cGAN_MT2F_UNET',[(0.76,0.10),(0.08,0.10)]),
('cycleGAN_MT2F_CLR',[(0.95,0.09),(0.00,0.02)]),
('cycleGAN_MT2F_UNET',[(0.95,0.09),(0.00,0.01)]),
('HE',[(0.69,0.19),(0.07,0.10)])]),
]

COLGRP = "Semantic segmentation"
COLUMNS = [ "Dataset", "Model", "ACC", "DSC"]

f = open(f'{PATH}/{TABLE}.out', 'w')

print( f"\\begin{{table}}[htb!]",file=f)
print( f"\\caption{{{CAPTION}}}",file=f)
print( f"\\centering",file=f)
if SMALL: print( f"\\small{{",file=f)

c = "c"
if len(COLUMNS)>1:
    for i in range(len(COLUMNS)-1):
        c+= "|c"

print(f"\\begin{{tabular}}{{l||{c}}}",file=f)
print(f"\cline{{0-{len(COLUMNS)}}}",file=f)

print(f"\multicolumn{{2}}{{}}{{}}&\multicolumn{{2}}{{|c}}{{{COLGRP}}} \\\\",file=f)

print(f"\hline",file=f)

c = f"&{COLUMNS[0]}"
if len(COLUMNS)>1:
    for i in range(len(COLUMNS)-1):
        c+= f"&{COLUMNS[i+1]}"

print(f"{c} \\\\",file=f)
print(f"\hline",file=f)

def format_Value(row):
    acc, dsc = row
    a, b = acc
    c, d = dsc
    return f"{a}$\\pm${b} & {c}$\\pm${d} "

def printRowParent(rowData):
    name, arr = rowData
    row_name, first_row = arr[0]
    global f
    out = f"\multirow{{{len(arr)}}}{{*}}{{{name}}} & {row_name} & {format_Value(first_row)} \\\\"
    print(out,file=f)
    if len(arr)>0:
        for i in range(len(arr)-1):
            row_name, row = arr[i+1]
            out = f"&{row_name}& {format_Value(row)} \\\\"
            print(out,file=f)
    print(f"\hline",file=f)

for parent in DATA:
    parentName, subDatas = parent
    printRowParent((parentName, subDatas))


print(f"\\end{{tabular}}",file=f) 
if SMALL: print(f"}}",file=f)
print(f"\\label{{tab:{LABEL}}}",file=f) 
print(f"\\end{{table}}",file=f) 

f.close()

