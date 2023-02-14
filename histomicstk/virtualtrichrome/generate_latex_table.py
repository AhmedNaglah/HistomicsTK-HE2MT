import pandas as pd
import sys

PATH = "D:/codes/media2/output"
TABLE = "LOO_cycleGAN"
CAPTION = "Leave-one-out (LOO) Results - cycleGAN"
COLUMNS = [ "Model", "MI", "NMI", "HC", "BCD"]
DATA = [
    ("cycleGAN-fold1", [(0.30,0.07), (0.07,0.02), (0.62,0.06), (0.31, 0.02)]),  
    ("cycleGAN-fold2", [(0.44,0.11), (0.10,0.02), (0.65,0.05), (0.33, 0.04)]),  
    ("cycleGAN-fold3", [(0.33,0.06), (0.07,0.01), (0.52,0.05), (0.37, 0.05)]),  
    ("cycleGAN-fold4", [(0.49,0.11), (0.11,0.02), (0.66,0.08), (0.39, 0.06)]),  
    ("cycleGAN-fold5", [(0.31,0.07), (0.07,0.02), (0.58,0.06), (0.40, 0.04)]),  
]

SMALL = False
LABEL = 'tab-22'

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
print(f"\hline",file=f)

c = f"&{COLUMNS[0]}"
if len(COLUMNS)>1:
    for i in range(len(COLUMNS)-1):
        c+= f"&{COLUMNS[i+1]}"

print(f"{c} \\\\",file=f)
print(f"\hline",file=f)

def format_Value(val):
    a, b = val
    return f"{a:.2f}$\\pm${b:.2f}"

def printRow(data):
    global f
    name, arr = data
    out = f"{name} & {format_Value(arr[0])} "
    if len(arr)>0:
        for i in range(len(arr)-1):
            out += f"& {format_Value(arr[i+1])}"
    out += "\\\\"
    print(out,file=f)
    print(f"\hline",file=f)

printRow(DATA[0])
if len(DATA)>1:
    for i in range(len(DATA)-1):
        printRow(DATA[i+1])

print(f"\\end{{tabular}}",file=f) 
if SMALL: print(f"}}",file=f)
print(f"\\label{{tab:{LABEL}}}",file=f) 
print(f"\\end{{table}}",file=f) 

f= sys.stdout
print( f"Done Table {TABLE}",file=f)

""" 
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
print(f"\hline",file=f)

c = f" & {COLUMNS[0]}"
if len(COLUMNS)>1:
    for i in range(len(COLUMNS)-1):
        c+= f"&{COLUMNS[i+1]}"

print(f"{c} \\\\",file=f)
print(f"\hline",file=f)

def printRow(arr, row_name):
    global f
    out = f"{row_name} & {arr[0]}"
    if len(arr)>0:
        for i in range(len(arr)-1):
            out += f"& {arr[i+1]}"
    out += "\\\\"
    print(out,file=f)
    print(f"\hline",file=f)

printRow(DATA[0], ROWS[0])
if len(DATA)>1:
    for i in range(len(DATA)-1):
        printRow(DATA[i+1], ROWS[i+1])

print(f"\\end{{tabular}}",file=f) 
if SMALL: print(f"}}",file=f)
print(f"\\label{{tab:{LABEL}}}",file=f) 
print(f"\\end{{table}}",file=f) 


 """