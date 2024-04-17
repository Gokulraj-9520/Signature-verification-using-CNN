import os
import shutil

value=['./BHSig260-Bengali/BHSig260-Bengali/','./BHSig260-Hindi/BHSig260-Hindi/','./CEDAR/CEDAR/']
try:
    os.makedirs("training")
    os.makedirs("training/forged")
    os.makedirs("training/genuine")
except:
    pass

for val in value:
    for i in os.listdir(val):
        number=os.path.join(val,i)
        for j in os.listdir(number):
            if "forge" in j:
                shutil.copy(os.path.join(number,j), "training/forged")
            elif "original" in j:
                shutil.copy(os.path.join(number,j),"training/genuine")
            elif "F" in j:
                shutil.copy(os.path.join(number,j),"training/forged")
            elif "G" in j:
                shutil.copy(os.path.join(number,j),"training/genuine")

 