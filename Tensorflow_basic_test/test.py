import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

f=open('stock_dataset.csv')  
df=pd.read_csv(f)     
data=np.array(df['H_Hight'])   

plt.figure()
plt.plot(data)
plt.show()

print (1)