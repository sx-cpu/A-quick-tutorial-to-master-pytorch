# ---------------------------------------------------------------------------
# The principle for data spliting, model selection and loss function.
# As well as the ability for basic coding and ploting.
# 
# How to preprocess data
# How to select the best model  
# ---------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt 


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# enumarate to find the best parameters
w_list = []
mse_list = []
for w in np.arange(0, 4.1, 0.1):
    print(f'w = {w}') 
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print(f'\t {x_val, y_val, y_pred_val, loss_val}')
    print(f'MSE={l_sum / 3}')
    w_list.append(w)
    mse_list.append(l_sum / 3) 

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('Loss')   
plt.show()
