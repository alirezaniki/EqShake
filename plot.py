import matplotlib.pyplot as plt
from csv import reader
import numpy as np



bw = 0.1
preds = []
with open('model_predictions.txt', 'r') as inp:
    lines = reader(inp, delimiter=" ")
    for line in lines:
        y, yhat = line
        diff = float(yhat) - float(y)
        preds.append(diff)

ave = round(np.mean(preds), 2)
std = round(np.std(preds), 2)

bins = np.arange(-2+bw/2, 2+bw/2, bw)
plt.hist(preds, bins=bins, align="mid", edgecolor='k', label=f'Std = {std}\nMean = {ave}') 

# Add labels and title
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('mag_hist.jpg', dpi=400)

# Show the plot
plt.show()