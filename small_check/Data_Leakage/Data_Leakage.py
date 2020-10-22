import pandas as pd

# Read the data
data = pd.read_csv('AER_credit_card_data.csv')


print (data.columns)

feature = "expenditure"
labels = ["card = No","card = Yes"]
data_hist = [data[data['card']==0][feature], data[data['card']==1][feature]]

print (data_hist)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

axs.hist(data_hist, label=labels)
plt.show()


## Select target
#y = data.card
#
## Select predictors
#X = data.drop(['card'], axis=1)
#
#print("Number of rows in the dataset:", X.shape[0])
#X.head()
