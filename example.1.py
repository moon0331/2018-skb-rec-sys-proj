import surprise
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=surprise.Dataset.load_builtin('ml-100k')
df=pd.DataFrame(data.raw_ratings, columns=['user','item','rate','id'])
del df['id']
print(df.head(10))

df_table=df.set_index(['user','item']).unstack()
print(df_table.iloc[212:222, 808:817].fillna(""))

'''plt.imshow(df_table)
plt.grid(False)
plt.xlabel("item")
plt.ylabel("user")
plt.title("Rate Matrix")
plt.show()'''

from surprise.model_selection import KFold

bsl_options = {
    'method': 'als',
    'n_epochs': 5,
    'reg_u': 12,
    'reg_i': 5
}
algo = surprise.BaselineOnly(bsl_options=bsl_options)

np.random.seed(0)
acc = np.zeros(3)
cv = KFold(3)
for i, (trainset, testset) in enumerate(cv.split(data)):
    algo.fit(trainset)
    predictions = algo.test(testset)
    acc[i] = surprise.accuracy.rmse(predictions, verbose=True)
print(acc.mean())

from surprise.model_selection import cross_validate

print(cross_validate(algo, data))  