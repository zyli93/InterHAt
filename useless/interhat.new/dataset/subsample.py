import pandas as pd

df = pd.read_csv("train_ori.csv")

df_pos = df[df['target'] == 1]
df_neg = df[df['target'] == 0]

df_sample_neg = df_neg.sample(frac=0.04, 
                              replace=False,
                              random_state=2018)

df_new = pd.concat([df_pos, df_sample_neg])
df_new = df_new.sample(frac=1)

df_new.to_csv("train.csv", index=False)
