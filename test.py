import pandas as pd


#kolonları düzenlemek ve doğru olduğundan emin olmak için test ettim
train_df = pd.read_csv("data/train_small.csv")

# kolonları TEMİZLE
train_df.columns = train_df.columns.str.strip()

print(train_df.columns)