import pandas as pd
import numpy as np

file_path = "D:\\jaswanth\\IV SEMISTER\\ML\LAB\\lab Session Data.xlsx"
data=pd.read_excel(file_path, sheet_name="Purchase data")
      
print(data)

X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = data["Payment (Rs)"].values

print("Features (X):")
print(X)    
print("Payment (y):  ")
print(y)

rank_X = np.linalg.matrix_rank(X)

print("Rank of Feature Matrix:", rank_X)
X_pinv = np.linalg.pinv(X)

cost = X_pinv.dot(y)

print("Cost of Candies     :", cost[0])
print("Cost of Mangoes (Kg):", cost[1])
print("Cost of Milk Packets:", cost[2])

data["Class"] = data["Payment (Rs)"].apply(
    lambda x: "RICH" if x > 200 else "POOR"
)


print(data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Class"]])