import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1️⃣ Load CSV data
data = pd.read_csv("study_data.csv")

X = data[["Hours"]]
y = data["Marks"]

# 2️⃣ Train ML model
model = LinearRegression()
model.fit(X, y)

# 3️⃣ Take user input
hours_input = float(input("Enter study hours: "))

input_df = pd.DataFrame([[hours_input]], columns=["Hours"])
predicted_marks = model.predict(input_df)[0]

print(f"Predicted Marks: {predicted_marks:.2f}")

# 4️⃣ Prepare line for graph
x_line = pd.DataFrame(
    np.linspace(0, 10, 100),
    columns=["Hours"]
)
y_line = model.predict(x_line)


# 5️⃣ Create graph
plt.figure(figsize=(8, 5))

# Actual data points
plt.scatter(X, y, color="blue", label="Actual Data")

# Prediction line
plt.plot(x_line, y_line, color="black", linewidth=2, label="Prediction Line")

# Predicted point
plt.scatter(hours_input, predicted_marks, color="green", s=100, label="Your Prediction")

# Graph settings
plt.xlim(0, 10)
plt.ylim(0, 105)

plt.xlabel("Study Hours")
plt.ylabel("Expected Marks")
plt.title("Study Time and Performance Prediction")

plt.legend()
plt.grid(True)

plt.show()
