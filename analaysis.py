# Unemployment Analysis in Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv(r"G:\bhavana's project\bhanava 25th(3)\covid\unemployment.csv")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

print("🔎 Dataset Preview:")
print(df.head())

# 2. Basic info
print("\nDataset Info:")
print(df.info())

# 3. Handle missing values (if any)
df = df.dropna()

# 4. Convert 'Date' column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# 5. Overall trend plot
plt.figure(figsize=(12,6))
sns.lineplot(x="Date", y="Estimated Unemployment Rate (%)", data=df, marker="o")
plt.title("📈 Unemployment Rate Over Time")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Date")
plt.show()

# 6. Covid-19 impact (highlight 2020–2021)
plt.figure(figsize=(12,6))
sns.lineplot(x="Date", y="Estimated Unemployment Rate (%)", data=df, color="blue")
plt.axvspan(pd.to_datetime("2020-03-01"), pd.to_datetime("2021-12-31"),
            color="red", alpha=0.2, label="Covid-19 Period")
plt.title("Covid-19 Impact on Unemployment")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Date")
plt.legend()
plt.show()

# 7. Regional analysis
plt.figure(figsize=(14,7))
sns.boxplot(x="Region", y="Estimated Unemployment Rate (%)", data=df)
plt.xticks(rotation=90)
plt.title("Unemployment Rate by Region")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# 8. Yearly averages
df["Year"] = df["Date"].dt.year
yearly_avg = df.groupby("Year")["Estimated Unemployment Rate (%)"].mean()
print("\n📊 Average Unemployment Rate by Year:")
print(yearly_avg)

yearly_avg.plot(kind="bar", figsize=(8,4), color="skyblue", edgecolor="black")
plt.title("Average Unemployment Rate per Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()
