import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("D:\\Aditya's Notes\\All Projects\\Customer Churn EDA Analysis using Python\\Customer Churn.csv")

# Basic Data Overview
print("Initial DataFrame Information:")
print(df.info())

# Data Cleaning: Replace blanks in 'TotalCharges' with 0 and convert to float
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)

print("DataFrame Information After Data Cleaning:")
print(df.info())

# Check for missing or duplicate values
print(f"Total Missing Values: {df.isnull().sum().sum()}")
print(f"Total Duplicate Rows: {df.duplicated().sum()}")
print(f"Duplicate Values on the basis of customerID: {df['customerID'].duplicated().sum()}")

# Convert 'SeniorCitizen' from 0/1 to 'Yes'/'No'
df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: "Yes" if x == 1 else "No")

# Plot Churn Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Churn', data=df)
ax.bar_label(ax.containers[0])
plt.title("Count of Customers by Churn", fontsize=14)
plt.show()

# Churn Percentage Pie Chart
churn_percent = df['Churn'].value_counts(normalize=True) * 100
plt.figure(figsize=(6, 6))
plt.pie(churn_percent, labels=churn_percent.index, autopct="%1.2f%%", startangle=90, colors=['skyblue', 'salmon'])
plt.title("Percentage of Churned Customers", fontsize=16)
plt.show()

# Analyze Gender and Churn
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="gender", data=df, hue="Churn", palette="muted")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Churn", loc='upper right')
ax.set_title("Churn by Gender", fontsize=14)
for container in ax.containers:
    ax.bar_label(container)
plt.show()

# Analyze Senior Citizen and Churn
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="SeniorCitizen", data=df, hue="Churn", palette="muted")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Churn", loc='upper right')
ax.set_title("Churn by Senior Citizen", fontsize=14)
for container in ax.containers:
    ax.bar_label(container)
plt.show()

# Stacked Bar Chart for Senior Citizen vs. Churn in Percentages
grouped = df.groupby(['SeniorCitizen', 'Churn']).size().unstack(fill_value=0)
percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
ax = percentages.plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'salmon'])
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center')
ax.set_title("Churn Percentage by Senior Citizen Status", fontsize=16)
plt.show()

# Tenure Distribution by Churn
plt.figure(figsize=(10, 6))
sns.histplot(x="tenure", data=df, bins=30, hue="Churn", multiple="stack", palette="muted", kde=False)
plt.title("Distribution of Tenure by Churn", fontsize=16)
plt.xlabel("Tenure (Months)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(title="Churn", loc='upper right')
plt.tight_layout()
plt.show()

# Contract Type and Churn
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Contract", data=df, hue="Churn", palette="muted")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Churn", loc='upper right')
ax.set_title("Churn by Contract Type", fontsize=14)
for container in ax.containers:
    ax.bar_label(container)
plt.show()

# Payment Method and Churn
plt.figure(figsize=(8, 6))
ax = sns.countplot(x="PaymentMethod", data=df, hue="Churn", palette="muted")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Churn", loc='upper right')
ax.set_title("Churn by Payment Method", fontsize=14)
for container in ax.containers:
    ax.bar_label(container)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Matrix (Selecting only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr = numeric_df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()
