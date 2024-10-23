# Customer Churn Analysis 📊

🌀 🔍 🐍 📊 📈

This project involves an **Exploratory Data Analysis (EDA)** of customer churn data using Python. The goal is to identify patterns and trends that help understand the factors contributing to customer churn. Below is an overview of the key processes, insights, and recommendations based on the analysis.

## Table of Contents
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Churn Distribution](#churn-distribution)
- [Key Drivers of Churn](#key-drivers-of-churn)
- [Visualizations and Insights](#visualizations-and-insights)
- [Customer Profiles and Recommendations](#customer-profiles-and-recommendations)
- [Advanced Visualizations](#advanced-visualizations)
- [Error Handling and Enhancements](#error-handling-and-enhancements)
- [Summary](#summary)

## Data Loading and Preprocessing
- The dataset is loaded into a Pandas DataFrame.
- Initial data cleaning involves replacing blank spaces in the TotalCharges column and converting it to a numeric type.
- Missing values and duplicate records are identified and handled.
- The SeniorCitizen column is converted from 0/1 to a more readable "Yes/No" format.

## Churn Distribution
- **Overall churn rate:** The analysis reveals that **26.54%** of the total customers have churned, while **73.46%** have remained. This highlights a significant portion of customers who discontinue services, necessitating a closer look at the factors driving this behavior.

## Key Drivers of Churn
- **Gender:** Churn appears fairly balanced across genders, with no significant difference in churn rates between male and female customers.
- **Senior Citizen Status:** A notable trend is observed among senior citizens, with a higher percentage of churn (approximately **42%** of senior citizens churned) compared to non-senior citizens (**23%**). This suggests older customers are more prone to leaving the service.
- **Contract Type:** Customers with **month-to-month contracts** are highly likely to churn (**43%** churn rate), significantly higher compared to those with **one-year** (**11%**) or **two-year contracts** (**3%**). This suggests that long-term contracts help in customer retention.
- **Tenure:** The tenure analysis indicates that customers who have been with the company for shorter periods (**1-2 months**) are more likely to churn, with the churn rate dropping as tenure increases. Long-term customers tend to stay, reflecting **low churn rates among high-tenure groups**.
- **Payment Method:** Customers using **electronic checks** as their payment method exhibit the highest churn rate (**41%**). In contrast, those using other payment methods like bank transfers or credit cards show significantly lower churn rates (**15-20%**).

## Visualizations and Insights
- Various visualizations such as count plots, pie charts, and stacked bar charts provide clear representations of key categorical variables, including gender, senior citizen status, contract type, and payment methods. Each chart is annotated with percentage labels for clarity, ensuring a better understanding of the distribution and churn drivers.
- The stacked bar chart for senior citizens and the histogram of tenure effectively visualize how these factors contribute to higher churn risk, allowing for quick visual comparisons.

## Customer Profiles and Recommendations
- Customers who are **senior citizens**, have **short tenures**, and prefer **month-to-month contracts** or **electronic payment methods** are at the highest risk of churn. These groups should be targeted with personalized retention strategies, such as offering **discounts for long-term contracts** or creating more **senior-friendly service options**.

## Advanced Visualizations
- **Stacked Bar Charts:** Stacked bar charts show the percentage of churn within different categories, such as senior citizen status.
- **Distribution Plots:** Histograms are used to visualize the distribution of key numerical features like MonthlyCharges and TotalCharges, offering insights into the spread of customer data.
- **Boxplot of Tenure vs Churn:** A boxplot illustrates the variation in tenure between customers who churned and those who did not, revealing a clear difference between the two groups.

## Error Handling and Enhancements
- Non-numeric columns are excluded from correlation matrix calculations to prevent errors.
- Legends and labels are adjusted for clarity in the plots, ensuring that visualizations are informative and easy to interpret.

## Summary
This analysis provides a comprehensive understanding of the customer churn dataset by leveraging statistical analysis and visualizations. It highlights important features influencing churn, enabling targeted strategies for customer retention, such as personalized interventions for at-risk customers.

## Technologies Used
- **Python** 🐍
- **Pandas** 📊
- **Seaborn** 📈
- **Matplotlib** 📉

## 📬 Contact

For any questions or feedback, please reach out:

- **Aditya Pathak**
- Email: [adityapathak034@gmail.com](mailto:adityapathak034@gmail.com)
- GitHub: [adityapathak0007](https://github.com/adityapathak0007)
- LinkedIn: [adityapathak07](https://linkedin.com/in/adityapathak07)

