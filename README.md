<html>
  <body>
    <a href="https://colab.research.google.com/drive/1s8hRDZQ71k_WMWmr99lVKWUoq-EMBpQl#scrollTo=9MexvIQyAW_y">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>

## Bigdata_Capstone_Project
# Smart Utility Fraud Detection with Python & Power BI  

---

## Overview
This report presents an analysis of sales data to understand key trends and patterns in sales performance. The dataset includes information on sales transactions, including salesperson details, country, product, date, amount, and boxes shipped. The analysis aims to provide insights into sales distribution, performance metrics, and potential areas for improvement.

---
## Problem statement
> **How can we increase the amount of box shipped and create more jobs at the same time?**

### Objective
The objective of this project is to analyze sales transaction data to uncover key trends, patterns, and insights that can help optimize sales performance, improve inventory management, and enhance decision-making for the business.

#### Key Challenges
1. Understanding Sales Distribution
 - How are sales distributed across different countries, products, and salespersons?
 - Which regions or products contribute the most to revenue?
2. Identifying Sales Trends
 - Are there seasonal or monthly trends in sales?
 - How does the number of boxes shipped correlate with sales revenue?
3. Evaluating Sales Performance.
 - Which salespersons are the top performers?
 - Are there significant differences in sales performance across regions?

---
## 🛠️ Tools & Technologies

- Python, Pandas, NumPy  
- Scikit-learn, XGBoost  
- Google Colab    

---
## 📊 Dataset Information and 🏷️ Sector of Focus

**Sector:** Sales  
**Problem Statement:** (Already Explained) <br>
**Dataset Title:** Sales dataset project
**Number of Dataset:** 1 
**Source Link:** https://chandoo.org/wp/wp-content/upl...
**Number of Rows and Columns:**  test_df ===> (1097, 5)  train_df   <br>
**Data Structure:** Structured (CSV) <br>
**Unstructured (Text, Images):** N/A <br>
**Data Status:** Requires Preprocessing <br>

---

##  Python Analytics Tasks

The Python notebook includes the following components:
- Loading data using panda analysing dataset information
  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')
```
```python
#Load the dataset
df = pd.read_csv('Sales.csv')

print("===DATA===")
df.head(10)

```
<img width="685" height="331" alt="Screenshot 2025-08-03 161532" src="https://github.com/user-attachments/assets/1446d496-5960-4c3f-9c75-ced748c93c1b" />
<img width="723" height="68" alt="Screenshot 2025-08-03 161624" src="https://github.com/user-attachments/assets/8c98f48b-f3b7-4007-bce2-7cb5536f9119" />

```python
#Print section title
print("=== MISSING VALUES ANALYSIS ===")

#Step 1: Count total missing (null) values for each column in the DataFrame
missing_data = df.isnull().sum()

print(missing_data)
```
<img width="269" height="188" alt="Screenshot 2025-08-03 162123" src="https://github.com/user-attachments/assets/5e8685a9-f3a2-4e7a-a569-ee50cde447af" />
```python
df.columns.tolist()
```
<img width="660" height="35" alt="Screenshot 2025-08-03 163145" src="https://github.com/user-attachments/assets/4703ca98-d0c9-4732-a6de-5715f67f7bb1" />

- Countries and boxes shipped
```python
df[['Country','Boxes Shipped']]
```
<img width="262" height="456" alt="Screenshot 2025-08-03 163357" src="https://github.com/user-attachments/assets/1fff9b2f-1191-4969-823c-782e3719f6cf" />

Dataset information

```python
print("DATASET INFO")
df.info()
```
<img width="394" height="274" alt="Screenshot 2025-08-03 163608" src="https://github.com/user-attachments/assets/ff977b14-ea6a-445e-85fd-fa43ea2f90d8" />

- Distribution of boxes shipped by country
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Country', y='Boxes Shipped')
plt.title('Distribution of Boxes Shipped by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
<img width="1157" height="658" alt="Distribution of boxes shipped by country" src="https://github.com/user-attachments/assets/9e7c3753-0973-483d-a628-2b2048774328" />

- Distribution of boxes shipped by month
  
```python
# Convert 'Date' to datetime and extract month
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
df['Month'] = df['Date'].dt.month_name()  # Full month name (e.g., "January")

# Create a boxplot to compare boxes shipped by month
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Month', y='Boxes Shipped', order=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
plt.title('Distribution of Boxes Shipped by Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
<img width="1249" height="608" alt="Distribution of boxes shipped in a month" src="https://github.com/user-attachments/assets/3ff524cc-1eaa-4fd3-8e28-7eb0fd9a516d" />

```python
#Categorical Variables (Country, Product, Sales Person)

# Top 10 Countries by Sales
plt.figure(figsize=(9, 3))
sns.countplot(data=df, y='Country', order=df['Country'].value_counts().index[:3])
plt.title("Top 3 Countries by Number of Orders")
plt.show()

# Top 10 Products Sold
plt.figure(figsize=(9, 3))
sns.countplot(data=df, y='Product', order=df['Product'].value_counts().index[:5])
plt.title("Top 5 Products Sold")
plt.show()
```
<img width="1013" height="341" alt="Top three countries by numbers of order" src="https://github.com/user-attachments/assets/addbd2fc-7367-435f-95a6-b50cec1c32ed" />
<img width="1041" height="347" alt="Top five products sold" src="https://github.com/user-attachments/assets/6796d567-bca3-428c-9eef-b703001b4c05" />7

```python
#Sales Amount vs. Month
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Month', y='Amount', estimator='sum', ci=None, order=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
plt.title("Total Sales ($) by Month")
plt.xticks(rotation=45)
plt.show()
```
<img width="1233" height="672" alt="Total sales ($)by month" src="https://github.com/user-attachments/assets/d0836902-1844-41b8-b160-8b3ca5f52e66" />

```python
#Correlation Heatmap (Numerical Variables)

corr = df[['Boxes Shipped', 'Amount']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Boxes Shipped & Sales Amount")
plt.show()
```
<img width="575" height="475" alt="Screenshot 2025-08-03 164746" src="https://github.com/user-attachments/assets/8cebd2b4-a398-4f81-9348-18d8e994b785" />

```python
#Time Series Analysis
#Monthly Sales Trend

#First, make sure we have Year and Month columns
#If your date is in a column like 'Date' with datetime format, extract Year and Month
#Assuming df has a column 'Date' with datetime values
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

#Now proceed with the groupby operation
monthly_sales = df.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='Month', y='Amount', hue='Year', marker='o')
plt.title("Monthly Sales Trend Over Time")
plt.xticks(rotation=45)
plt.show()
```
<img width="1211" height="626" alt="Screenshot 2025-08-03 164937" src="https://github.com/user-attachments/assets/81374cb1-e584-4392-9566-edc3bef8deff" />

```python
#Advanced Insights

#Top Salespersons by Revenue
top_salespersons = df.groupby('Sales Person')['Amount'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_salespersons.values, y=top_salespersons.index, palette='viridis')
plt.title("Top 10 Salespersons by Revenue ($)")
plt.show()
```
<img width="1242" height="600" alt="Top 10 sales person by amount" src="https://github.com/user-attachments/assets/c033ebc0-be4e-4427-b65a-8814996c0018" />

```python
#Sales by Product Category

product_sales = df.groupby('Product')['Amount'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=product_sales.values, y=product_sales.index, palette='magma')
plt.title("Top 10 Products by Revenue ($)")
plt.show()
```

<img width="1266" height="612" alt="Top 10 products revenue ($)" src="https://github.com/user-attachments/assets/f3918920-ffb5-4e1c-bbe9-d462ebfd29e6" />

# Power bi dashboard and analysis

>An interactive dashboard was developed to visualize the analytical results. Key features include:

- Overview Page: Project context and summary insights
  
<img width="1370" height="734" alt="Screenshot 2025-08-03 165554" src="https://github.com/user-attachments/assets/f46fca88-c5d0-4859-b700-45961afd6202" />

- Visuals: Bar charts, pie charts, line graphs, scatter plots
  
<img width="1370" height="737" alt="image" src="https://github.com/user-attachments/assets/a07c37e8-5a21-4af5-a53f-b032bc2850eb" />
<img width="1368" height="742" alt="image" src="https://github.com/user-attachments/assets/8ce56f56-b207-43e9-ba82-a198f1ad2fc2" />

- Filters,Slicers & DAX formulas: Date ranges, categories, dynamic comparisons

<img width="1386" height="745" alt="image" src="https://github.com/user-attachments/assets/341eb64e-7fc3-4579-ab7f-a4fe060cbb32" />
<img width="1374" height="741" alt="image" src="https://github.com/user-attachments/assets/e6fa47ab-fbd8-43af-8d62-5c92be73ac5e" />


# Data-Driven Recommendations & Insights
> Based on the sales data analysis, here are actionable insights and strategic recommendations to improve business performance:

## 1. Sales Performance Insights
### Key Findings:

- Top-Selling Countries: Australia and India show the highest variability in boxes shipped, indicating fluctuating demand.
- Product Trends: "Peanut Butter Cubes" and "Mint Chip Choco" are among the top-selling products.
- Salesperson Performance: Certain salespersons (e.g., "Jehu Rudeforth") appear frequently in high-value transactions.

### Recommendations:

#### ✅ Focus on High-Demand Regions:

- Allocate more marketing and inventory resources to Australia and India, where sales variability suggests untapped potential.
- Investigate why some countries (e.g., New Zealand) have fewer transactions—are there market entry barriers?

#### ✅ Optimize Product Offerings:

- Promote best-sellers like Peanut Butter Cubes in regions with lower sales.
- Consider bundling or discounts on less popular products to boost demand.

#### ✅ Incentivize Top Salespeople:

- Reward high-performing salespersons (e.g., commission boosts, recognition).
- Analyze their strategies and replicate them across the team.

## 2. Shipping & Logistics Insights
### Key Findings:

1. High Variability in Shipments:
   - Average boxes shipped: 161.80
   - Extreme outliers (e.g., 709 boxes in one order) suggest bulk purchases or inconsistent demand.

2. Potential Inefficiencies: Some transactions have very low shipment volumes (e.g., 1 box), which may not be cost-effective.

### Recommendations:
i. 🚚 Implement Tiered Shipping Policies:
   - Offer discounts for bulk orders (e.g., >200 boxes) to encourage larger shipments.
   - Apply a minimum order requirement to reduce small, unprofitable shipments.

ii. 📦 Improve Inventory Forecasting:
   - Use historical sales trends to predict demand spikes (e.g., seasonal trends).
   - Partner with logistics providers in high-demand regions to reduce delays.

## 3. Pricing & Revenue Optimization
### Key Findings:
 - The "Amount" column had formatting issues (e.g., "$5,320"), requiring cleaning for analysis.
 - No clear correlation yet between price per box and sales volume.

#### Recommendations:
💰 Dynamic Pricing Strategy:
   
   - Test price adjustments on low-demand products to stimulate sales.
   - Offer volume-based discounts to incentivize larger purchases.

📊 Further Analysis Needed:

   - Calculate profit margins per product to identify most lucrative items.
   - Compare pricing strategies across regions to optimize revenue.

## 4. Data Quality & Future Analysis
### Key Findings:

- The dataset is clean (no missing values), but the "Amount" column required preprocessing.
- No time-based trends were analyzed yet (e.g., monthly/quarterly sales).

### Recommendations:
📅 Time-Series Analysis:

- Analyze seasonal trends (e.g., holiday spikes, monthly fluctuations).
- Adjust inventory and promotions based on cyclical demand.

🤖 Predictive Modeling:

- Build a sales forecasting model using historical data.
- Use machine learning to predict future demand per region/product.

## Conclusion
This analysis highlights opportunities to boost sales, optimize logistics, and refine pricing strategies. Next steps include:

1. Deep-dive into time-based trends (seasonality, YoY growth).
2. Implement tiered shipping policies to improve efficiency.
3. Develop a dynamic pricing model to maximize profitability.














  </body>
</html>

