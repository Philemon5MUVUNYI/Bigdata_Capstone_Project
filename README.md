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

  </body>
</html>
