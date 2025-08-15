SAMPLE CODE

PYTHON CODE
# importing requiry libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

!pip install plotly
import plotly.express as px
import pandas as pd

#loading the dataset
df=pd.read_csv('Ecommerce_data.csv')
df

# Functions
df.head()
df.tail()
df.info()
df.describe()
df['customer_segment'].value_counts()
df['customer_region'].unique()
df.isnull()
df.isnull().sum()
df.notnull()
df.notnull().sum()
df[df.duplicated()]
df.groupby('customer_segment').agg({'profit_per_order':['mean','sum']})
df.sort_values(by='sales_per_order',ascending=False)
df.filter(items=['customer_segment','sales_per_order'])
df['delivery_status']=df['delivery_status'].replace({'Late':'Delayed','On-time':'OnTime'})
df.dropna()
df['cumulative_sales']=df['sales_per_order'].cumsum()
df.count()
df['sales_per_order'].mean()
df['sales_per_order'].median()
df['sales_per_order'].mode()
df['sales_per_order'].min()
df['sales_per_order'].max()
df['sales_per_order'].std()
df['sales_per_order'].var()
df['sales_per_order'].quantile(0.90)
df.pivot_table(index='customer_region',values='profit_per_order',aggfunc='mean')
df2=df[['customer_id','order_id']].copy()
df_merged=df.merge(df2,on='customer_id',suffixes=('','_duplicate'))

# Download a cleaned data set
df.to_csv('clean.csv')

#visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# Plot the monthly sales trend using a line chart
df['order_date'] = pd.to_datetime(df['order_date'], format='mixed', dayfirst=False)
monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['sales_per_order'].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales)
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a boxplot showing the distribution of profit per order across customer segments
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='customer_segment', y='profit_per_order')
plt.title('Profit Distribution by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Profit per Order')
plt.tight_layout()
plt.show()

# Draw a heatmap to visualize the correlation values
df['shipment_delay'] = df['days_for_shipment_real'] - df['days_for_shipment_scheduled']
plt.figure(figsize=(10, 8))
corr = df[['sales_per_order', 'profit_per_order', 'order_item_discount',
           'order_quantity', 'shipment_delay']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap of Key Metrics')
plt.tight_layout()
plt.show()

# Create a horizontal barplot of total sales for the top 10 cities
top_cities = df.groupby('customer_city')['sales_per_order'].sum().largest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='Blues_r')
plt.title('Top 10 Cities by Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('City')
plt.tight_layout()
plt.show()

# Create a pie chart using shipping type counts
shipping_counts = df['shipping_type'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(shipping_counts, labels=shipping_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Order Distribution by Shipping Type')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Create a violin plot to show the distribution of profit per order for each product category
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='category_name', y='profit_per_order', palette='Set3')
plt.title('Profit Distribution by Product Category')
plt.xlabel('Category')
plt.ylabel('Profit per Order')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a histogram for the 'order_quantity' column
plt.figure(figsize=(10, 6))
sns.histplot(df['order_quantity'], bins=30, kde=True, color='teal')
plt.title('Order Quantity Distribution')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Create a pairplot  for selected numeric columns
sns.pairplot(df[['sales_per_order', 'profit_per_order', 'order_item_discount', 'order_quantity']], corner=True)
plt.suptitle('Pairwise Relationships of Key Metrics', y=1.02)
plt.show()
plt.figure(figsize=(8, 5))

# Create a count plot to show how many orders came from each customer segment
sns.countplot(data=df, x='customer_segment', palette='pastel')
plt.title('Order Count by Customer Segment')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("your_file.csv")

# Select features and target
features = ['order_quantity', 'order_item_discount', 
            'days_for_shipment_scheduled', 'days_for_shipment_real']
target = 'sales_per_order'

# Clean data
df_clean = df[features + [target]].dropna()

# Define inputs and output
X = df_clean[features]
y = df_clean[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Linear Regression: Actual vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
