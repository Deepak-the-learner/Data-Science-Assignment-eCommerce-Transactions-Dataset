import pandas as pd
import matplotlib.pyplot as plt
import os

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

customers = pd.read_csv('datasets/Customers.csv')
products = pd.read_csv('datasets/Products.csv')
transactions = pd.read_csv('datasets/Transactions.csv')

merged_data = pd.merge(transactions, customers, on='CustomerID')
merged_data = pd.merge(merged_data, products, on='ProductID')

region_sales = merged_data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar', title='Sales by Region', color='skyblue')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.savefig(f"{output_dir}/region_sales.png")
plt.close()

category_sales = merged_data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
category_sales.plot(kind='bar', title='Sales by Category', color='green')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.savefig(f"{output_dir}/category_sales.png")
plt.close()

top_products = merged_data.groupby('ProductName')['TotalValue'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='bar', title='Top 10 Products', color='orange')
plt.xlabel('Product Name')
plt.ylabel('Total Sales')
plt.savefig(f"{output_dir}/top_products.png")
plt.close()

merged_data['TransactionDate'] = pd.to_datetime(merged_data['TransactionDate'])
sales_by_date = merged_data.groupby(merged_data['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
sales_by_date.plot(kind='line', title='Sales Over Time', color='blue')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.savefig(f"{output_dir}/sales_over_time.png")
plt.close()

customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
signups_by_month = customers.groupby(customers['SignupDate'].dt.to_period('M'))['CustomerID'].count()
signups_by_month.plot(kind='line', title='New Customer Signups', color='purple')
plt.xlabel('Month')
plt.ylabel('Number of Signups')
plt.savefig(f"{output_dir}/signups_over_time.png")
plt.close()

top_customers = merged_data.groupby('CustomerName')['TotalValue'].sum().sort_values(ascending=False).head(10)
top_customers.plot(kind='bar', title='Top 10 Customers', color='red')
plt.xlabel('Customer Name')
plt.ylabel('Total Sales')
plt.savefig(f"{output_dir}/top_customers.png")
plt.close()
