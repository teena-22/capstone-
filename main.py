
# Part 1 : Importing and loading data

import numpy as np
import csv

# Read the file
with open("freshflow_sales.csv", "r") as file:

    header = file.readline().strip().split(",")    # columns name

    rows = np.loadtxt(file, delimiter = ",")    # read remaining numerical rows

# convert data into a numpy array

data = np.array(rows, dtype=float)
print("Dataset loaded")

# Find shape, rows, and columns
print("Shape:", data.shape)
print("Rows:", data.shape[0])  # 0 means rows
print("Columns:", data.shape[1])  # 1 means columns

# Find dimensions
print("Dimensions:", data.ndim)

# Find datatype
print("Data Type:", data.dtype)

# Read start 5 rows
print(data[:5])

# Columns names
print(header)

# Part 2 : Checking missing values and data statistics

# Find missing values
for i, column in enumerate(header):
    missing_count = np.sum(np.isnan(data[:,i]))
    print(column, ":", missing_count)

# Find unique values
for i, column in enumerate(header):
    unique_values = np.unique(data[:,i])
    print(column, ":", len(unique_values))

# Calculate statistical information
for i, column in enumerate(header):
    print("\nColumn:", column)
    print("Minimum:", np.nanmin(data[:,i]))
    print("Maximum:", np.nanmax(data[:,i]))
    print("Mean:", np.nanmean(data[:,i]))
    print("Median:", np.nanmedian(data[:,i]))
    print("Standard Deviation:", np.nanstd(data[:,i]))

# Part 3 : Data Validation

# Find duplicate transaction IDs
transaction_ids = data[:,0]   # : means rows (all), 0 is column index no

unique_ids, count = np.unique(transaction_ids, return_counts=True)
print("Unique ids:", unique_ids)
print(count)

duplicate_ids = unique_ids[count > 1]
print("Duplicate Transaction IDs:", duplicate_ids)
print("Number of duplicate IDs:", len(duplicate_ids))

# Find invalid outlet IDs
outlet_ids = data[:,2]   # 2 is 3rd column (index number)

invalid_outlets = (outlet_ids < 1) | (outlet_ids > 3)

print("Invalid Outlets:", np.sum(invalid_outlets))

# Find invalid quantities
quantity = data[:,5]
invalid_quantity = quantity <= 0
print("Invalid Quantity:", np.sum(invalid_quantity))

# Find invalid unit prices
unit_prices = data[:,6]
invalid_unit_prices = (unit_prices <= 0) & (~np.isnan(unit_prices))
print("Invalid Unit Prices:", np.sum(invalid_unit_prices))

# Find invalid unit costs
unit_cost = data[:,7]
invalid_unit_costs = (unit_cost <= 0) & (~np.isnan(unit_cost))
print("Invalid Unit Costs:", np.sum(invalid_unit_costs))

# Find invalid discount rates
discount_rate = data[:,8]
invalid_discount_rates = (discount_rate < 0) | (discount_rate > 1)
print("Invalid Discount Rates:", np.sum(invalid_discount_rates))

# Find invalid customer ratings
customer_rating = data[:,11]
invalid_customer_ratings = ((customer_rating < 1) | (customer_rating > 5)) & (~np.isnan(customer_rating))
print("Invalid Customer Ratings:", np.sum(invalid_customer_ratings))

# Check stock records
stock = data[:,9]
invalid_negative_stock = stock < 0
zero_stock = stock == 0
print("Negative Stock:", np.sum(invalid_negative_stock))
print("Zero Stock:", np.sum(zero_stock))

# Select product and category codes
product_codes = data[:,3]

category_codes = data[:,4]

## Create product-category mapping
product_category_mapping = {
    1:1,
    2:1,
    3:1,
    4:1,
    5:2,
    6:2,
    7:3,
    8:4
}

expected_category = np.array([product_category_mapping[product_code] for product_code in product_codes])      #list comprehension

## Find category mismatches
category_mismatch = category_codes != expected_category
print("Category Mismatch:", np.sum(category_mismatch))

# Display mismatched records
mismatched_records = data[category_mismatch]
print(mismatched_records)

# Display specific information about incorrect records
print("Transaction IDs:", mismatched_records[:,0])
print("Product codes:", mismatched_records[:,3])
print("Actual category:", mismatched_records[:,4])   # categories currently present in the dataset
print("Expected category:", expected_category[category_mismatch])     # what the categories should have been according mapping.

# Payment validation and missing values

# Check invalid payment
payment_code = data[:,10]
invalid_payment_codes = (payment_code < 1) | (payment_code > 4)
print("Invalid Payment Codes:", np.sum(invalid_payment_codes))

# Check invalid day records
day_index = data[:,1]
invalid_day_index = (day_index < 1) | (day_index > 30)
print("Invalid Day Index:", np.sum(invalid_day_index))

# Check missing values again
for i, column in enumerate(header):
    missing_count = np.sum(np.isnan(data[:,i]))
    print(column, ":", missing_count)

# Calculate total missing values
total_missing = np.sum(np.isnan(data))
print("Total Missing:", total_missing)

# Create a copy of the dataset
clean_data = data.copy()

# Find records with missing unit prices
unit_price_index = header.index("unit_price")
missing_price_mask =  np.isnan(data[:,unit_price_index])
missing_price_records = data[missing_price_mask]
print("Records with missing price:")
print(missing_price_records)

# Find records with missing unit costs
unit_cost_index = header.index("unit_cost")
missing_cost_mask = np.isnan(data[:, unit_cost_index])
missing_cost_records = data[missing_cost_mask]
print("Records with missing unit costs:")
print(missing_cost_records)

# Find records with missing customer ratings
rating_index = header.index("customer_rating")
missing_rating_mask = np.isnan(data[:, rating_index])
missing_rating_records = data[missing_rating_mask]
print("Number of missing ratings:", len(missing_rating_records))
print(missing_rating_records)

# Create clean dataset again
clean_data = data.copy()

print("Original shape:", data.shape)
print("Clean data shape:", clean_data.shape)

# Check category mismatches before correction
product_code = clean_data[:, 3]
category_code = clean_data[:, 4]

expected_category = np.array([product_category_mapping[product] for product in product_code])

category_mismatch = category_code != expected_category

print("Mismatches before correction:", np.sum(category_mismatch))

# After correction category mismatches
clean_data[category_mismatch, 4] = expected_category[category_mismatch]

print("Mismatches after correction:", np.sum(clean_data[:, 4] != expected_category))

# Find invalid outlet records
outlet_ids = clean_data[:, 2]

invalid_outlet_mask = (outlet_ids < 1) | (outlet_ids > 3)

invalid_outlet_records = clean_data[invalid_outlet_mask]

print("Invalid outlet records:")
print(invalid_outlet_records)

# Count invalid outlet records
print("Invalid outlet records:",
      np.sum((clean_data[:, 2] < 1) | (clean_data[:, 2] > 3)))

# Find invalid quantity records
quantity = clean_data[:, 5]

invalid_quantity_mask = quantity <= 0

invalid_quantity_records = clean_data[invalid_quantity_mask]

print("Invalid quantity count:", np.sum(invalid_quantity_mask))

# Find invalid unit price records
unit_price = clean_data[:, 6]

invalid_price_mask = (unit_price <= 0) & (~np.isnan(unit_price))

invalid_price_records = clean_data[invalid_price_mask]

print("Invalid unit price records:")
print(invalid_price_records)

# Count unit prices
print("Invalid unit prices:", np.sum(invalid_price_mask))

# PART 5: Invalid costs, discounts, ratings and high prices
# Invalid unit cost records
unit_cost = clean_data[:, 7]

invalid_cost_mask = (unit_cost <= 0) & (~np.isnan(unit_cost))

invalid_cost_records = clean_data[invalid_cost_mask]

print("Invalid unit cost records:")
print(invalid_cost_records)

# Count invalid costs
print("Invalid unit costs:", np.sum(invalid_cost_mask))

# Invalid discount records
discount = clean_data[:, 8]

invalid_discount_mask = (discount < 0) | (discount > 1)

invalid_discount_records = clean_data[invalid_discount_mask]

print("Invalid discount records:")
print(invalid_discount_records)

# Invalid discount count
print("Invalid discount records:", np.sum(invalid_discount_mask))

# Invalid customer ratings
rating = clean_data[:, 11]

invalid_rating_mask = (((rating < 1) | (rating > 5)) & (~np.isnan(rating)))

invalid_rating_records = clean_data[invalid_rating_mask]

print("Invalid rating records:")
print(invalid_rating_records)

# Invalid ratings count
print("Invalid customer ratings:", np.sum(invalid_rating_mask))

# Unusally high unit prices
unit_price = clean_data[:, 6]

high_price_mask = unit_price > 200

high_price_records = clean_data[high_price_mask]

print("Unusually high price records:")
print(high_price_records)
print("Number of records:", np.sum(high_price_mask))

# Highest and top 10 prices
valid_prices = unit_price[~np.isnan(unit_price)]

print("Highest unit price:", np.max(valid_prices))
print("Top 10 unit prices:", np.sort(valid_prices)[-10:][::-1])

# Checking product wise
product_codes = clean_data[:, 3]

for product in np.unique(product_codes):
    prices = unit_price[product_codes == product]
    prices = prices[~np.isnan(prices)]

    print(
        "Product:", int(product),
        "| Minimum:", np.min(prices),
        "| Median:", np.median(prices),
        "| Maximum:", np.max(prices)
    )

# Suspicious products check
suspicious_products = [4, 7]

for product in suspicious_products:
    mask = ((clean_data[:, 3] == product) & (clean_data[:, 6] > 100))

    print("\nProduct:", product)
    print(clean_data[mask][:, [0, 3, 6]])

# High price transaction IDs
high_price_ids = [101468, 102005, 103950]

high_price_anomaly_mask = np.isin(clean_data[:, 0], high_price_ids)

print("High-price anomalies flagged:", np.sum(high_price_anomaly_mask))

# Discount values and its counting
discount = clean_data[:, 8]

unique_discounts, discount_counts = np.unique(discount, return_counts=True)

for value, count in zip(unique_discounts, discount_counts):
    print("Discount:", value, "| Count:", count)

# High discount records
discount = clean_data[:, 8]

high_discount_mask = (discount > 0.25) & (discount <= 1)

high_discount_records = clean_data[high_discount_mask]

print("High discount records:")
print(high_discount_records)

print("Number of high discounts:", np.sum(high_discount_mask))

# Cleaning summary
cleaning_summary = {
    "Product-category mismatches": 4,
    "Invalid outlet IDs": 2,
    "Invalid quantities": 4,
    "Invalid unit prices": 2,
    "Invalid unit costs": 2,
    "Invalid discounts": 2,
    "Invalid customer ratings": 3,
    "High unit-price anomalies": 3,
    "High discount anomalies": 5
}

for issue, count in cleaning_summary.items():
    print(issue, ":", count)

# PART 6: Select valid financial data
valid_transaction_mask = (
    (clean_data[:, 5] > 0) &
    (clean_data[:, 6] > 0) &
    (clean_data[:, 7] > 0) &
    (clean_data[:, 8] >= 0) &
    (clean_data[:, 8] <= 1)
)

print("Valid financial transactions:", np.sum(valid_transaction_mask))

print("Invalid financial transactions:", np.sum(~valid_transaction_mask))

# Missing prices and costs
missing_price_mask = np.isnan(clean_data[:, 6])
missing_cost_mask = np.isnan(clean_data[:, 7])

print("Missing unit prices:", np.sum(missing_price_mask))
print("Missing unit costs:", np.sum(missing_cost_mask))

# Valid financial records
valid_financial_mask = (
    (clean_data[:, 5] > 0) &
    (clean_data[:, 6] > 0) &
    (clean_data[:, 7] > 0) &
    (clean_data[:, 8] >= 0) &
    (clean_data[:, 8] <= 1)
)

print("Valid financial records:",np.sum(valid_financial_mask))

# Create financial data
financial_data = clean_data[valid_transaction_mask].copy()

print("Original records:", len(data))
print("Financial analysis records:", len(financial_data))
print("Financial dataset shape:", financial_data.shape)

# PART 7: Financial calculations - Gross sales, Discount, Net revenue, COGS and profit
quantity = financial_data[:, 5]
unit_price = financial_data[:, 6]

gross_sales = quantity * unit_price

print("Total Gross Sales:", np.sum(gross_sales))

# Total Discount Amount
discount_rate = financial_data[:, 8]

discount_amount = gross_sales * discount_rate

print("Total Discount Amount:", np.sum(discount_amount))

# Net revenue
net_revenue = gross_sales - discount_amount

print("Total Net Revenue:", round(np.sum(net_revenue),2))

# COGS(Cost of Goods Sold)
quantity = financial_data[:, 5]
unit_cost = financial_data[:, 7]

cogs = quantity * unit_cost

print("Total COGS:", np.sum(cogs))

# Total gross profit
gross_profit = net_revenue - cogs

print("Total Gross Profit:", round(np.sum(gross_profit),2))

# Gross profit margin
gross_margin = (gross_profit / net_revenue) * 100

print("Overall Gross Profit Margin:", round(np.mean(gross_margin),2), "%")

# Weighted Overall Gross Profit Margin
overall_gross_margin = (np.sum(gross_profit) / np.sum(net_revenue)) * 100

print("Overall Gross Profit Margin:", round(overall_gross_margin,2), "%")

# Average transaction value
average_transaction_value = np.mean(net_revenue)

print("Average Transaction Value:", round(average_transaction_value,2))

# Average quantity per transaction
average_quantity = np.mean(financial_data[:, 5])

print("Average Quantity per Transaction:", round(average_quantity,2))

# Total Units Sold
total_units_sold = np.sum(financial_data[:, 5])

print("Total Units Sold:", total_units_sold)

# Average Selling Price Per Unit
average_selling_price = np.sum(net_revenue) / total_units_sold

print("Average Selling Price per Unit:", round(average_selling_price,2))

# Average cost per unit
average_cost_per_unit = np.sum(cogs) / total_units_sold

print("Average Cost per Unit:", round(average_cost_per_unit,2))

# Gross profit per unit
profit_per_unit = average_selling_price - average_cost_per_unit

print("Gross Profit per Unit:", profit_per_unit)

# Overall disount percentage
total_gross_sales = np.sum(gross_sales)
total_discount = np.sum(discount_amount)

discount_percentage = (total_discount / total_gross_sales) * 100

print("Overall Discount Percentage:", round(discount_percentage,2), "%")

# PART 8: Product-wise analysis
product_codes = np.unique(financial_data[:, 3])

for product in product_codes:
    product_mask = financial_data[:, 3] == product
    product_revenue = np.sum(net_revenue[product_mask])

    print("Product:", int(product),
          "| Net Revenue:", round(product_revenue, 2))

# Highest revenue product
product_revenues = []

for product in product_codes:
    product_mask = financial_data[:, 3] == product
    product_revenue = np.sum(net_revenue[product_mask])
    product_revenues.append(product_revenue)

product_revenues = np.array(product_revenues)

best_index = np.argmax(product_revenues)

print("Best Product Code:", int(product_codes[best_index]))
print("Highest Net Revenue:", round(product_revenues[best_index], 2))

# Product wise total units sold
for product in product_codes:
    product_mask = financial_data[:, 3] == product

    total_quantity = np.sum(financial_data[product_mask, 5])

    print("Product:", int(product),
          "| Total Units Sold:", int(total_quantity))

# Best selling product
product_quantities = []

for product in product_codes:
    product_mask = financial_data[:, 3] == product

    total_quantity = np.sum(
        financial_data[product_mask, 5]
    )

    product_quantities.append(total_quantity)

product_quantities = np.array(product_quantities)

best_quantity_index = np.argmax(product_quantities)

print("Best-Selling Product:", int(product_codes[best_quantity_index]))
print("Total Units Sold:", int(product_quantities[best_quantity_index]))

# Product-wise Gross Profit
for product in product_codes:
    product_mask = financial_data[:, 3] == product

    product_profit = np.sum(gross_profit[product_mask])

    print("Product:", int(product),
          "| Gross Profit:", round(product_profit, 2))

# Most profitable product
product_profits = []

for product in product_codes:
    product_mask = financial_data[:, 3] == product
    product_profit = np.sum(gross_profit[product_mask])

    product_profits.append(product_profit)

product_profits = np.array(product_profits)

best_profit_index = np.argmax(product_profits)

print("Most Profitable Product:", int(product_codes[best_profit_index]))
print("Highest Gross Profit:", round(product_profits[best_profit_index], 2))

# Product wise gross margin
for product in product_codes:
    product_mask = financial_data[:, 3] == product

    product_revenue = np.sum(net_revenue[product_mask])
    product_profit = np.sum(gross_profit[product_mask])

    product_margin = (product_profit / product_revenue) * 100

    print("Product:", int(product),
          "| Gross Margin:", round(product_margin, 2), "%")

# Highest margin product
product_margins = []

for product in product_codes:
    product_mask = financial_data[:, 3] == product

    product_revenue = np.sum(net_revenue[product_mask])
    product_profit = np.sum(gross_profit[product_mask])

    product_margin = (product_profit / product_revenue) * 100

    product_margins.append(product_margin)

product_margins = np.array(product_margins)

best_margin_index = np.argmax(product_margins)

print("Highest Margin Product:", int(product_codes[best_margin_index]))
print("Highest Gross Margin:", round(product_margins[best_margin_index], 2), "%")

# PART 9: Outlet wise analysis
# Outlet wise net revenue
outlet_codes = np.unique(financial_data[:, 2])

for outlet in outlet_codes:
    outlet_mask = financial_data[:, 2] == outlet

    outlet_revenue = np.sum(net_revenue[outlet_mask])

    print("Outlet:", int(outlet),
          "| Net Revenue:", round(outlet_revenue, 2))

# Valid outlet records
valid_outlet_mask = (
    (financial_data[:, 2] >= 1) &
    (financial_data[:, 2] <= 3)
)

valid_outlet_data = financial_data[valid_outlet_mask]

valid_outlet_revenue = net_revenue[valid_outlet_mask]

print("Valid Outlet Records:", len(valid_outlet_data))
print("Valid Outlet Revenue:",  round(np.sum(valid_outlet_revenue),2))

# Best revenue outlet
outlet_codes = np.array([1, 2, 3])

outlet_revenues = []

for outlet in outlet_codes:
    outlet_mask = valid_outlet_data[:, 2] == outlet

    revenue = np.sum(valid_outlet_revenue[outlet_mask])

    outlet_revenues.append(revenue)

outlet_revenues = np.array(outlet_revenues)

best_outlet_index = np.argmax(outlet_revenues)

print("Best Outlet:", int(outlet_codes[best_outlet_index]))
print("Highest Revenue:", round(outlet_revenues[best_outlet_index], 2))

# Outlet wise gross profit
for outlet in outlet_codes:
    outlet_mask = valid_outlet_data[:, 2] == outlet

    outlet_profit = np.sum(
        gross_profit[valid_outlet_mask][outlet_mask]
    )

    print("Outlet:", int(outlet),
          "| Gross Profit:", round(outlet_profit, 2))

# Oulet wise gross margin
for outlet in outlet_codes:
    outlet_mask = valid_outlet_data[:, 2] == outlet

    outlet_revenue = np.sum(
        valid_outlet_revenue[outlet_mask]
    )

    outlet_profit = np.sum(
        gross_profit[valid_outlet_mask][outlet_mask]
    )

    outlet_margin = (outlet_profit / outlet_revenue) * 100

    print("Outlet:", int(outlet),
          "| Gross Margin:", round(outlet_margin, 2), "%")

# Outlet wise total units sold
for outlet in outlet_codes:
    outlet_mask = valid_outlet_data[:, 2] == outlet

    total_quantity = np.sum(
        valid_outlet_data[outlet_mask, 5]
    )

    print("Outlet:", int(outlet),
          "| Total Units Sold:", int(total_quantity))

# Best selling outlet
outlet_quantities = []

for outlet in outlet_codes:
    outlet_mask = valid_outlet_data[:, 2] == outlet

    total_quantity = np.sum(
        valid_outlet_data[outlet_mask, 5]
    )

    outlet_quantities.append(total_quantity)

outlet_quantities = np.array(outlet_quantities)

best_quantity_index = np.argmax(outlet_quantities)

print("Best-Selling Outlet:", int(outlet_codes[best_quantity_index]))
print("Total Units Sold:", int(outlet_quantities[best_quantity_index]))


day_indices = np.unique(financial_data[:, 1])

for day in day_indices:
    day_mask = financial_data[:, 1] == day

    daily_revenue = np.sum(net_revenue[day_mask])

    print("Day:", int(day),
          "| Net Revenue:", round(daily_revenue, 2))

daily_revenues = []

for day in day_indices:
    day_mask
