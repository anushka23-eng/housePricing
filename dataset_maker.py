import pandas as pd

# Create a dictionary with sample data
data = {
    "Area (sqft)": [1500, 2000, 1200, 1800, 2500, 1000, 2200, 1700, 2400, 1300],
    "Bedrooms": [3, 4, 2, 3, 4, 2, 3, 3, 4, 2],
    "Bathrooms": [2, 3, 1, 2, 3, 1, 2, 2, 3, 1],
    "Floors": [1, 2, 1, 2, 2, 1, 2, 1, 3, 1],
    "Year Built": [2005, 2010, 2000, 2015, 2020, 1995, 2018, 2012, 2022, 2008],
    "price": [250000, 350000, 180000, 300000, 450000, 150000, 400000, 290000, 470000, 200000],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("house_price_data.csv", index=False)

print("house_price_data.csv has been created.")
