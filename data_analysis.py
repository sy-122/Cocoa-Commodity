import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import seaborn as sns
from meteostat import Point, Daily

#Date
start_date = dt.datetime(2019, 1, 1)
end_date = dt.datetime(2024, 6, 26)

#Download cocoa commodity price from yahoo
cocoa = yf.download('CC', start_date, end_date)


#Graph cocoa price with trend line
cocoa['Close: 30 Mean'] = cocoa['Close'].rolling(window=30).mean()
cocoa_nona = cocoa.dropna(subset=['Close: 30 Mean'])
cocoa_numeric_index = (cocoa_nona.index - cocoa_nona.index[0]).days
trend = np.polyfit(cocoa_numeric_index, cocoa_nona['Close: 30 Mean'], 1)
trendline = np.poly1d(trend)
cocoa['Close: 30 Mean'].plot(figsize=(15,5))
plt.plot(cocoa_nona.index, trendline(cocoa_numeric_index), 'r--')
plt.title('Cocoa Price')
plt.xlabel('Date')
plt.ylabel('Cocoa Price ($)')
plt.show()

#Understand the price for each month
cocoa['Month'] = cocoa.index.month
avg_month_price = cocoa.groupby('Month')['Close'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.boxplot(x='Month', y='Close', data=cocoa, showfliers=False)
sns.lineplot(x=avg_month_price['Month'] - 1, y=avg_month_price['Close'], marker='o', color='red', linestyle='-', linewidth=2, label='Mean')
plt.title('Monthly Cacao Price Distribution with Mean')
plt.xlabel('Month')
plt.ylabel('Cocoa Price ($)')
plt.grid(True)

#Get meteostat weather data
weather_data = {}

#Create point of most well-known coca porudcing regions within top cocas producing countries
cocoa_regions = {
    'san_pedro_ivory_coast': Point(4.7485, -6.6363),
    'kumasi_ghana': Point(6.6666, -1.6163),
    'los_rios_ecuador': Point(-1.0493, -79.6167),
    'buea_cameroon': Point(4.1550, 9.2313),
    'bahia_brazil': Point(-14.7935, -39.0460),
    'san_martin_peru': Point(-6.5000, -76.3667)
}

#Get weather data for each regions
for region_name, region_point in cocoa_regions.items():
    region_data = Daily(region_point, start_date, end_date)
    region_data = region_data.fetch()
    weather_data[region_name] = region_data

fig,ax = plt.subplots(dpi=150, figsize=(15,5))

#Plot weather data
for region_name, data in weather_data.items():
    rolling_mean = data['tavg'].rolling(window=7).mean()
    ax.plot(data.index, rolling_mean, label=region_name)

ax.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###Understand Correlation###
import statsmodels.api as sm

#Combine Temperature Data
all_temps = pd.concat([df['tavg'] for df in weather_data.values()], axis=1)
overall_avg_temp = all_temps.mean(axis=1)
combined_df = pd.merge(overall_avg_temp.to_frame(name='overall_tavg'),
                       cocoa, left_index=True, right_index=True)

# US Dollar Index Data
us_fx = yf.download('DX-Y.NYB', start=start_date, end=end_date)['Close']
us_fx.name = 'US_Dollar_Index'
combined_df = pd.merge(combined_df, us_fx, left_index=True, right_index=True)

# Download Sugar Price Data
sugar = yf.download('SB=F', start=start_date, end=end_date)['Close']
sugar.name = 'Sugar_Price'
combined_df = pd.merge(combined_df, sugar, left_index=True, right_index=True)

combined_df.to_csv('cocoa_combined.csv', index=False)

# Standardize the predictors
X_standardized = (combined_df[['overall_tavg', 'US_Dollar_Index', 'Sugar_Price']] - combined_df[['overall_tavg', 'US_Dollar_Index', 'Sugar_Price']].mean()) / combined_df[['overall_tavg', 'US_Dollar_Index', 'Sugar_Price']].std()

# Add a constant term
X_standardized = sm.add_constant(X_standardized)

# Fit the regression model using standardized predictors
model = sm.OLS(Y, X_standardized).fit()

# Print the summary of the standardized model
print(model.summary())

