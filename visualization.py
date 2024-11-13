import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("MarvelMovies.csv")
print(data.describe())

#Budget Versus Revenue Analysis
data['profit ($m)']=data['worldwide gross ($m)']-data['budget']
data['ROI']=(data['profit ($m)']/data['budget'])*100

print(data[['movie', 'budget', 'worldwide gross ($m)', 'profit ($m)', 'ROI']].head())

plt.figure(figsize=(8, 5))
plt.hist(data['ROI'], bins=10, color='teal', edgecolor='darkblue')
plt.title("Distribution of ROI")
plt.xlabel("ROI (%)")
plt.ylabel("Frequency")
plt.show()

def get_marker_size(roi):
    if roi <= 200:
        return 30    
    elif roi <= 600:
        return 60    
    elif roi <= 1000:
        return 90     
    else:
        return 120   

data['marker_size'] = data['ROI'].apply(get_marker_size)

categories = data['category'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

plt.figure(figsize=(10, 6))
for i, cat in enumerate(categories):
    cat_data = data[data['category'] == cat]
    plt.scatter(cat_data['budget'], cat_data['worldwide gross ($m)'],
                label=cat, color=colors[i], alpha=0.6, s=cat_data['marker_size'])

plt.title("Budget vs. Worldwide Gross by Category")
plt.xlabel("Budget (in millions)")
plt.ylabel("Worldwide Gross (in millions)")

ROIs = [200, 600, 1000, 1200]
sizes = [get_marker_size(roi) for roi in ROIs]

def get_label(roi):
    if roi <= 200:
        return '<= 200%'   
    elif roi <= 600:
        return '<= 600%'   
    elif roi <= 1000:
        return '<= 1000%'    
    else:
        return '>= 1000%'  

for roi, size in zip(ROIs, sizes):
    plt.scatter([], [], s=size, color='grey', alpha=0.6, label="ROI: "+get_label(roi))

plt.legend(title="Category & ROI Marker Size", loc="upper left")
plt.grid(True)
plt.show()

#Visualizing Budget Recovery
data_sorted_budget = data
data_sorted_budget['% budget recovered'] = data_sorted_budget['% budget recovered'].str.replace('%', '').astype(float) # converting to float from string
data_sorted_budget = data_sorted_budget.sort_values(by='% budget recovered', ascending=True)
movies = data_sorted_budget['movie']
budget_recovery = data_sorted_budget['% budget recovered']
x_ticks = np.arange(0, data_sorted_budget['% budget recovered'].max() + 100, 100)
plt.figure(figsize=(15, 8))
plt.barh(movies, budget_recovery, color='blue')
plt.ylabel("Movie Title")
plt.xlabel("Budget Recovery Percentage")
plt.title("Budget Recovery Percentage of Marvel Movies")
plt.xticks(x_ticks,rotation=45, ha='right')
plt.tight_layout()  
plt.show()

#Comparing Audience & Critic Scores
score_data = data
score_data['audience % score'] = score_data['audience % score'].str.replace('%', '').astype(float)
score_data['critics % score'] = score_data['critics % score'].str.replace('%', '').astype(float)

score_data = data.sort_values(by='audience % score', ascending=True)

indices = np.arange(len(score_data))

plt.figure(figsize=(12, 8))
plt.bar(indices, score_data['audience % score'], width=.4, label="Audience Score", color="magenta")
plt.bar(indices + .4, score_data['critics % score'], width=.4, label="Critics Score", color="teal")


y_ticks=np.arange(0,110,10)
plt.xlabel("Movies")
plt.ylabel("Score (%)")
plt.title("Audience vs. Critics Scores for Marvel Movies")
plt.xticks(indices + .4 / 2, score_data['movie'], rotation=90)
plt.yticks(y_ticks)  
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(score_data['audience % score'], score_data['worldwide gross ($m)'], color="orange", alpha=0.7)
plt.xlabel("Audience Score (%)")
plt.ylabel("Worldwide Gross (in millions)")
plt.title("Audience Score vs Worldwide Gross")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(score_data['critics % score'], score_data['worldwide gross ($m)'], color="red", alpha=0.7)
plt.xlabel("Critics Score (%)")
plt.ylabel("Worldwide Gross (in millions)")
plt.title("Critics Score vs Worldwide Gross")
plt.grid(True)
plt.show()

audience_corr = score_data['audience % score'].corr(score_data['worldwide gross ($m)'])
critics_corr = score_data['critics % score'].corr(score_data['worldwide gross ($m)'])

print(f"Correlation ~ Audience Score and Worldwide Gross: {audience_corr:.3f}")
print(f"Correlation ~ Critics Score and Worldwide Gross: {critics_corr:.3f}")

X = score_data[['critics % score', 'audience % score','budget']]  
y = score_data['worldwide gross ($m)']  
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

plt.figure(figsize=(10, 6))

plt.scatter(y, y_pred, color="blue", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color="red")  

plt.xlabel("Actual Worldwide Gross (in millions)")
plt.ylabel("Predicted Worldwide Gross (in millions)")
plt.title("Actual vs Predicted Worldwide Gross")
plt.grid(True)
plt.show()

#Relationship Between Opening Weekend & Worldwide Gross
plt.figure(figsize=(8, 6))
plt.scatter(data['opening weekend ($m)'], data['worldwide gross ($m)'], color="green", alpha=0.7)
plt.xlabel("Opening Weekend Gross ($m)")
plt.ylabel("Worldwide Gross ($m)")
plt.title("Opening Weekend vs. Worldwide Gross for Marvel Movies")
plt.grid(True)
plt.show()

correlation = data['opening weekend ($m)'].corr(data['worldwide gross ($m)'])
print("Correlation ~ Opening Weekend and Worldwide Gross:", correlation)

X = data[['opening weekend ($m)']]
y = data['worldwide gross ($m)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model=LinearRegression()
model.fit(X_scaled,y)

y_pred = model.predict(X_scaled)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

x_unscaled = scaler.inverse_transform(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(x_unscaled, y, color="blue", alpha=0.6, label="Actual Data")
plt.plot(x_unscaled, y_pred, color="red", linewidth=2, label="Predicted")

plt.xlabel("Opening Weekend Earnings (in millions)")
plt.ylabel("Worldwide Gross (in millions)")
plt.title("Linear Regression: Opening Weekend Earnings vs. Worldwide Gross")
plt.legend()
plt.grid(True)
plt.show()