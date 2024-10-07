#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
banana = pd.read_csv('https://github.com/yumayuma/retl603/raw/main/banana.csv')
banana.head()


# In[99]:


#Question 1
sns.regplot(x="price_kg", y="sales_kkg", data=banana, line_kws={'color':
'red'}, ci=95, x_jitter=0.1)


# In[41]:


#Question 2
import statsmodels.api as sm

# Define the dependent variable (DV) and predictor variable
y = banana['sales_kkg']
X = banana['price_kg']

# Add a constant term to the predictor variable
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression analysis
print(model.summary())


# In[6]:


#Q3
# Create dummy variables for the 'retailer' categorical variable
dummy_retailer = pd.get_dummies(banana['retailer'], drop_first=True)

# Combine the dummy variables with the existing data
X = pd.concat([X, dummy_retailer], axis=1)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression analysis
print(model.summary())



# In[51]:


import numpy as np
import matplotlib.pyplot as plt

# Define the range of price_kg values
z = np.arange(start=1.0, stop=3.0, step=0.1)

# Coefficients from your regression model
intercept_express = 4.7241
price_coeff_express = -1.3754
intercept_grocery = 4.7241 + 9.9333
price_coeff_grocery = -1.3754

# Calculate the sales predictions for 'express' and 'grocery'
sales_express = intercept_express + price_coeff_express * z
sales_grocery = intercept_grocery + price_coeff_grocery * z

# Create separate plots for 'express' and 'grocery'
plt.plot(z, sales_express, 'b-', label='express')
plt.plot(z, sales_grocery, 'r-', label='grocery')

# Add labels and legend
plt.xlabel('Price per Kilogram')
plt.ylabel('Sales in Thousands of Kilograms')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[49]:


#Q4 
model = sm.OLS.from_formula('sales_kkg ~ price_kg * C(retailer, Treatment(reference="express"))', banana).fit()
print(model.summary())



# In[53]:


import numpy as np
import matplotlib.pyplot as plt

# Define the range of price_kg values
z = np.arange(start=1.0, stop=3.0, step=0.1)

# Coefficients from your updated regression model
intercept_express = 3.8080
intercept_grocery = 3.8080 + 14.0440
price_coeff_express = -0.8629
price_coeff_grocery = -0.8629 - 2.5285

# Calculate the sales predictions for 'express' and 'grocery'
sales_express = intercept_express + price_coeff_express * z
sales_grocery = intercept_grocery + price_coeff_grocery * z

# Create separate plots for 'express' and 'grocery'
plt.plot(z, sales_express, 'b-', label='express')
plt.plot(z, sales_grocery, 'r-', label='grocery')

# Add labels and legend
plt.xlabel('Price per Kilogram')
plt.ylabel('Sales in Thousands of Kilograms')
plt.legend(loc='upper left')

# Show the plot
plt.show()


# In[54]:


#Q5
express = banana[ (banana['retailer'] =='express') ]
model = sm.OLS.from_formula('sales_kkg ~ price_kg*organic', express).fit()
print(model.summary())


# In[58]:


z =  np.arange(start=2, stop=6, step=1)
plt.plot(z, 3.5288 - 0.5448 * z, 'b-', label='regular')
plt.plot(z, 3.5288 - 1.9522 + (-0.5448 + 0.6434) * z, 'r-', label='organic')
plt.legend(loc="upper left")
plt.show()


# In[59]:


#Q6
# Define the price and organic indicators
price = 2
organic_yes = 1
organic_no = 0

# Calculate the forecast for organic bananas
sales_organic = 3.5288 - 1.9522 * organic_yes - 0.5448 * price + 0.6434 * price * organic_yes

# Calculate the forecast for regular bananas
sales_regular = 3.5288 - 1.9522 * organic_no - 0.5448 * price + 0.6434 * price * organic_no

print("Forecast for Organic Bananas at $2 price:", sales_organic)
print("Forecast for Regular Bananas at $2 price:", sales_regular)


# In[60]:


#Q7
import matplotlib.pyplot as plt
import seaborn as sns

# Filter the data for non-organic bananas in express stores
express_non_organic = banana[(banana['retailer'] == 'express') & (banana['organic'] == 'No')]

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='week', y='sales_kkg', data=express_non_organic, label='Non-Organic - Express', color='blue', marker='o')

# Add labels and title
plt.xlabel('Week')
plt.ylabel('Sales in Thousands of Kilograms')
plt.title('Relationship between Non-Organic Banana Sales and Week in Express Stores')

# Show the plot
plt.show()


# In[62]:


import statsmodels.api as sm

# Filter the data for non-organic bananas in express stores
express_non_organic = banana[(banana['retailer'] == 'express') & (banana['organic'] == 'No')]

# Define the dependent variable (DV) and predictor variable
y = express_non_organic['sales_kkg']
X = express_non_organic['week']

# Add a constant term to the predictor variable
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression analysis
print(model.summary())


# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of sales and week
sns.scatterplot(data=express_non_organic, x='week', y='sales_kkg', label='Sales - Non-Organic', color='blue')

# Linear regression line
plt.plot(express_non_organic['week'], model.predict(X), color='red', label='Linear Regression')

# Labels and title
plt.xlabel('Week')
plt.ylabel('Sales (in 1000 kg)')
plt.title('Relationship between Sales and Week for Non-Organic Bananas (Express Stores)')
plt.legend()

# Show the plot
plt.show()


# In[64]:


#Q8
import statsmodels.api as sm

# Define the dependent variable (DV) and predictor variable
y = express_non_organic['sales_kkg']
X = express_non_organic[['price_kg', 'week']]

# Initialize a list to store the model summaries
model_summaries = []

# Fit polynomial regression models from degree 1 to 5
for degree in range(1, 6):
    # Create polynomial features
    X_poly = X.copy()
    for i in range(2, degree + 1):
        X_poly[f'week^{i}'] = X['week'] ** i

    # Add a constant term
    X_poly = sm.add_constant(X_poly)

    # Fit the polynomial regression model
    model = sm.OLS(y, X_poly).fit()
    model_summaries.append(model.summary())

# Print the model summaries
for degree, summary in enumerate(model_summaries, start=1):
    print(f"Polynomial Degree {degree} Model:")
    print(summary)
    print("-" * 50)


# In[ ]:


###Here are the summaries of the polynomial regression models for non-organic banana sales in express stores with different degrees of polynomials:

**Polynomial Degree 1 Model:**
- R-squared: 0.315
- Adj. R-squared: 0.288
- AIC: 13.76
- BIC: 19.61

**Polynomial Degree 2 Model:**
- R-squared: 0.389
- Adj. R-squared: 0.351
- AIC: 9.874
- BIC: 17.68

**Polynomial Degree 3 Model:**
- R-squared: 0.426
- Adj. R-squared: 0.377
- AIC: 8.632
- BIC: 18.39

**Polynomial Degree 4 Model:**
- R-squared: 0.489
- Adj. R-squared: 0.433
- AIC: 4.590
- BIC: 16.30

**Polynomial Degree 5 Model:**
- R-squared: 0.775
- Adj. R-squared: 0.745
- AIC: -36.13
- BIC: -22.47

The Polynomial Degree 5 Model has the highest R-squared value (0.775), indicating that it explains the most variation in the data, and the lowest AIC and BIC values. This model is the best among the polynomial models. It suggests a high-degree polynomial relationship between sales, price, and week.

The choice of the best model should also consider the practical implications and the trade-off between model complexity and fit. In this case, the Polynomial Degree 5 Model provides the best fit to the data based on R-squared and information criteria.


# In[93]:


#Q9
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=603)
y_true, y_pred = [], []
for train_index, test_index in kf.split(express_non_organic):
    train, val = express_non_organic.iloc[train_index], express_non_organic.iloc[test_index]

    model = sm.OLS.from_formula('sales_kkg ~ price_kg + week ', train).fit()

    if len(y_true) == 0:
        print(model.summary())

    y_pred_val = model.predict(val)
    y_true.extend(val['sales_kkg'].tolist())
    y_pred.extend(y_pred_val.tolist())


mse = mean_squared_error(y_true, y_pred)
print('5-fold CV Mean Squared Error:', mse)


# In[95]:


kf = KFold(n_splits=5, shuffle=True, random_state=603)
y_true, y_pred = [], []

for train_index, test_index in kf.split(express_non_organic):
    train, val = express_non_organic.iloc[train_index], express_non_organic.iloc[test_index]

    model = sm.OLS.from_formula('sales_kkg ~ price_kg + week + I(week**2)', train).fit()

    if len(y_true) == 0:
        print(model.summary())

    y_pred_val = model.predict(val)
    y_true.extend(val['sales_kkg'].tolist())
    y_pred.extend(y_pred_val.tolist())


mse = mean_squared_error(y_true, y_pred)
print('5-fold CV Mean Squared Error:', mse)


# In[98]:


kf = KFold(n_splits=5, shuffle=True, random_state=603)
y_true, y_pred = [], []

for train_index, test_index in kf.split(express_non_organic):
    train, val = express_non_organic.iloc[train_index], express_non_organic.iloc[test_index]

    model = sm.OLS.from_formula('sales_kkg ~ price_kg + week + I(week**2) + I(week**3)', train).fit()

    if len(y_true) == 0:
        print(model.summary())

    y_pred_val = model.predict(val)
    y_true.extend(val['sales_kkg'].tolist())
    y_pred.extend(y_pred_val.tolist())


mse = mean_squared_error(y_true, y_pred)
print('5-fold CV Mean Squared Error:', mse)


# In[96]:


kf = KFold(n_splits=5, shuffle=True, random_state=603)
y_true, y_pred = [], []

for train_index, test_index in kf.split(express_non_organic):
    train, val = express_non_organic.iloc[train_index], express_non_organic.iloc[test_index]

    model = sm.OLS.from_formula('sales_kkg ~ price_kg + week + I(week**2) + I(week**3) + I(week**4)', train).fit()

    if len(y_true) == 0:
        print(model.summary())

    y_pred_val = model.predict(val)
    y_true.extend(val['sales_kkg'].tolist())
    y_pred.extend(y_pred_val.tolist())


mse = mean_squared_error(y_true, y_pred)
print('5-fold CV Mean Squared Error:', mse)


# In[97]:


kf = KFold(n_splits=5, shuffle=True, random_state=603)
y_true, y_pred = [], []

for train_index, test_index in kf.split(express_non_organic):
    train, val = express_non_organic.iloc[train_index], express_non_organic.iloc[test_index]

    model = sm.OLS.from_formula('sales_kkg ~ price_kg + week + I(week**2) + I(week**3) + I(week**4) + I(week**5)', train).fit()

    if len(y_true) == 0:
        print(model.summary())

    y_pred_val = model.predict(val)
    y_true.extend(val['sales_kkg'].tolist())
    y_pred.extend(y_pred_val.tolist())


mse = mean_squared_error(y_true, y_pred)
print('5-fold CV Mean Squared Error:', mse)


# In[ ]:


###According to the five 5-fold MSEs, the 5 degree polynomial model is the best since it has the smallest MSE.

