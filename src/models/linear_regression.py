import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


from src.config import Config


deals_df: pd.DataFrame = pd.read_csv(Config.POST_PROCESSED_DEALS_LOCATION)
X = deals_df[[
    Config.DEAL_DATE_FIELD_NAME,
    Config.DEAL_TYPE_FIELD_NAME,
    Config.ROOMS_NUMBER_FIELD_NAME,
    Config.SIZE_FIELD_NAME,
    Config.BUILDING_BUILD_YEAR_FIELD_NAME,
    Config.BUILDING_NUM_FLOORS_FIELD_NAME,
    Config.STARTING_FLOOR_NUMBER_FIELD_NAME,
    Config.NUMBER_FLOORS_FIELD_NAME,
    Config.SETTLEMENT_NUMERIC_FIELD_NAME,
    Config.X_ITM_FIELD_NAME,
    Config.Y_ITM_FIELD_NAME
]]
y = deals_df[Config.DEAL_SUM_FIELD_NAME]

# if want to use the deal type categorical value
X = pd.get_dummies(X, columns=[Config.DEAL_TYPE_FIELD_NAME], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (optional, but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')
