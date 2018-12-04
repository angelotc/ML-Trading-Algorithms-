# Use the previous 10 bars' movements to predict the next movement.

# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def initialize(context):
    context.security = sid(5061)# Trade SPY
    context.model = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 150)

    context.lookback = 5 # Look back 5 days, was 3 days
    context.history_range = 178 #Only consider the past 200 days' history, was 400

    # Generate a new model every week
    schedule_function(create_model, date_rules.week_end(), time_rules.market_close(minutes=10))

    # Trade 1 minute after the start of every day
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=1))

def create_model(context, data):
    # Get the relevant daily prices
    ##Changed to volume
    recent_volumes = data.history(context.security, 'volume', context.history_range, '1d').values
    
    # Get the price changes
    # Volume here too
    volume_changes = np.diff(recent_volumes).tolist()

    X = [] # Independent, or input variables
    Y = [] # Dependent, or output variable
    
    # For each day in our history
    for i in range(context.history_range-context.lookback-1):
        X.append(volume_changes[i:i+context.lookback]) # Store prior price changes
        Y.append(volume_changes[i+context.lookback]) # Store the day's price change

    context.model.fit(X, Y) # Generate our model

def trade(context, data):
    if context.model: # Check if our model is generated
        
        # Get recent prices
        recent_volumes = data.history(context.security, 'volume', context.lookback+1, '1d').values
        
        # Get the price changes
        volume_changes = np.diff(recent_volumes).tolist()
        
        # Predict using our model and the recent prices
        prediction = context.model.predict(volume_changes)
        record(prediction = prediction)
        
        # Go long if we predict the price will rise, short otherwise
        ## Volume again!
        if prediction > 0:
            order_target_percent(context.security, 1.0)
        else:
            order_target_percent(context.security, -1.0)