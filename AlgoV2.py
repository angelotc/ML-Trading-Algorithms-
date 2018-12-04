# Use the previous 10 bars' movements to predict the next movement.

# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import talib
from sklearn.linear_model import BayesianRidge

def initialize(context):
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))                  
    
    
    context.security_list = [ sid(5061),  # MSFT  - Tech
                              sid(23103), # ANTM - Healthcare,
                             sid(2968),   # #NEE -  energy
                            sid(19659)] #   XLP -  Consumer staples
    
    context.volume_scalers = {}
    context.price_scalers  = {}
    
    context.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    context.BR_models = {}
    context.GBR_models = {}
  
    context.prediction = np.zeros_like(context.security_list)

    context.lookback = 5 # Look back 5 days, was 3 days
    context.history_range = 150 #Only consider the past 200 days' history, was 400

    # Generate a new model every week
    schedule_function(create_model, date_rules.week_end(), time_rules.market_close(minutes=10))

    # Trade 1 minute after the start of every day
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=1))
    schedule_function(
    short,
    date_rules.week_start(days_offset=1),
    time_rules.market_open(minutes=1)
  )


def create_model(context, data):
    '''
    Called in initialize() via the schedule_function. Trains a model at the end of every trading week for each and every
    stock. 
    '''
       
    for idx, security in enumerate(context.security_list):
        train_model(context, data, idx)
        
def train_model(context, data, idx):    
    '''
    Given the Called in initialize() via the schedule_function. 
    '''   
    recent_volumes = data.history(context.security_list[idx], 'volume', context.history_range, '1d').values
    recent_prices = data.history(context.security_list[idx], 'price', context.history_range, '1d').values

    # Get the price changes
    # Volume here too
    volume_changes = np.diff(recent_volumes).tolist()
    price_changes = np.diff(recent_prices).tolist()

    # Create a scaler for every stock for every feature and store it within our dictionary along with its respective fit
    context.volume_scalers[idx] = context.scaler.fit(volume_changes)
    X_vol_changes_train = context.volume_scalers[idx].transform(volume_changes).tolist()
    
    
    context.price_scalers[idx] = context.scaler.fit(price_changes)
    X_price_changes_train = context.price_scalers[idx].transform(price_changes).tolist()
    
    
    X = [] # Independent, or input variables
    Y = [] # Dependent, or output variable
    
    # For each day in our history
    for i in range(150-context.lookback-1):
        # Store scaled training data
        X.append(X_price_changes_train[i:i+context.lookback] + X_vol_changes_train[i:i+context.lookback])
        # Store our test data 
        Y.append(volume_changes[i+context.lookback]+price_changes[i+context.lookback]) # Store the day's volume change
    
    
    
    BR = BayesianRidge()
    GBR = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 150)
    
    # Generate our models
    BR.fit(X, Y) 
    GBR.fit(X, Y)

    # Store our models
    context.BR_models[idx] = BR
    context.GBR_models[idx] = GBR


def trade(context, data):
    '''
    Called in initialize(). Will check if our models have been trained, and it will make a decision to buy depending 
    on the indicator's predicted value.
    '''
    if context.BR_models and context.GBR_models: # Check if our model is generated
        for idx, security in enumerate(context.security_list):
            # Get recent prices and volume
            recent_volumes = data.history(security, 'volume', context.lookback+1, '1d').values
            recent_prices = data.history(security, 'price', context.lookback+1, '1d').values

            # Get the price and volume changes
            volume_changes = np.diff(recent_volumes).tolist()
            price_changes = np.diff(recent_prices).tolist()
            
            # Use our MinMaxScaler on testing data 
            volume_changes = context.volume_scalers[idx].transform(volume_changes).tolist()
            price_changes = context.price_scalers[idx].transform(price_changes).tolist()
            
            weight = 1.0 / len(context.security_list)  
            # Predict using our model and the recent prices
            prediction = 0.5*(context.BR_models[idx].predict( price_changes + volume_changes)) + 0.55*(context.GBR_models[idx].predict( price_changes + volume_changes))
            
            record(prediction = prediction)
        
            # Go long if we predict the price will rise, short otherwise
            ## Volume again!
            if prediction > 0:
                order_target_percent(security, +weight)
            else:
                order_target_percent(security, +weight)

def short(context, data):
    '''
    Same as trade function, but sells instead and it only decides to sell at the beginning of every week. 
    '''
    if context.BR_models and context.GBR_models: # Check if our model is generated
        for idx, security in enumerate(context.security_list):
            # Get recent prices and volume
            # Short is the same as trade, but it's only ran once per week at the beginning. 
            recent_volumes = data.history(security, 'volume', context.lookback+1, '1d').values
            recent_prices = data.history(security, 'price', context.lookback+1, '1d').values

            # Get the price and volume changes
            volume_changes = np.diff(recent_volumes).tolist()
            price_changes = np.diff(recent_prices).tolist()
            
            # Use our MinMaxScaler on testing data 
            volume_changes = context.volume_scalers[idx].transform(volume_changes).tolist()
            price_changes = context.price_scalers[idx].transform(price_changes).tolist()
            
            # Assign the weight of each stock evenly
            weight = 1.0 / len(context.security_list)   

            # Predict overall delta using our model and the recent prices
            prediction = (0.3*(context.BR_models[idx].predict(price_changes + volume_changes))) + (0.7*(context.GBR_models[idx].predict(price_changes + volume_changes)))
            
            record(prediction = prediction)
        
            # Go long if we predict the price will rise, short otherwise
            ## Volume again!
            if prediction < 0 :
                order_target_percent(security, -weight)