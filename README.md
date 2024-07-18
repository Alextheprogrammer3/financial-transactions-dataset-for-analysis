## Project: Optimizing Financial Analysis with L-BFGS-B Algorithm

### Objective

In this project, I utilized the L-BFGS-B optimization algorithm to optimize a specific function using financial data. The objective was to find the optimal values for two variables that minimize the function, reflecting a scenario common in financial analysis where minimizing costs or maximizing returns is crucial.

### Methodology

1. **Data Collection and Preprocessing**
   - Gathered relevant financial data, ensuring it was cleaned and formatted appropriately.
   - Example data snippet:
     ```python
     print(df.tail())
     print(df.tail())
     <bound method NDFrame.tail of    Timestamp  TransactionAmount TransactionType
     0 2023-01-01               1000        transfer
     1 2023-02-01               1200        transfer
     2 2023-03-01                800      withdrawal
     3 2023-04-01               1500         payment
     4 2023-05-01                900        transfer>
     ```

2. **Optimization Algorithm Implementation**
   - Implemented the L-BFGS-B algorithm to optimize the financial metric. Here's the running log:
     ```
     RUNNING THE L-BFGS-B CODE

                * * *

     Machine precision = 2.220D-16
      N =            5     M =           10

     At X0         0 variables are exactly at the bounds

     At iterate    0    f=  8.94378D+00    |proj g|=  7.66424D-02
      This problem is unconstrained.

     At iterate    5    f=  8.86358D+00    |proj g|=  1.43241D-02

     At iterate   10    f=  8.85027D+00    |proj g|=  6.12008D-04

     At iterate   15    f=  8.84819D+00    |proj g|=  3.18529D-03

     At iterate   20    f=  8.84806D+00    |proj g|=  2.38696D-03
     ```

3. **Code Implementation**

   ```python
   # Example Python code demonstrating various financial analysis techniques
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from statsmodels.tsa.arima.model import ARIMA
   import plotly.express as px
   from sklearn.cluster import KMeans
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import tensorflow as tf
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Dropout
   from tensorflow.keras.optimizers import Adam
   from sklearn.metrics import classification_report, confusion_matrix
   from statsmodels.tsa.statespace.sarimax import SARIMAX

   # Sample data
   data = {
       'Timestamp': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
       'TransactionAmount': [1000, 1200, 800, 1500, 900],
       'TransactionType': ['transfer', 'transfer', 'withdrawal', 'payment', 'transfer'],
   }

   df = pd.DataFrame(data)

   # Feature engineering
   df['payment_failed'] = np.where(df['TransactionType'] == 'payment', 1, 0)

   # Select features and target
   X = df[['TransactionAmount']]
   y = df['payment_failed']

   # Split data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Standardize features
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # Build TensorFlow model with improved architecture and regularization
   model = Sequential([
       Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
       Dropout(0.3),
       Dense(32, activation='relu'),
       Dropout(0.3),
       Dense(1, activation='sigmoid')
   ])

   # Use Adam optimizer with a lower learning rate for better convergence
   optimizer = Adam(learning_rate=0.001)

   model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

   # Train the model with batch normalization and early stopping
   early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

   history = model.fit(X_train_scaled, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_data=(X_test_scaled, y_test),
                       callbacks=[early_stopping])

   # Evaluate the model
   y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
   print(confusion_matrix(y_test, y_pred))
   print(classification_report(y_test, y_pred))

   # ARIMA model fitting
   model_arima = ARIMA(df['TransactionAmount'], order=(1, 1, 1))
   arima_result = model_arima.fit()

   # SARIMA model fitting
   model_sarima = SARIMAX(df['TransactionAmount'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
   sarima_result = model_sarima.fit()

   # K-means clustering
   kmeans = KMeans(n_clusters=3, random_state=42)
   df['Cluster'] = kmeans.fit_predict(df[['TransactionAmount', 'Frequency']])
