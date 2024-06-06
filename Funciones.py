def read_config(file_path):
    import yaml

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def descargar_datos():
    import gdown

    # Ruta al archivo config.yaml
    config_file_path = 'config.yaml'

    # Leer el archivo de configuración
    config = read_config(config_file_path)

    # Acceder a los valores del archivo YAML
    data_url = config['data']

    # Ruta local donde se guardará el archivo descargado
    output = 'archivo.csv'

    # Descargar el archivo
    gdown.download(data_url, output, quiet=False)

    # Devolvemos la ruta de la base de datos
    return output

def cargar_limpiar_datos(output):
    import pandas as pd
    # Creamos un dataframe con la base de datos
    df = pd.read_csv(output)

    # Ponemos la fecha en tipo fecha 
    df["date"] = pd.to_datetime(df["date"])

    # Eliminamos las columnas timestamp, close_time, ignore
    df = df.drop(columns=["timestamp", "close_time", "ignore"])

    return df

def crear_tabla_por_horas(df):
    df.set_index('date', inplace=True)

    hourly_data = df.resample('h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'number_of_trades': 'sum'
    })

    hourly_data.reset_index(inplace=True)
    df.reset_index(inplace=True)
    
    return hourly_data

def crear_tabla_por_dia(df):
    df.set_index('date', inplace=True)

    daily_data = df.resample('d').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'number_of_trades': 'sum'
    })

    daily_data.reset_index(inplace=True)
    df.reset_index(inplace=True)
    
    return daily_data

def crear_modelo(df):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    datas = {
        'date': df["date"],
        'close': df["close"]
    }
    data = pd.DataFrame(datas)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    data = data.ffill()

    # Preprocesamiento de datos
    # Supongamos que usamos los últimos 60 días para predecir el próximo día
    window_size = 60
    features = []
    labels = []
    for i in range(window_size, len(data)):
        features.append(data['close'][i-window_size:i])
        labels.append(data['close'][i])

    # Convertir a arrays de numpy para entrenamiento
    features, labels = np.array(features), np.array(labels)

    train_features = features[:]
    train_labels = labels[:]

    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features_scaled = scaler.fit_transform(train_features)  # Ajustar y transformar los datos de entrenamiento

    # Reshape features para el modelo LSTM (muestras, pasos de tiempo, características)
    train_features = np.reshape(train_features_scaled, (train_features_scaled.shape[0], train_features_scaled.shape[1], 1))

    #entrenamos el codigo con todos los datos

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(50),
        Dropout(0.2),
        BatchNormalization(),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate= 0.001)

    # Compilar el modelo
    model.compile(optimizer = optimizer, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    history = model.fit(
        train_features, train_labels,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )

    # Tomar los últimos 'window_size' puntos de los datos como punto de partida
    input_sequence = data['close'][-window_size:].values.reshape(1, -1)
    input_sequence_scaled = scaler.transform(input_sequence)

    predictions = []
    for _ in range(30):  # Predice 30 valores futuros
        # Reshape del input para que coincida con la entrada del modelo (1, window_size, 1)
        reshaped_input = np.reshape(input_sequence_scaled, (1, window_size, 1))
        # Predecir el siguiente valor
        predicted_value = model.predict(reshaped_input)
        predictions.append(predicted_value[0][0])
        # Añadir la predicción al input para la siguiente iteración
        input_sequence_scaled = np.append(input_sequence_scaled[:, 1:], predicted_value, axis=1)
    
    # Transformar los valores predichos a la escala original
    return predictions 

def predict_values(model, df, horas):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    datas = {
        'date': df["date"],
        'close': df["close"]
    }
    data = pd.DataFrame(datas)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    data = data.ffill()

    # Preprocesamiento de datos
    # Supongamos que usamos los últimos 60 días para predecir el próximo día
    window_size = 60
    features = []
    labels = []
    for i in range(window_size, len(data)):
        features.append(data['close'][i-window_size:i])
        labels.append(data['close'][i])

    # Convertir a arrays de numpy para entrenamiento
    features, labels = np.array(features), np.array(labels)

    train_features = features[:]
    train_labels = labels[:]

    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features_scaled = scaler.fit_transform(train_features)  # Ajustar y transformar los datos de entrenamiento

    # Reshape features para el modelo LSTM (muestras, pasos de tiempo, características)
    train_features = np.reshape(train_features_scaled, (train_features_scaled.shape[0], train_features_scaled.shape[1], 1))


    # Tomar los últimos 'window_size' puntos de los datos como punto de partida
    input_sequence = data['close'][-window_size:].values.reshape(1, -1)
    input_sequence_scaled = scaler.transform(input_sequence)

    predictions = []
    for _ in range(horas):  # Predice 30 valores futuros
        # Reshape del input para que coincida con la entrada del modelo (1, window_size, 1)
        reshaped_input = np.reshape(input_sequence_scaled, (1, window_size, 1))
        # Predecir el siguiente valor
        predicted_value = model.predict(reshaped_input)
        predictions.append(predicted_value[0][0])
        # Añadir la predicción al input para la siguiente iteración
        input_sequence_scaled = np.append(input_sequence_scaled[:, 1:], predicted_value, axis=1)
    
    # Transformar los valores predichos a la escala original
    return predictions 



def plot_chart(dataframe,title):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=dataframe['date'],
                                         open=dataframe['open'],
                                         high=dataframe['high'],
                                         low=dataframe['low'],
                                         close=dataframe['close'])])
    fig.update_layout(title=title,
                      xaxis_title='Date',
                      yaxis_title='Price (USDT)',
                      xaxis_rangeslider_visible=False)
    return fig

def plot_chart_pred(df,predictions, horas):
    import pandas as pd
    import plotly.express as px
    
    datas = {
        'date': df["date"],
        'close': df["close"]
    }
    data = pd.DataFrame(datas)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    data = data.ffill()

    date_range = pd.date_range(start=data.index.max(), periods=horas, freq='H')

    predictions = pd.DataFrame(data=predictions, index=date_range)

    fig = px.line(data, x=data.index, y='close', labels={'close': 'Value'}, title='Comparison of Real and Predicted Values')
    fig.add_scatter(x=predictions.index, y=predictions[0], mode='lines', name='Predicted Values', opacity=0.7)
    
    fig.update_layout(xaxis_title='Time (index)', yaxis_title='Value')
 

    return fig



