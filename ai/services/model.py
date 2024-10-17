from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Загрузка модели (укажите правильный путь к файлу модели)
model = tf.keras.models.load_model('../models/model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    
    # Предобработка данных
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Предсказания
    predictions = model.predict(df_scaled)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(port=5000)
