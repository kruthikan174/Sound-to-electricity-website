from flask import Flask, render_template, request, send_file, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_visualization')
def data_visualization():
    # Load and process the dataset (assuming your dataset processing logic here)
    # Example using random data:
    regions = ['North', 'South', 'East', 'West']
    thermal = [400, 600, 800, 700]
    nuclear = [300, 500, 700, 600]
    hydro = [200, 400, 600, 500]

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.bar(regions, thermal, color='blue')
    plt.title('Thermal Generation')
    plt.xlabel('Region')
    plt.ylabel('Generation (in MU)')

    plt.subplot(1, 3, 2)
    plt.bar(regions, nuclear, color='green')
    plt.title('Nuclear Generation')
    plt.xlabel('Region')
    plt.ylabel('Generation (in MU)')

    plt.subplot(1, 3, 3)
    plt.bar(regions, hydro, color='orange')
    plt.title('Hydro Generation')
    plt.xlabel('Region')
    plt.ylabel('Generation (in MU)')

    plt.tight_layout()
    plt.savefig('static/data_visualization.png')
    plt.close()

    return render_template('data_visualization.html')

# Step 1: Create the CSV file
def create_csv():
    data = {
        "Sound Input (dB)": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140],
        "Output Voltage (mV)": [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.2, 3.0, 4.0, 5.2, 6.6, 8.2, 10.0, 12.0, 14.2, 16.6, 19.2, 22.0, 25.0, 28.2, 31.6]
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join('static', 'Sound_Data.csv')
    df.to_csv(csv_path, index=False)

# Step 2: Generate the graph
def generate_plot():
    csv_path = os.path.join('static', 'Sound_Data.csv')
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Sound Input (dB)', y='Output Voltage (mV)', marker='o')
    sns.lineplot(data=df, x='Sound Input (dB)', y='Output Voltage (mV)', markers=True, dashes=False)
    plt.title('Sound Input (dB) vs Output Voltage (mV)')
    plt.xlabel('Sound Input (dB)')
    plt.ylabel('Output Voltage (mV)')
    plt.grid(True)
    plot_path = os.path.join('static', 'sound_vs_voltage.png')
    plt.savefig(plot_path)
    plt.close()

# Ensure the CSV and plot are created
create_csv()
generate_plot()

# Load the sound data CSV file
sound_data_file = "static/Sound_Data.csv"
sound_data = pd.read_csv(sound_data_file)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/electricity-sources')
def electricity_sources():
    return render_template('electricity_sources.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename)

@app.route('/plot')
def plot():
    return send_file(os.path.join('static', 'sound_vs_voltage.png'), mimetype='image/png')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sound_decibel = float(request.form['sound_decibel'])

        # Load and train the model (assuming this is a small dataset for simplicity)
        X = sound_data[['Sound Input (dB)']].values
        Y = sound_data['Output Voltage (mV)'].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        reg = LinearRegression()
        reg.fit(X_train, Y_train)

        # Predict the output voltage
        predicted_voltage = reg.predict([[sound_decibel]])[0]

        return render_template('predict.html', sound_decibel=sound_decibel, predicted_voltage=predicted_voltage)

    return render_template('predict.html')

@app.route('/decibels-to-voltage', methods=['GET', 'POST'])
def decibels_to_voltage():
    predicted_voltage = None
    if request.method == 'POST':
        sound_decibel = float(request.form['sound_decibel'])

        # Load and train the model
        X = sound_data[['Sound Input (dB)']].values
        Y = sound_data['Output Voltage (mV)'].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        reg = LinearRegression()
        reg.fit(X_train, Y_train)

        # Predict the output voltage
        predicted_voltage = reg.predict([[sound_decibel]])[0]

    return render_template('decibels_to_voltage.html', predicted_voltage=predicted_voltage)

if __name__ == '__main__':
    app.run(debug=True)
