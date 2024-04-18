import pickle

# Load the trained model from the .pkl file
with open('knn_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the StandardScaler
with open('scaler.pkl', 'rb') as file:
    sc = pickle.load(file)

new_data_point = [[131, 38, 19, 23, 75, 6, 90]] # Example new data point

# Apply the StandardScaler to the new data point
scaled_new_data_point = sc.transform(new_data_point)

# Predict the crop type using the loaded model
predicted_crop = classifier.predict(scaled_new_data_point)
print(predicted_crop)