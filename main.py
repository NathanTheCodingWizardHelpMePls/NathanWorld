import keras
from keras import layers
from keras import ops
from keras import models
import numpy as np
inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
# outputs = layers.Dense(10)(x)
# model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

encoder_input = layers.Input(shape=(3, 3))
random_input = np.random.random((2, 3, 3)) 

# dense1 = layers.Dense(64, activation="relu")(encoder_input)
# dense2 = layers.Dense(64, activation="relu")(dense1)
# output = layers.Dense(64, activation="relu")(dense1)
# model = keras.Model(inputs=encoder_input, outputs=output, name="hi")
# print(model.summary())
# print(model(random_input))



def generate_data(batch_size=100):
    X = np.random.rand(batch_size, 3, 4).astype(np.float32)  # Random tensors with shape (batch_size, 3, 4)
    y = np.min(X, axis=(1, 2))  
    return X, y

# Build the Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(3, 4)),  # Flatten the 3x4 tensor into a 1D array
    keras.layers.Dense(64, activation='relu'),  # Hidden layer
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Output layer with a single unit for the max value prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Generate training data
X_train, y_train = generate_data(1000)

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32)

# Test the model with new data
X_test, y_test = generate_data(10)
predictions = model.predict(X_test)
one_dx = []

def find_nearest(array, value):
    l = []
    for i in range(len(array)):
        l.append(abs(array[i])-value)
    return(min(l)+value)
x1 = []
x2 = []
x3 = []

for i in range(len(X_test)):
    print(f"Test tensor:\n{X_test[i]}")
    e = len(str(predictions[i][0]))
    x1 = X_test[i][0].tolist()
    x2 = X_test[i][1].tolist()
    x3 = X_test[i][2].tolist()
    del one_dx[:]
    
    for a in range(3):
        for y in range(4):
            if a == 0: one_dx.append(x1[y])
            if a == 1: one_dx.append(x2[y])
            if a == 2: one_dx.append(x3[y])

    closest = find_nearest(one_dx, predictions[i][0])
    predictions[i] = closest
    y_test[i] = round(y_test[i], e)
    print(f"Predicted value: {predictions[i]}")
    print(f"Actual value: {y_test[i]}")
    print(f"Loss: {predictions[i]-y_test[i]}")
    print("---")
