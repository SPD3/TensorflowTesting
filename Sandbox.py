import tensorflow as tf
import numpy as np

X = []
for i in range(10):
    input = []
    for j in range(6):
        input.append(np.random.random())
    X.append(input)

y = []
for i in range(10):
    if(np.random.random() > 0.5):
        y.append(1.0)
    else:
        y.append(0.0)
    
input = tf.keras.layers.Input(shape=(len(X[0]),))
x = tf.keras.layers.Dense(10, activation="relu")(input)
x = tf.keras.layers.Dense(10, activation="relu")(x)
x = tf.keras.layers.Dense(10, activation="relu")(x)
output = tf.keras.layers.Dense(1, activation="softmax")(x)

model = tf.keras.Model(inputs=input, outputs=output)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)],
)

print("X: " , X)
print("y: " , y)

train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

model.fit(train_dataset, epochs=10)

