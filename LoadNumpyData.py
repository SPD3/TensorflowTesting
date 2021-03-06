import numpy as np
import tensorflow as tf

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']

print("First Training example:")
print(train_examples[:1])
print("Training examples shape:")
print(train_examples.shape)
print("First Training label:")
print(train_labels[:1])
print("Training label shape:")
print(train_labels.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

inputs = tf.keras.layers.Input(shape=(28,28))
x = tf.keras.layers.Flatten(input_shape=(28, 28))(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(10)(x)

if(False):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
else:
    model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)