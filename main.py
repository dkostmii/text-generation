# Source: https://www.tensorflow.org/text/tutorials/text_generation

import tensorflow as tf
import numpy as np
import os
import time

from model import MyModel
from step import OneStep

text = open('corpus.txt', 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

ids_from_chars = tf.keras.layers.StringLookup(
  vocabulary=list(vocab), mask_token=None)

chars_from_ids = tf.keras.layers.StringLookup(
  vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


all_ids = ids_from_chars(tf.strings.unicode_split(text, input_encoding='UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(sequence):
  input_text = sequence[:-1]
  target_text = sequence[1:]
  return input_text, target_text


dataset = sequences.map(split_input_target)

BATCH_SIZE = 4
BUFFER_SIZE = 400

dataset = (
  dataset
  .shuffle(BUFFER_SIZE)
  .batch(BATCH_SIZE, drop_remainder=True)
  .prefetch(tf.data.experimental.AUTOTUNE)
)

print(dataset)

vocab_size = len(ids_from_chars.get_vocabulary())
embedding_dim = 256
rnn_units = 1024

model = MyModel(
  vocab_size=vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "chk_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_prefix,
  save_freq=20,
  save_weights_only=True
)

EPOCHS = 100

latest = tf.train.latest_checkpoint(checkpoint_dir)

if not latest:
  print("Restoring from scratch.")
  history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
else:
  model.load_weights(latest)
  print("Restored from {}".format(latest))

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['Hello'])
result = [next_char]

for n in range(200):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
