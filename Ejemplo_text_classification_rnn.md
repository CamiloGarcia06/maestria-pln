---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="hX4n9TsbGw-f" -->
##### Copyright 2018 The TensorFlow Authors.
<!-- #endregion -->

```python id="0nbI5DtDGw-i"
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

<!-- #region id="9TnJztDZGw-n" -->
# Text classification with an RNN
<!-- #endregion -->

<!-- #region id="AfN3bMR5Gw-o" -->
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/text/tutorials/text_classification_rnn"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/text_classification_rnn.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/text/blob/master/docs/tutorials/text_classification_rnn.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/text/docs/tutorials/text_classification_rnn.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>
<!-- #endregion -->

<!-- #region id="lUWearf0Gw-p" -->
This text classification tutorial trains a [recurrent neural network](https://developers.google.com/machine-learning/glossary/#recurrent_neural_network) on the [IMDB large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) for sentiment analysis.
<!-- #endregion -->

<!-- #region id="_2VQo4bajwUU" -->
## Setup
<!-- #endregion -->

```python id="z682XYsrjkY9"
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()
```

<!-- #region id="1rXHa-w9JZhb" -->
Import `matplotlib` and create a helper function to plot graphs:
<!-- #endregion -->

```python id="Mp1Z7P9pYRSK"
import matplotlib.pyplot as plt


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
```

<!-- #region id="pRmMubr0jrE2" -->
## Setup input pipeline


The IMDB large movie review dataset is a *binary classification* dataset—all the reviews have either a *positive* or *negative* sentiment.

Download the dataset using [TFDS](https://www.tensorflow.org/datasets). See the [loading text tutorial](https://www.tensorflow.org/tutorials/load_data/text) for details on how to load this sort of data manually.

<!-- #endregion -->

```python id="SHRwRoP2nVHX" colab={"base_uri": "https://localhost:8080/"} outputId="5f59c6e6-fb3f-49e8-df19-be349b327375"
!pip install -q importlib-resources

dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec
```

<!-- #region id="nWA4c2ir7g6p" -->
Initially this returns a dataset of (text, label pairs):
<!-- #endregion -->

```python id="vd4_BGKyurao" colab={"base_uri": "https://localhost:8080/"} outputId="76ed497f-0c35-4955-a029-2cef7d96583e"
for example, label in train_dataset.take(2):
  print('text: ', example.numpy())
  print('label: ', label.numpy())
```

<!-- #region id="z2qVJzcEluH_" -->
Next shuffle the data for training and create batches of these `(text, label)` pairs:
<!-- #endregion -->

```python id="dDsCaZCDYZgm"
BUFFER_SIZE = 10000
BATCH_SIZE = 64
```

```python id="VznrltNOnUc5"
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
```

```python id="jqkvdcFv41wC" colab={"base_uri": "https://localhost:8080/"} outputId="fdad6685-8592-41e7-cbae-b93651945697"
for example, label in train_dataset.take(1):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:3])
```

<!-- #region id="s5eWCo88voPY" -->
## Create the text encoder
<!-- #endregion -->

<!-- #region id="TFevcItw15P_" -->
The raw text loaded by `tfds` needs to be processed before it can be used in a model. The simplest way to process text for training is using the `TextVectorization` layer. This layer has many capabilities, but this tutorial sticks to the default behavior.

Create the layer, and pass the dataset's text to the layer's `.adapt` method:
<!-- #endregion -->

<!-- #region id="lolMkwNfwQBM" -->
## ¿Que es esto?
<!-- #endregion -->

```python id="uC25Lu1Yvuqy"
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
```

<!-- #region id="IuQzVBbe3Ldu" -->
The `.adapt` method sets the layer's vocabulary. Here are the first 20 tokens. After the padding and unknown tokens they're sorted by frequency:
<!-- #endregion -->

```python id="tBoyjjWg0Ac9" colab={"base_uri": "https://localhost:8080/"} outputId="425b5bbd-2393-4fc4-bda4-6fb031adad76"
vocab = np.array(encoder.get_vocabulary())
vocab[:20]
```

<!-- #region id="mjId5pua3jHQ" -->
Once the vocabulary is set, the layer can encode text into indices. The tensors of indices are 0-padded to the longest sequence in the batch (unless you set a fixed `output_sequence_length`):
<!-- #endregion -->

<!-- #region id="MR-zCGGBxarf" -->
## ¿Esta matrix es tiene un numero de filas fija, al igual que el numero de columnas?
<!-- #endregion -->

```python id="RGc7C9WiwRWs" colab={"base_uri": "https://localhost:8080/"} outputId="91b2134d-46b5-49b5-8ee5-b718753dd137"
encoded_example = encoder(example)[:3].numpy()
encoded_example
```

<!-- #region id="F5cjz0bS39IN" -->
With the default settings, the process is not completely reversible. There are three main reasons for that:

1. The default value for `preprocessing.TextVectorization`'s `standardize` argument is `"lower_and_strip_punctuation"`.
2. The limited vocabulary size and lack of character-based fallback results in some unknown tokens.
<!-- #endregion -->

```python id="N_tD0QY5wXaK" colab={"base_uri": "https://localhost:8080/"} outputId="68189697-389b-4000-af63-4d9018769bf1"
for n in range(3):
  print("Original: ", example[n].numpy())
  print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
  print()
```

<!-- #region id="bjUqGVBxGw-t" -->
## Create the model
<!-- #endregion -->

<!-- #region id="W7zsmInBOCPO" -->
![A drawing of the information flow in the model](https://github.com/tensorflow/text/blob/master/docs/tutorials/images/bidirectional.png?raw=1)
<!-- #endregion -->

<!-- #region id="bgs6nnSTGw-t" -->
Above is a diagram of the model.

1. This model can be build as a `tf.keras.Sequential`.

2. The first layer is the `encoder`, which converts the text to a sequence of token indices.

3. After the encoder is an embedding layer. An embedding layer stores one vector per word. When called, it converts the sequences of word indices to sequences of vectors. These vectors are trainable. After training (on enough data), words with similar meanings often have similar vectors.

  This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a `tf.keras.layers.Dense` layer.

4. A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input on the next timestep.

  The `tf.keras.layers.Bidirectional` wrapper can also be used with an RNN layer. This propagates the input forward and backwards through the RNN layer and then concatenates the final output.

  * The main advantage of a bidirectional RNN is that the signal from the beginning of the input doesn't need to be processed all the way through every timestep to affect the output.  

  * The main disadvantage of a bidirectional RNN is that you can't efficiently stream predictions as words are being added to the end.

5. After the RNN has converted the sequence to a single vector the two `layers.Dense` do some final processing, and convert from this vector representation to a single logit as the classification output.

<!-- #endregion -->

<!-- #region id="V4fodCI7soQi" -->
The code to implement this is below:
<!-- #endregion -->

```python id="LwfoBkmRYcP3"
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

<!-- #region id="QIGmIGkkouUb" -->
Please note that Keras sequential model is used here since all the layers in the model only have single input and produce single output. In case you want to use stateful RNN layer, you might want to build your model with Keras functional API or model subclassing so that you can retrieve and reuse the RNN layer states. Please check [Keras RNN guide](https://www.tensorflow.org/guide/keras/rnn#rnn_state_reuse) for more details.
<!-- #endregion -->

<!-- #region id="kF-PsCk1LwjY" -->
The embedding layer [uses masking](https://www.tensorflow.org/guide/keras/masking_and_padding) to handle the varying sequence-lengths. All the layers after the `Embedding` support masking:
<!-- #endregion -->

```python id="87a8-CwfKebw" colab={"base_uri": "https://localhost:8080/"} outputId="cfbae6a9-126f-4893-de25-6205da4319c9"
print([layer.supports_masking for layer in model.layers])
```

<!-- #region id="ZlS0iaUIWLpI" -->
To confirm that this works as expected, evaluate a sentence twice. First, alone so there's no padding to mask:
<!-- #endregion -->

```python id="O41gw3KfWHus" colab={"base_uri": "https://localhost:8080/"} outputId="d0fe1dfa-62b9-4633-aed9-e3cad46f51d7"
# predict on a sample text without padding.

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
# Convertimos el string a un tensor de TensorFlow para evitar problemas de tipos
predictions = model.predict(tf.constant([sample_text]))
print(predictions[0])
```

<!-- #region id="K0VQmGnEWcuz" -->
Now, evaluate it again in a batch with a longer sentence. The result should be identical:
<!-- #endregion -->

```python id="UIgpuTeFNDzq" colab={"base_uri": "https://localhost:8080/"} outputId="e1fb0113-2431-42b7-cf58-9d17546fec04"
# predict on a sample text with padding

padding = "the " * 2000
predictions = model.predict(tf.constant([sample_text, padding]))
print(predictions[0])
```

<!-- #region id="sRI776ZcH3Tf" -->
Compile the Keras model to configure the training process:
<!-- #endregion -->

```python id="kj2xei41YZjC"
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
```

<!-- #region id="zIwH3nto596k" -->
## Train the model
<!-- #endregion -->

```python id="hw86wWS4YgR2" colab={"base_uri": "https://localhost:8080/"} outputId="fab23c58-19cc-4eaf-81de-457465f756d9"
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)
```

```python id="BaNbXi43YgUT" colab={"base_uri": "https://localhost:8080/"} outputId="c555a148-e91e-42db-9ad7-45321bf63be3"
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
```

```python id="OZmwt_mzaQJk" colab={"base_uri": "https://localhost:8080/", "height": 468} outputId="46233896-080c-4752-d607-fa590c94da6f"
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
```

<!-- #region id="DwSE_386uhxD" -->
Run a prediction on a new sentence:

If the prediction is >= 0.0, it is positive else it is negative.
<!-- #endregion -->

```python id="ZXgfQSgRW6zU" colab={"base_uri": "https://localhost:8080/"} outputId="6581a02d-f1bb-4bbc-e979-657d35780c8c"
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(tf.constant([sample_text]))
print(predictions)
```
