import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GRU

import pandas as pd
import numpy as np

import unicodedata

from pathlib import Path
import re
import shutil
from datetime import datetime
from pprint import pprint


models_dir = Path('.') / 'models'
punctuation = '“”…‘’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

# helper function for emojis
def deEmojify(text):
    regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regex_pattern.sub(r'',text)

def strip_accents(text):
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def clean_text(text):
    re_text = text

    # removes links, pics, mentions, and some emojis
    for pattern in [
      'http://\S+|https://\S+', 'pic.\S+', '@\S+',
      r'\([^)]*\)', '🤣', '✊']:
      re_text = re.sub(pattern, '', re_text)

    # strip out the rest of emojis
    re_text = deEmojify(re_text)

    re_text = strip_accents(re_text)

    return re_text

# gets every unique word without leading/trailing punctation or white space
# it had to be done in two steps because we also had to split on '/n' which was present
# in the corpus
def string_to_list_of_words(text):
  re_text = np.array(
      list(map(lambda s: s.strip(), 
               filter(lambda s: len(s) > 0 and s.strip(),
                  re.split(r'([\s' + re.escape(punctuation) + r'])',
                  text.replace('/n','\n'))))))
  
  return [w.lower() for w in re_text]

def process_new_text(text):
  return string_to_list_of_words( clean_text(text) )

def fetch_data():
  return pd.read_csv('https://gist.githubusercontent.com/jhigginbotham64/2c253f29576a05e1cf92790a18edecaf/raw/cf991dbfd7969aac33c92f414c7a9b217229d834/infowars.csv',encoding='utf-8')

def create_corpus():
  # read uploaded csv file
  # change this to wherever your file is loaded in your gdrive instance
  df = fetch_data()
  #drops ever instance of an element with a value NaN 
  df = df.dropna(subset=['title', 'content'])

  # get columns as numpy arrays
  titles = df['title'].to_numpy()
  articles = df['content'].to_numpy()

  # form initial text by concatenating all titles with their articles, then cleans it
  corpus = clean_text( '/n'.join([ 
      title + " " + article for title, article in zip(titles, articles)
  ]))

  # gets every unique word without leading/trailing punctation or white space
  words_in_corpus = string_to_list_of_words(corpus)
  chars_in_corpus = [c for c in corpus]

  return words_in_corpus, chars_in_corpus, corpus, df

def init_model(embedding_dim=256, rnn_units=1024, batch_size=64):
  layers = [
      Embedding(words['nunique'], embedding_dim, batch_input_shape=[batch_size, None]),
      GRU(rnn_units, 
          return_sequences=True,
          stateful=True,
          recurrent_initializer='glorot_uniform'),
      Dense(words['nunique'])
  ]

  return Sequential(layers)

# for creating training examples
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

# loss function we'll use for the model later
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def model_name(sequence_length, num_training_epochs):
  return f'seq{sequence_length}_ep{num_training_epochs}_{datetime.isoformat(datetime.now())}'

def prep_training_dataset(sequence_length=10, batch_size=64, buffer_size=10000):
  ''' 
    text_as_int is the text we want to prep, already converted to integers
    
    sequence_length is the maximum length sentence we want for a single training input in characters

    batch size is the number of examples in a training batch

    buffer size is to shuffle the dataset with
    (TF data is designed to work with possibly infinite sequences,
    so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    it maintains a buffer in which it shuffles elements).
  '''
  # Create training examples / targets
  word_dataset = tf.data.Dataset.from_tensor_slices(words['as_int'])

  sequences = word_dataset.batch(sequence_length+1, drop_remainder=True)

  dataset_unshuffled = sequences.map(split_input_target)

  # now we have batches of 64 input/target pairs,
  # where the input and target are both 100-char 
  # sentences...shuffled...
  dataset = dataset_unshuffled.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

  return dataset

def generate_text(model, start_string, num_generate=50):
  # Evaluation step (generating text using the learned model)

  # Converting our start string to numbers (vectorizing)
  # mappings must have been created in a different cell prior to calling this function
  input_eval = [words['map_from'][w] for w in process_new_text(start_string)]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(words['map_to'][predicted_id])

  ret = start_string
  for w in text_generated:
    if w in punctuation:
      ret += w
    else:
      ret += ' ' + w
  return ret

def create_text_generator(sequence_length=10, num_training_epochs=5, mname=None):
  model = init_model()

  model.compile(optimizer='adam', loss=loss)

  if mname is None:
    mname = model_name(sequence_length, num_training_epochs)

  checkpoint_dir = models_dir / 'training_checkpoints'
  checkpoint_model_dir = checkpoint_dir / mname
  checkpoint_prefix = checkpoint_model_dir / "ckpt_{epoch}"
  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

  model.fit(prep_training_dataset(sequence_length=sequence_length), epochs=num_training_epochs, callbacks=[checkpoint_callback])

  model = init_model(batch_size=1)

  model.load_weights(tf.train.latest_checkpoint(checkpoint_model_dir)).expect_partial()

  model.build(tf.TensorShape([1, None]))

  model_names[model] = mname

  shutil.rmtree(checkpoint_dir)

  return model

def list_models():
  if models_dir.is_dir():
    for m in models_dir.iterdir():
      print(str(m))

# would have liked to set attribute on model, but nooooooo
def save(m):
  m.save(models_dir / model_names[m], overwrite=True)

# https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
def load(mname):
  if mname in model_names.values():
    return list(model_names.keys())[list(model_names.values()).index(mname)]
  m = load_model(models_dir / mname)
  model_names[m] = mname
  return m

model_names = {}

# create corpus
# corpus is just the whole thing cleaned and with titles and articles appended
# corpus_word_list is the list of words, corpus_char_list is the characters,
# df is the raw corpus as a Pandas dataframe
corpus_word_list, corpus_char_list, corpus, df = create_corpus()

# global vocab
# used inside most important functions
# to clarify: words have already been filtered,
# the numbers here are indices, not frequencies
words = {}

words['unique'] = sorted(set(corpus_word_list))
words['nunique'] = len(words['unique'])
words['map_from'] = {w:i for i, w in enumerate(words['unique'])}
words['map_to'] = np.array(words['unique'])
words['as_int'] = np.array([words['map_from'][w] for w in corpus_word_list])

# global character vocab, in case anyone's interested
chars = {}

chars['unique'] = sorted(set(corpus_char_list))
chars['nunique'] = len(chars['unique'])
chars['map_from'] = {c:i for i, c in enumerate(chars['unique'])}
chars['map_to'] = np.array(chars['unique'])
chars['as_int'] = np.array([chars['map_from'][c] for c in corpus_char_list])

