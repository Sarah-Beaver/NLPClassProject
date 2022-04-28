# !pip install keras-tuner
import warnings  # stop annoying tf info dumping
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM, Attention

import pandas as pd
import numpy as np

import unicodedata

from gc import collect
from pathlib import Path
import re
import shutil
from datetime import datetime
from pprint import pprint
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import Counter
from numpy.random import choice 
from random import shuffle
import math
import nltk
from nltk import word_tokenize, sent_tokenize
nltk.download('punkt')
try:
  #tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
  tf.keras.mixed_precision.set_global_policy('mixed_float16')
  import keras_tuner as kt
except:
  pass


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

# for creating training examples
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

# loss function we'll use for the model later
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def model_name(sequence_length, num_training_epochs):
    return f'seq{sequence_length}_ep{num_training_epochs}_{str(datetime.now().month)+"_"+str(datetime.now().day) +"_"+str(datetime.now().year) +"_"}'

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

def init_model(embedding_dim=256, num_layers=2,sequence_length=10, rnn_units=1024, bs=64,normalize=0,
                   dropout_rate=.2, regularize_rate=.01):
  inputs = tf.keras.Input(batch_input_shape=[bs, None])
  x = Embedding(words['nunique'], embedding_dim, input_length=sequence_length,batch_input_shape=[bs, None])(inputs)
  if normalize == 1:
    x = tf.keras.layers.Normalization()(x)
  for i in range(num_layers): 
    x = GRU(rnn_units, return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform')(x)
    if dropout_rate > 0.0:
      x = tf.keras.layers.Dropout(dropout_rate)(x)
  if regularize_rate >0.0:
    x = Dense(words['nunique'], activation='softmax', activity_regularizer=tf.keras.regularizers.L2(regularize_rate)) (x)
  else:
    x = Dense(words['nunique'], activation='softmax')(x)
  model = tf.keras.Model(inputs,x)
  return model

def create_text_generator(sequence_length=10, num_training_epochs=5, mname=None, embedding_dim=256, rnn_units=1024, batch_size=64, num_layers=1, lr=.001, epsilon=1e-08,
                              normalize=0,dropout_rate=.2, regularize_rate=0):
  
  model = init_model(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=batch_size,
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)

  optimizer = tf.keras.optimizers.Adamax(learning_rate=lr,epsilon=epsilon,name='Adamax')
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')


  if mname is None:
    mname = model_name(sequence_length, num_training_epochs)

  checkpoint_dir = models_dir / 'training_checkpoints'
  checkpoint_model_dir = checkpoint_dir / mname
  checkpoint_prefix = checkpoint_model_dir / "ckpt_{epoch}"
  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

  model.fit(prep_training_dataset(sequence_length=sequence_length), epochs=num_training_epochs, callbacks=[checkpoint_callback])

  model = init_model(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=1, 
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)

  model.load_weights(tf.train.latest_checkpoint(checkpoint_model_dir)).expect_partial()

  model.build(tf.TensorShape([1, None]))

  model_names[model] = mname

  shutil.rmtree(checkpoint_dir)

  return model

def init_LSTMmodel(embedding_dim=256, num_layers=2,sequence_length=10, rnn_units=1024, bs=64,normalize=0,
                   dropout_rate=.2, regularize_rate=.01):
    inputs = tf.keras.Input(batch_input_shape=[bs, None])
    x = Embedding(words['nunique'], embedding_dim, input_length=sequence_length,batch_input_shape=[bs, None])(inputs)
    if normalize == 1:
      x = tf.keras.layers.Normalization()(x)
    for i in range(num_layers): 
      x = LSTM(int(rnn_units/num_layers), return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
      if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    if regularize_rate >0.0:
      x = Dense(words['nunique'], activation='softmax', activity_regularizer=tf.keras.regularizers.L2(regularize_rate)) (x)
    else:
      x = Dense(words['nunique'], activation='softmax')(x)
    model = tf.keras.Model(inputs,x)
    return model

def create_text_generatorLSTM(sequence_length=10, num_training_epochs=5, mname=None, embedding_dim=256, rnn_units=1024, batch_size=64, num_layers=1, lr=.001, epsilon=1e-08,
                              normalize=0,dropout_rate=.2, regularize_rate=0):

  model = init_LSTMmodel(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=batch_size,
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)

  optimizer = tf.keras.optimizers.Adamax(learning_rate=lr,epsilon=epsilon,name='Adamax')
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

  if mname is None:
    mname = model_name(sequence_length, num_training_epochs)

  checkpoint_dir = models_dir / 'LSTM_training_checkpoints'
  checkpoint_model_dir = checkpoint_dir / mname
  checkpoint_prefix = checkpoint_model_dir / "ckpt_{epoch}"
  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

  model.fit(prep_training_dataset(sequence_length=sequence_length), epochs=num_training_epochs, callbacks=[checkpoint_callback])

  model = init_LSTMmodel(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=1, 
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)

  model.load_weights(tf.train.latest_checkpoint(checkpoint_model_dir)).expect_partial()

  model.build(tf.TensorShape([1, None]))

  model_names[model] = mname

  shutil.rmtree(checkpoint_dir)

  return model

##########################################################################################
# John's experiments to alleviate the terrible sentences we have produced for ourselves
##########################################################################################
class tuned_models:
  def __init__(self, words, chars, corpus):
      self.words = words
      self.chars = chars
      self.tuned_LSTM = None
      self.tuned_GRU = None
      self.auto_encoder = None
      self.transfer_learning = None
      self.training, self.validation = None, None
      self.trigram = None
      self.corpus = corpus
      self.tuner = None
      self.tuned_model = None
      self.hyperperameter = pickle.load( open( "tuner.pkl", "rb"))
  def dataset(self, sequence_length=10, batch_size=64, split_testing = 0.1, buffer_size=10000, seed = 3):
      dataset, validation = train_test_split(words['as_int'], test_size=split_testing, random_state=seed)
      def split(input):
          word_dataset = tf.data.Dataset.from_tensor_slices(words['as_int'])
          sequences = word_dataset.batch(sequence_length+1, drop_remainder=True)
          dataset_unshuffled = sequences.map(split_input_target)
          return dataset_unshuffled.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
      self.training, self.validation = split(dataset), split(validation)

  def generate_text(self, model, start_string = "Antifa calls for", num_generate=50, temperature = 1.0, num_samples=1, print_text=False, resetable=False):
      input_eval = [self.words['map_from'][w] for w in process_new_text(start_string)]
      input_eval = tf.expand_dims(input_eval, 0)
      text_generated = []
      if resetable: model.reset_states()
      for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=num_samples)[-1,0].numpy()
        text_generated.append(self.words['map_to'][predicted_id])

        input_eval = text_generated
        if len(input_eval)>10:
            input_eval = input_eval[-10:]
        input_eval = [self.words['map_from'][w] for w in input_eval]
        input_eval = tf.expand_dims(input_eval, 0)

      ret = start_string
      for w in text_generated:
        if w in punctuation:
          ret += w
        else:
          ret += ' ' + w
      if print_text: 
        print(ret)
        print()
      return ret

  def tune_LSTM(self,sequence_length=10, epochs=5, mname=None, embedding_dim=256, rnn_units=1024, bs=64, num_layers=1, lr=.001, epsilon=1e-08,
                              normalize=0,dropout_rate=.2, regularize_rate=0):
      def init_LSTMmodel(embedding_dim=256, num_layers=2,sequence_length=10, rnn_units=1024, bs=64,normalize=0,
                   dropout_rate=.2, regularize_rate=.01):
          inputs = tf.keras.Input(batch_input_shape=[bs, None])
          x = Embedding(words['nunique'], embedding_dim, input_length=sequence_length,batch_input_shape=[bs, None])(inputs)
          if normalize == 1:
            x = tf.keras.layers.Normalization()(x)
          for i in range(num_layers): 
            x = LSTM(int(rnn_units/num_layers), return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
            if dropout_rate > 0.0:
              x = tf.keras.layers.Dropout(dropout_rate)(x)
          if regularize_rate >0.0:
            x = Dense(words['nunique'], activation='softmax', activity_regularizer=tf.keras.regularizers.L2(regularize_rate)) (x)
          else:
            x = Dense(words['nunique'], activation='softmax')(x)
          model = tf.keras.Model(inputs,x)
          return model

      if self.training == None: self.dataset(sequence_length, bs, 0.1)
      model = init_LSTMmodel(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=bs,
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)

      optimizer = tf.keras.optimizers.Adamax(learning_rate=lr,epsilon=epsilon,name='Adamax')
      model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

      if mname is None:
        mname = model_name(sequence_length, epochs)
      checkpoint_dir = models_dir / 'LSTM_training_checkpoints'
      checkpoint_model_dir = checkpoint_dir / mname
      checkpoint_prefix = checkpoint_model_dir / "ckpt_{epoch}"
      checkpoint_callback=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=False)]

      model.fit(self.training, validation_data=self.validation, epochs=epochs, callbacks=checkpoint_callback)
      model.weights

      model = init_LSTMmodel(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=1, 
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)
      
      model.load_weights(tf.train.latest_checkpoint(checkpoint_model_dir)).expect_partial()
      model.build(tf.TensorShape([1, None]))
      model_names[model] = mname
      shutil.rmtree(checkpoint_dir)
      self.tuned_LSTM = model

  def tune_GRU(self,sequence_length=10, epochs=5, mname=None, embedding_dim=256, rnn_units=1024, bs=64, num_layers=1, lr=.001, epsilon=1e-08,
                              normalize=0,dropout_rate=.2, regularize_rate=0):
      def init_GRUmodel(embedding_dim=256, num_layers=2,sequence_length=10, rnn_units=1024, bs=64,normalize=0,
                   dropout_rate=.2, regularize_rate=.01):
          inputs = tf.keras.Input(batch_input_shape=[bs, None])
          x = Embedding(words['nunique'], embedding_dim, input_length=sequence_length,batch_input_shape=[bs, None])(inputs)
          if normalize == 1:
            x = tf.keras.layers.Normalization()(x)
          for i in range(num_layers): 
            x = GRU(rnn_units, return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform')(x)
            if dropout_rate > 0.0:
              x = tf.keras.layers.Dropout(dropout_rate)(x)
          if regularize_rate >0.0:
            x = Dense(words['nunique'], activation='softmax', activity_regularizer=tf.keras.regularizers.L2(regularize_rate)) (x)
          else:
            x = Dense(words['nunique'], activation='softmax')(x)
          model = tf.keras.Model(inputs,x)
          return model

      if self.training == None: self.dataset(sequence_length, bs, 0.1)
      model = init_GRUmodel(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=bs,
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)

      optimizer = tf.keras.optimizers.Adamax(learning_rate=lr,epsilon=epsilon,name='Adamax')
      model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

      if mname is None:
        mname = model_name(sequence_length, epochs)
      checkpoint_dir = models_dir / 'GRU_training_checkpoints'
      checkpoint_model_dir = checkpoint_dir / mname
      checkpoint_prefix = checkpoint_model_dir / "ckpt_{epoch}"
      checkpoint_callback=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=False)]

      model.fit(self.training, validation_data=self.validation,epochs=epochs, callbacks=checkpoint_callback)

      model = init_GRUmodel(embedding_dim=embedding_dim, sequence_length=sequence_length, num_layers=num_layers, rnn_units=rnn_units, bs=1, 
                         normalize=normalize, dropout_rate=dropout_rate,  regularize_rate=regularize_rate)
      
      model.load_weights(tf.train.latest_checkpoint(checkpoint_model_dir)).expect_partial()
      model.build(tf.TensorShape([1, None]))
      model_names[model] = mname
      shutil.rmtree(checkpoint_dir)
      self.tuned_GRU = model

  def tune_final_best_model(self, epochs=60, mname=None, bs=200, sequence_length = 10, config = False,embedding_dim=500):
      if config: self.dataset(sequence_length, bs, 0.4)
      elif self.training == None: self.dataset(sequence_length, bs, 0.1)
      import keras_tuner as kt
      def init_final_best_model(hp,bs=bs,sequence_length=sequence_length,embedding_dim=embedding_dim):
          num_layers = hp.Choice('num_layers', [1, 2])
          rnn_units = hp.Choice('rnn_units', [512, 1024])
          L2 = hp.Choice('L2', [0.005, 0.01, 0.03])
          DROP = hp.Choice('drop_percentage', [0.0, .2])
          epsilon = hp.Choice('epsilon', [1e-06])
          beta_2 = hp.Choice('beta_2', [0.999])
          beta_1 = hp.Choice('beta_1', [0.9])
          lr = hp.Choice('lr', [0.05])
          normaliztion = hp.Choice("normalize", [1,0])
          
          optimizer = tf.keras.optimizers.Adamax(learning_rate=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,name='Adamax')

          inputs = tf.keras.Input(batch_input_shape=[bs, None])
          x = Embedding(words['nunique'], embedding_dim, input_length=sequence_length,batch_input_shape=[bs, None])(inputs)
          if normaliztion == 1: x = tf.keras.layers.Normalization()(x)
          for _ in range(num_layers): 
            x = GRU(rnn_units, return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform',activity_regularizer=tf.keras.regularizers.L2(L2))(x)
            x = tf.keras.layers.Dropout(DROP)(x)
          x = Dense(words['nunique'], activation='softmax', activity_regularizer=tf.keras.regularizers.L2(L2))(x)

          model = tf.keras.Model(inputs,x)
          model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
          collect()
          return model
      
      if mname is None:
        mname = model_name(sequence_length, epochs)
      
      if config: 
        checkpoint_callback=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)]
      else: 
        checkpoint_dir = models_dir / 'hypertuned_training_checkpoints'
        checkpoint_model_dir = checkpoint_dir / mname
        checkpoint_prefix = checkpoint_model_dir / "ckpt_{epoch}"
        checkpoint_callback=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]
        model = init_final_best_model(self.hyperperameter,bs=bs)
        model.summary()
        model.fit(self.training, validation_data=self.validation,epochs=epochs, callbacks=checkpoint_callback)
        model = init_final_best_model(self.hyperperameter,bs=1)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_model_dir)).expect_partial()
        model.build(tf.TensorShape([1, None]))
        model_names[model] = mname
        shutil.rmtree(checkpoint_dir)
        self.tuned_model = model
        #save(model)

      if config:
        tuner = kt.Hyperband(init_final_best_model,objective='val_loss',max_epochs=100,hyperband_iterations=5)
        tuner.search(self.training, validation_data=self.validation,epochs=epochs, callbacks=checkpoint_callback)
        self.tuner = tuner
        self.hyperperameter = tuner.get_best_hyperparameters()[0]
        pickle.dump(self.hyperperameter,open( "tuner.pkl", "wb" ))
        pickle.dump(self.hyperperameter,open( "tuner_save.pkl", "wb" ))
        self.tune_final_best_model(config=False)
            
  def GAN_failure(self):
    ''' 
    Progress on this model was halted mid-way when it became apparant that the GAN loss between generator and discriminator are non-differentiable if taken as a sample of most probable words, like we have been doing. 
    Another GAN model using transformers way replace this if time permits. 
  '''
    def create_text_generator_GAN_LSTM(sequence_length=10, num_training_epochs=5, mname=None, lr_gen=0.005, lr_des=0.005, bs=64, temperature=1):
      def GAN_loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
      def GAN_des_loss(labels, logits):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
      def init_GAN_LSTMmodel(embedding_dim=256, rnn_units=1024, bs=64):
        inputs = tf.keras.Input(batch_input_shape=[bs, None])
        x = Embedding(words['nunique'], embedding_dim, input_length=10,batch_input_shape=[bs, None])(inputs)
        x = LSTM(int(rnn_units/2), return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
        x = LSTM(int(rnn_units/2), return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
        x = Dense(words['nunique'], activation='softmax')(x)
        return tf.keras.Model(inputs,x)
      def init_GAN_des(embedding_dim=256, rnn_units=1024, bs=64):
        layers = [
            Embedding(words['nunique'], embedding_dim, input_length=10,batch_input_shape=[bs, None]),
            LSTM(int(rnn_units/2), return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            LSTM(int(rnn_units/2), return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            Dense(1, activation='sigmoid')]
        return Sequential(layers)
      from itertools import islice
      dataset = tf.data.Dataset.from_tensor_slices(words['as_int']).batch(sequence_length+1, drop_remainder=True).map(split_input_target)
      x = tf.data.Dataset.from_tensor_slices([x for (x,_) in dataset]).batch(bs, drop_remainder=True)
      y = tf.data.Dataset.from_tensor_slices([y for (_,y) in dataset]).batch(bs, drop_remainder=True)
      dataset = tf.data.Dataset.zip((x,y))

      discriminator = init_GAN_des(bs=bs)
      #discriminator.build(tf.TensorShape([bs, None]))
      model = init_GAN_LSTMmodel(bs=bs)
      discriminator.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
      #model.compile(optimizer='adam', loss=loss)  # , jit_compile=True

      if mname is None:
        mname = model_name(sequence_length, num_training_epochs)
      checkpoint_dir = models_dir / 'GAN_LSTM_training_checkpoints'
      checkpoint_model_dir = checkpoint_dir / mname
      checkpoint_prefix = checkpoint_model_dir / "ckpt_{epoch}"
      checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

      #model.fit(dataset, epochs=1, callbacks=[checkpoint_callback])
      
      dataset = tf.data.Dataset.from_tensor_slices(words['as_int']).batch(sequence_length+1, drop_remainder=True).map(split_input_target)#.shuffle(100000)
      x = tf.data.Dataset.from_tensor_slices([x for (x,_) in dataset])
      y = tf.data.Dataset.from_tensor_slices([y for (_,y) in dataset])
      dataset = tf.data.Dataset.zip((x,y))
      no = np.array([0]*64).reshape([bs,1])
      yes = np.array([1]*64).reshape([bs,1])
      #model.compile(optimizer='adam', loss=GAN_loss)
      
      def fix_shape(inputs):
          fake = np.array([[tf.random.categorical(fake / temperature, num_samples=1)[-1,0].numpy()] for fake in inputs])
          return tf.concat([np.delete(xx,[0],1),fake.reshape([bs,1])],1)

      @tf.function()#input_signature=[tf.TensorSpec(None, tf.int64)]
      def tf_function(input):
        y = tf.numpy_function(fix_shape, list(input), tf.int64)
        return y

      # https://machinelearningmastery.com/how-to-code-the-generative-adversarial-network-training-algorithm-and-loss-functions/
      def gan(generator, discriminator):
        discriminator.trainable = False
        generator.trainable = True
        model = Sequential()
        model.add(generator)
        #model.add(tf.keras.layers.Lambda(tf_function, name="tf_function"))
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        #discriminator.trainable = True
        return model
      gan=gan(model, discriminator)

      i = 0
      while True:
        discriminator.reset_states()
        model.reset_states()
        
        try:
          xx = np.array(list(islice(x, int(bs*i), int(bs*(i+1)))))
          yy = np.array(list(islice(y, int(bs*i), int(bs*(i+1)))))
        except: break
        i+=1
        fake = model.predict_on_batch(xx)
        fake = np.array([[tf.random.categorical(fake / temperature, num_samples=1)[-1,0].numpy()] for fake in fake])
        fake = tf.concat([np.delete(xx,[0],1),fake.reshape([bs,1])],1)
        discriminator.train_on_batch(x=fake, y = no)
        discriminator.train_on_batch(x=yy, y = yes)

        result = discriminator.predict_on_batch(fake)
        #result = np.array([result[-1] for result in result])
        #print(result.shape,result)
        print(result.shape)
        print(xx.shape)
        print(fake.shape)
        
        gan.train_on_batch(x=xx, y=yes)
        #model.train_on_batch(x=xx, y=result)
        break
      

      model = init_GAN_LSTMmodel(bs=1)
      model.load_weights(tf.train.latest_checkpoint(checkpoint_model_dir)).expect_partial()
      model.build(tf.TensorShape([1, None]))
      model_names[model] = mname
      shutil.rmtree(checkpoint_dir)

      return model
    model3 = create_text_generator_GAN_LSTM(num_training_epochs=20)
    print(generate_text(model3, "Antifa calls for", temperature = 1))
  ##########################################################################################
  # John's evaluative functions
  ##########################################################################################
  def split_train_test(self):
        sents = list(self.corpus.sents())
        shuffle(sents)
        cutoff = int(0.8*len(sents))
        training_set = sents[:cutoff]
        test_set = [[word.lower() for word in sent] for sent in sents[cutoff:]]
        return training_set, test_set
  def calculate_smoothing(self,sentences, bigram, smoothing_function, parameter):
        total_log_prob = 0
        test_token_count = 0
        for sentence in sentences:
            test_token_count += len(sentence) + 1 # have to consider the end token
            total_log_prob += smoothing_function(sentence, bigram, parameter)
        return math.exp(-total_log_prob / test_token_count)
  def smoothing(self):
    class Trigram():
      # Imported from lab 6 (John Rutledge's)
      def __init__(self):
          self.trigram_counts = defaultdict(Counter)
          self.bigram_counts = defaultdict(Counter)
          self.unigram_counts = Counter()
          self.context = defaultdict(Counter)
          self.tri_context = defaultdict(Counter)
          self.start_count = 0
          self.token_count = 0
          self.vocab_count = 0
      
      def convert_sentence(self, sentence):
          return ["<s>"] + [w.lower() for w in sentence] + ["</s>"]
      
      def get_counts(self, sentences):
          # collect unigram counts
          for sentence in sentences:
              sentence = self.convert_sentence(sentence)
              for word in sentence[1:]:  # from 1, because we don't need the <s> token
                  self.unigram_counts[word] += 1
              self.start_count += 1
              
          # collect bigram counts
          for sentence in sentences:
              sentence = self.convert_sentence(sentence)
              bigram_list = zip(sentence[:-1], sentence[1:])
              for bigram in bigram_list:
                  self.bigram_counts[bigram[0]][bigram[1]] += 1
                  self.context[bigram[1]][bigram[0]] += 1

          # collect trigram counts
          for sentence in sentences:
              sentence = self.convert_sentence(sentence)
              trigram_list = zip(sentence[0:], sentence[1:], sentence[2:])
              for w1,w2,w3 in trigram_list:
                  self.trigram_counts[(w1,w2)][w3] += 1
                  self.tri_context[w3][(w1,w2)] += 1
                  
          self.token_count = sum(self.unigram_counts.values())
          self.vocab_count = len(self.unigram_counts.keys())
    

    self.trigram = Trigram()
    self.trigram.get_counts(sent_tokenize(self.corpus))
  def Interpolate_Trigram(self, test_set):
      """Input text"""
      self.smoothing()
      test_set = nltk.Text(test_set)
      def interpolation(sentence, trigram, lambdas):
        bigram_lambda = lambdas[0]
        unigram_lambda = lambdas[1]
        trigram_lambda = lambdas[2]
        zerogram_lambda = 1 - unigram_lambda - bigram_lambda - trigram_lambda
        
        sentence = trigram.convert_sentence(sentence)
        bigram_list = zip(sentence[:-1], sentence[1:])
        trigram_list = list(zip(*[sentence[x:] for x in range(0, 3)]))
        prob = 0
        for w1, prev_word, word in trigram_list:
            # bigram probability
            sm_trigram_counts = trigram.trigram_counts[(w1,prev_word)][word]
            sm_bigram_counts = trigram.bigram_counts[prev_word][word]
            if sm_bigram_counts == 0: interp_bigram_counts = 0
            else:
                if prev_word == "<s>": u_counts = trigram.start_count
                else: u_counts = trigram.unigram_counts[w1]
                interp_bigram_counts = sm_bigram_counts / (float(u_counts) * bigram_lambda + [1 if float(float(u_counts) * bigram_lambda)==0 else 0][0])
                
            if sm_trigram_counts == 0: interp_trigram_counts = 0
            else:
                if prev_word == "<s>": u_counts = trigram.start_count
                else: u_counts = trigram.unigram_counts[w1]
                interp_trigram_counts = sm_trigram_counts / (float(u_counts) * trigram_lambda + [1 if float(float(u_counts) * trigram_lambda)==0 else 0][0])

            # unigram probability
            interp_unigram_counts = (trigram.unigram_counts[word] / trigram.token_count) * unigram_lambda

            # "zerogram" probability: this is to account for out-of-vocabulary words, this is just 1 / |V|
            vocab_size = len(trigram.unigram_counts)
            interp_zerogram_counts = (1 / float(vocab_size)) * zerogram_lambda
        
            prob += math.log(interp_trigram_counts + interp_bigram_counts + interp_unigram_counts + interp_zerogram_counts)
        return prob

      self.trigram.get_counts(self.corpus)
      return self.calculate_smoothing(test_set, self.trigram, interpolation, (0.7, 0.19, .1))
  def loss(self):
    pass

##########################################################################################
# End of John's experiments
##########################################################################################

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

second_iteration = tuned_models(words,chars,corpus)