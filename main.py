from nlp_project import *

model = create_text_generator(num_training_epochs=20)

print(generate_text(model, "Antifa calls for"))

model2 = create_text_generatorLSTM(num_training_epochs=5)

print(generate_text(model2, "Antifa calls for"))
              
# GAN Loss
# Preprocessing using NLTK/Spacy
# LSTM model"
   
