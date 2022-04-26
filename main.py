from nlp_project import *

model = create_text_generator(num_training_epochs=20)

print(generate_text(model, "Antifa calls for"))

model2 = create_text_generatorLSTM(num_training_epochs=20)

print(generate_text(model2, "Antifa calls for"))

model3 = create_text_generatorLSTM(num_training_epochs=20, num_layers=2)

print(generate_text(model3, "Antifa calls for"))



              
# GAN Loss
# Preprocessing using NLTK/Spacy
# LSTM model"
   
