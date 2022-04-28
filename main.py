from nlp_project import *

model = create_text_generator(num_training_epochs=20)

print(generate_text(model, "Antifa calls for"))

model2 = create_text_generatorLSTM(num_training_epochs=20)

print(generate_text(model2, "Antifa calls for"))

model3 = create_text_generatorLSTM(num_training_epochs=20, num_layers=2)

print(generate_text(model3, "Antifa calls for"))

embedding_dim = [128, 256, 512, 1024]
num_layers = [1, 2, 3 ]
rnn_units = [256, 512, 1024]
L2_rate =[0.0, 0.01, 0.02, 0.05]
DROP_rate = [0.0, 0.08, .1, .2, .5]
lr = [0.001,0.1,0.01,0.005]
normalization = [1,0]
epsilon = [1e-08,1e-07,1e-06,1e-05]
sequence_length=[10,15,20,25]

for seq_len in sequence_length:
  for layers in num_layers:
    for emb in embedding_dim:
      for units in rnn_units:
        for L2 in L2_rate:
          for drop in DROP_rate:
            for learningrate in lr:
              for normalize in normalization:
                for eps in epsilon:
                    modelLSTM = create_text_generatorLSTM(num_training_epochs=1,num_layers=layers, embedding_dim=emb, rnn_units=units, sequence_length=seq_len,
                                                      regularize_rate=L2, dropout_rate=drop, lr=learningrate, normalize=normalize, epsilon=eps)
                    print("results for LSTM text geration with",layers,"layer(s)","normalize:",bool(normalize),"droput rate:",drop,"regularize rate",L2, 
                          "embedings:",emb,"Units:",units,"epsilon:",eps,"input length:",seq_len) 
                    print(generate_text(modelLSTM, "Antifa calls for"))

                    modelGRU = create_text_generator(num_training_epochs=1,num_layers=layers, embedding_dim=emb, rnn_units=units, sequence_length=seq_len,
                                                      regularize_rate=L2, dropout_rate=drop, lr=learningrate, normalize=normalize, epsilon=eps)
                    print("results for GRU text geration with",layers,"layer(s)","normalize:",bool(normalize),"droput rate:",drop,"regularize rate",L2, 
                          "embedings:",emb,"Units:",units,"epsilon:",eps,"input length:",seq_len) 
                    print(generate_text(modelGRU, "Antifa calls for"))


for seq_len in sequence_length:
  for layers in num_layers:
    for emb in embedding_dim:
      for units in rnn_units:
        for L2 in L2_rate:
          for drop in DROP_rate:
            for learningrate in lr:
              for normalize in normalization:
                for eps in epsilon:
                  second_iteration.tune_LSTM(epochs=1,num_layers=layers, embedding_dim=emb, rnn_units=units, sequence_length=seq_len,
                                                      regularize_rate=L2, dropout_rate=drop, lr=learningrate, normalize=normalize, epsilon=eps)
                  print("results for LSTM text geration with",layers,"layer(s)","normalize:",bool(normalize),"droput rate:",drop,"regularize rate",L2, 
                          "embedings:",emb,"Units:",units,"epsilon:",eps,"input length:",seq_len) 
                  ltsm = second_iteration.generate_text(second_iteration.tuned_LSTM, print_text=True)

                  second_iteration.tune_GRU(epochs=1,num_layers=layers, embedding_dim=emb, rnn_units=units, sequence_length=seq_len,
                                                      regularize_rate=L2, dropout_rate=drop, lr=learningrate, normalize=normalize, epsilon=eps)
                  print("results for GRU text geration with",layers,"layer(s)","normalize:",bool(normalize),"droput rate:",drop,"regularize rate",L2, 
                          "embedings:",emb,"Units:",units,"epsilon:",eps,"input length:",seq_len) 
                  gru = second_iteration.generate_text(second_iteration.tuned_GRU, print_text=True)              


second_iteration.Interpolate_Trigram("repeated manic fists")  

#second_iteration.tuner
second_iteration.tuner.results_summary()

tuner = second_iteration.pickle_save()
pickle.dump(tuner.get_best_hyperparameters()[0],open( "tuner.pkl", "wb" ))