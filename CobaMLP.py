import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import speech_recognition as sr
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


df = pd.read_csv("Data10k_Csv.csv", skiprows=0)
df.drop(columns=["Unnamed: 0", "tweet_id", "location", "username", "deEmoji"], inplace=True)
df.head()

df_ter = df[df['sentiment']=='Terindikasi Depresi']
df_tdk = df[df['sentiment']=='Tidak Terindikasi']
print(df_ter.shape)
print(df_tdk.shape)

df_clean = df_tdk.sample(df_ter.shape[0])

df_balanced = pd.concat([df_clean, df_ter])
df_balanced.shape

print(df_balanced['sentiment'].value_counts())

#Train Test Split
xtrain, xtest, ytrain, ytest = train_test_split(df_balanced['cleanTweet'], df_balanced['is_depress'],
                                                stratify=df_balanced['is_depress'], test_size=0.25, random_state=0)
print(xtest.value_counts(10))
##Parameter
vocab_size = 10000
max_length = 128
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(xtrain)
word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(xtrain)
xtrain = pad_sequences(train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_seq = tokenizer.texts_to_sequences(xtest)
xtest = pad_sequences(test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

trainlabels = np.expand_dims(ytrain, axis=1)
ytrain = np.array(trainlabels)

testlabels = np.expand_dims(ytest, axis=1)
ytest = np.array(testlabels)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

print(f'number of words in word_index: {len(word_index)}')
print(f'word_index: {word_index}')

##Model
embedding_dim = 128
dense_dim = 64

model = tf.keras.Sequential([
    tf.keras.layers.Dense(embedding_dim, activation='relu', input_dim=128),
    tf.keras.layers.Dense(dense_dim, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

matrik = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=matrik)

model.summary()

##Train Model
history = model.fit(xtrain, ytrain, epochs=10, validation_data=(xtest, ytest))

##Graphic
# Plot Utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

# Plot the accuracy and loss history
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


y_predict = model.predict(xtest)

y_predict = np.where(y_predict > 0.5, 1, 0)
y_predict

cm = confusion_matrix(ytest, y_predict)
cm

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("prediksi")
plt.ylabel("ketepatan")

print(classification_report(ytest, y_predict))

model.predict(xtrain)

#Speech Recognition
r = sr.Recognizer()
result = ""

with sr.Microphone() as source:
  print("Start Speaking")
  recording = r.listen(source)
  print("Times out")

  try:
    result = r.recognize_google(recording, language="id-ID")
    print(result)
  except r.UnknowValueError:
    print("Theres error, please try again")
  except Exception as e:
    print(e)

#Predict
#Convert the result to a token sequence
token_list = tokenizer.texts_to_sequences([result])[0]

# Pad the sequence
token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')

# Feed to the model and get the probabilities for each index
probabilities = model.predict(token_list)

# Menginterpretasikan hasil prediksi
if probabilities > 0.5:
    sentiment = "Positif"
else:
    sentiment = "Negatif"

# Menampilkan hasil prediksi
print("Hasil prediksi sentimen:", sentiment, probabilities)

# tidak terindikasi itu Nilainya besar, dan juga sebaliknya
# Get the index with the highest probability
predicted = np.argmax(probabilities, axis=-1)[0]

print(predicted)

# Save and Convert model to tflite
model.save("FinalModel_MLP.h5")
model.save_weights("Weights_MLP.h5")
tflite_model = tf.keras.models.load_model('FinalModel_MLP.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.uint8]
convert_model = converter.convert()
open ('nlp-mlp-optimizer.tflite', 'wb').write(convert_model)