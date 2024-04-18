# EX : 06 Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The aim is to develop a Bidirectional LSTM-based RNN model to identify specific items in text by vectorizing phrases using embedding techniques.
<img src="https://github.com/Jenishajustin/named-entity-recognition/assets/119405070/dd172e64-0bc6-49ca-84db-2cae786ab353" height=450 width=450>


## DESIGN STEPS

### STEP 1:
Import the necessary libraries.
### STEP 2:
Fill the NULL values in the dataset if exist.
### STEP 3:
Create a list and find the number of unique words and tags in the dataset.
### STEP 4:
Create a dictionary for the words and tags and their respective Index values.
### STEP 5:
Build the Bi-directional LSTM based model.
### STEP 6:
Compile, fit ,transform and plot.
## PROGRAM
```
Developed By : J.JENISHA
Reg. No. : 212222230056
```

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)
data = data.fillna(method="ffill")
data.head(50)


print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())
print("Unique words are:",words)
print("Unique tags are:", tags)
num_words = len(words)
num_tags = len(tags)
num_words
num_tags

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
sentences[0]
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
word2idx
tag2idx
plt.hist([len(s) for s in sentences], bins=50)
plt.show()
X1 = [[word2idx[w[0]] for w in s] for s in sentences]
type(X1[0])
X1[0]
max_len = 50
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
X[0]
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
X_train[0]
y_train[0]

input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                                   input_length=max_len)(input_word)
dropout = layers.SpatialDropout1D(0.1)(embedding_layer)
bid_lstm = layers.Bidirectional(
    layers.LSTM(units=150,return_sequences=True,
                recurrent_dropout=0.1))(dropout)
output = layers.TimeDistributed(
    layers.Dense(num_tags,activation="softmax"))(bid_lstm)
model = Model(input_word, output)
model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test),
          batch_size=50, epochs=3,)
metrics = pd.DataFrame(model.history.history)
metrics.head()

print("J JENISHA\n212222230056")
metrics[['accuracy','val_accuracy']].plot()
print("J JENISHA\n212222230056")
metrics[['loss','val_loss']].plot()

i = 300
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
<img src="https://github.com/Jenishajustin/named-entity-recognition/assets/119405070/f79f1de7-421f-4819-9171-a8305611c9dc" height=450 width=450>

### Training accuracy, Validation accuracy Vs Iteration Plot
<img src="https://github.com/Jenishajustin/named-entity-recognition/assets/119405070/1ad58484-7563-4669-93dc-5ca58bcbb302" height=450 width=450>


### Sample Text Prediction
<img src="https://github.com/Jenishajustin/named-entity-recognition/assets/119405070/75ef8733-19fe-4598-b266-de848b0fb9aa" height=550 width=450>


## RESULT
Bidirectional LSTM-based RNN model for Named Entity Recognition using embedding techniques is developed successfully.
