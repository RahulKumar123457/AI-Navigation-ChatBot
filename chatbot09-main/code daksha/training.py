import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Function to preprocess patterns
def preprocess_patterns(pattern):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(pattern) if word not in ignore_words]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Preprocess patterns
        w = preprocess_patterns(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Shuffle documents
random.shuffle(documents)

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Batch processing
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch_documents = documents[i:i+batch_size]
    for doc in batch_documents:
        # Initializing bag of words
        bag = [1 if w in doc[0] else 0 for w in words]
        # Output is '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)



# Create train and test lists. X - patterns, Y - intents
X = np.array([t[0] for t in training])
Y = np.array([t[1] for t in training])

print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Y[0]), activation='softmax')) # Output layer

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=200, batch_size=5, verbose=1)


# Save the model
model.save('model.h5')

print("Model created and saved")