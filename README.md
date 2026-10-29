import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Embedding, LSTM, Add
)

# Load and normalize Fashion-MNIST images
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0

class_names = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot"
]

train_captions = ["startseq a photo of a " + class_names[i] + " endseq"
                  for i in y_train[:4000]]
test_image = x_test[0]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_captions)
vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(c.split()) for c in train_captions)

X_img, X_seq, y_next = [], [], []

for image, caption in zip(x_train[:4000], train_captions):
    seq = tokenizer.texts_to_sequences([caption])[0]

    for t in range(1, len(seq)):
        partial = pad_sequences([seq[:t]], maxlen=max_len, padding="post")[0]
        X_img.append(image)
        X_seq.append(partial)
        y_next.append(seq[t])

X_img = np.array(X_img)
X_seq = np.array(X_seq)
y_next = to_categorical(y_next, num_classes=vocab_size)

image_input = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, activation="relu", padding="same")(image_input)
x = MaxPooling2D()(x)
x = Conv2D(64, 3, activation="relu", padding="same")(x)
x = MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
image_features = Dense(128, activation="relu")(x)

caption_input = Input(shape=(max_len,))
x = Embedding(vocab_size, 128, mask_zero=True)(caption_input)
caption_features = LSTM(128)(x)

x = Add()([image_features, caption_features])
x = Dense(128, activation="relu")(x)
output = Dense(vocab_size, activation="softmax")(x)

model = Model([image_input, caption_input], output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit([X_img, X_seq], y_next, epochs=5, batch_size=64, validation_split=0.1)

index_to_word = {index: word for word, index in tokenizer.word_index.items()}

caption = "startseq"
for _ in range(max_len):
    seq = tokenizer.texts_to_sequences([caption])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="post")
    prediction = model.predict([test_image[None, ...], seq], verbose=0)
    next_word = index_to_word.get(int(np.argmax(prediction)), "")

    if not next_word or next_word == "endseq":
        break

    caption += " " + next_word

print("Generated caption:", caption.replace("startseq", "").strip())
