import string

import numpy as np
from keras.datasets import imdb
from tensorflow import keras


def predict(review):

    # Load model
    model = keras.models.load_model('reviews.h5')
    print("model loaded")

    # Prepare the text
    # 1) remove punctuation
    # 2) split words by whitespace
    # 3) remove numerics because later text is mapped to numerics
    translator = str.maketrans('', '', string.punctuation)
    review_edited = review.translate(translator)
    review_edited = review.translate(string.punctuation)
    review_edited = review_edited.lower().split(' ')
    review_edited = [word for word in review_edited if word.isalpha()]
    print(review_edited)

    # We need the same dictionary as for training
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown, any word that is not inside the dictionary
    word_index["<UNUSED>"] = 3

    # Map sentence to array
    tensor = [1]
    for word in review_edited:
        if word in word_index and word_index[word] < 10000:  # limit to first 10000 words
            tensor.append(word_index[word])
        else:
            tensor.append(2)

    # Create input tensor
    padded_input = keras.preprocessing.sequence.pad_sequences([tensor],
                                                              maxlen=256,
                                                              padding='post',
                                                              value=word_index["<PAD>"])

    # Do prediction
    prediction = model.predict(np.array([padded_input][0]))[0][0]

    # Do classification
    if prediction >= 0.5:
        classification = "good"
    else:
        classification = "bad"

    print(prediction, classification)
    return prediction, classification
