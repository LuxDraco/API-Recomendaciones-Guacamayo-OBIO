import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

import tensorflow.keras as tf


def predicciones(user_id_param):
    ratings_df = pd.read_csv("ventas.csv")
    ratings_df.head()

    print(ratings_df.shape)
    print(ratings_df.user_id.nunique())
    print(ratings_df.product_id.nunique())
    ratings_df.isna().sum()

    from sklearn.model_selection import train_test_split

    Xtrain, Xtest = train_test_split(ratings_df, test_size=0.2, random_state=1)
    print(f"Shape of train data: {Xtrain.shape}")
    print(f"Shape of test data: {Xtest.shape}")

    # Get the number of unique entities in books and users columns
    nbook_id = ratings_df.product_id.nunique()
    nuser_id = ratings_df.user_id.nunique()

    # Book input network
    input_books = tf.layers.Input(shape=[1])
    embed_books = tf.layers.Embedding(nbook_id + 1, 15)(input_books)
    books_out = tf.layers.Flatten()(embed_books)

    # user input network
    input_users = tf.layers.Input(shape=[1])
    embed_users = tf.layers.Embedding(nuser_id + 1, 15)(input_users)
    users_out = tf.layers.Flatten()(embed_users)

    conc_layer = tf.layers.Concatenate()([books_out, users_out])
    x = tf.layers.Dense(512, activation='relu')(conc_layer)
    x_out = x = tf.layers.Dense(1, activation='relu')(x)
    model = tf.Model([input_books, input_users], x_out)

    opt = tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.summary()

    hist = model.fit([Xtrain.product_id, Xtrain.user_id], Xtrain.rating,
                     batch_size=16,
                     epochs=100,
                     verbose=1,
                     validation_data=([Xtest.product_id, Xtest.user_id], Xtest.rating))

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(train_loss, color='r', label='Train Loss')
    plt.plot(val_loss, color='b', label='Validation Loss')
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    #plt.show()

    # model.save('model')

    products_array = np.zeros(7)
    for i in range(0, 7):
        products_array[i] = i + 1

    users_array = np.full(7, user_id_param)

    print(products_array)
    print(users_array)
    pred = model.predict([products_array, users_array])
    return pred.tolist()


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello from Guacamayo Obio'


@app.route('/prediccion', methods=['POST'])
def init_pred():
    rr = predicciones(float(request.args.get('userid')))
    print(rr)
    return jsonify(rr)


if __name__ == "__main__":
    app.run()
