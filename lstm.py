import tensorflow as tf
import tflearn

def lstm(net, dropout, batch_size, dimensions):
    net = tflearn.layers.recurrent.lstm(net,
                                       n_units=dimensions*2,
                                       activation="softsign",
                                       weights_init="xavier")
    net = tf.nn.dropout(net, dropout)
    net = tflearn.fully_connected(net, n_units=dimensions*2, activation="relu")
    net  = tf.reshape(net, (batch_size, dimensions, 2))
    return tf.nn.softmax(net)