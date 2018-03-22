import tensorflow as tf
import tflearn

def lstm(net, dropout, batch_size, dimensions, **kw):
    net = tflearn.layers.recurrent.lstm(net,
                                        n_units=128,
                                        activation="softsign",
                                        weights_init="xavier",
                                        return_seq=True)
    net = tf.nn.dropout(net, dropout)
    net = tflearn.layers.recurrent.lstm(net,
                                        n_units=256,
                                        activation="softsign",
                                        weights_init="xavier",
                                        return_seq=True)
    net = tf.nn.dropout(net, dropout)
    net = tflearn.layers.recurrent.lstm(net,
                                        n_units=512,
                                        activation="softsign",
                                        weights_init="xavier",
                                        return_seq=True)
    net = tf.nn.dropout(net, dropout)
    net = tflearn.layers.recurrent.lstm(net,
                                        n_units=256,
                                        activation="softsign",
                                        weights_init="xavier",
                                        return_seq=True)
    net = tf.nn.dropout(net, dropout)
 
    net = tflearn.layers.recurrent.lstm(net,
                                        n_units=128,
                                        activation="softsign",
                                        weights_init="xavier",
                                        return_seq=False)
    net = tf.nn.dropout(net, dropout)
    net = tflearn.fully_connected(net, n_units=dimensions*2, activation="relu")
    net  = tf.reshape(net, (batch_size, dimensions, 2))
    return tf.nn.softmax(net)