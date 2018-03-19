import tensorflow as tf
import tflearn


def dnn(net, batch_size, dimensions, timestep):
    net = tf.reshape(net, (batch_size, dimensions*timestep))
    net = tflearn.fully_connected(net,
                                  n_units=dimensions*timestep,
                                  activation="relu", weights_init="truncated_normal")
    net = tflearn.fully_connected(net,
                                  n_units=dimensions*timestep*2,
                                  activation="relu", weights_init="truncated_normal")
    net = tflearn.fully_connected(net,
                                  n_units=dimensions*timestep,
                                  activation="relu", weights_init="truncated_normal")
    net = tflearn.fully_connected(net,
                                  n_units=dimensions*timestep/2,
                                  activation="relu", weights_init="truncated_normal")
    net = tflearn.fully_connected(net,
                                  n_units=dimensions*2,
                                  activation="sigmoid", weights_init="truncated_normal")
    net = tf.reshape(net, (batch_size, dimensions, 2))
    return tf.nn.softmax(net)