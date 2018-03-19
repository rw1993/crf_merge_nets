import tensorflow as tf
import tflearn

def cnn(net, batch_size, dimensions, timestep):
    net = tf.reshape(net, (batch_size, dimensions, timestep))
    net = tflearn.conv_1d(net, dimensions*2, 3, activation="relu", weights_init="truncated_normal")
    net = tflearn.conv_1d(net, dimensions*4, 3, activation="relu", weights_init="truncated_normal")
    net = tflearn.conv_1d(net, dimensions*2, 3, activation="relu", weights_init="truncated_normal")
    net = tflearn.conv_1d(net, dimensions, 3, activation="relu", weights_init="truncated_normal")
    net = tflearn.flatten(net)
    net = tflearn.fully_connected(net, 256, activation="relu", weights_init="truncated_normal")
    net = tflearn.fully_connected(net, 128, activation="relu", weights_init="truncated_normal")
    net = tflearn.fully_connected(net, dimensions*2, activation="sigmoid", weights_init="truncated_normal")
    net = tf.reshape(net, (batch_size, dimensions, 2))
    return tf.nn.softmax(net)



def main():
    batch_size = 1
    dimensions = 40
    timestep = 10
    test_tensor = tf.placeholder(tf.float32, shape=(batch_size, timestep, dimensions))
    net = cnn(net=test_tensor, batch_size=batch_size, dimensions=dimensions, timestep=timestep)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()