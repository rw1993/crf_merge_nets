from lstm import lstm
from dnn import dnn
from mfi_function import crf_layer
import tensorflow as tf
from cnn import cnn

def lstm_only(batch_size, dimensions , timestep, distances, **kw):
    bx_tenor = tf.placeholder(tf.float32, (batch_size, timestep, dimensions))
    dropout_tenor = tf.placeholder(tf.float32, ())
    with tf.variable_scope("lstm"):
        net = lstm(net=bx_tenor, dropout=dropout_tenor,
                batch_size=batch_size, dimensions=dimensions)
    return net, bx_tenor, dropout_tenor
 
def merge(batch_size, dimensions, timestep, distances):
    bx_tenor = tf.placeholder(tf.float32, (batch_size, timestep, dimensions))
    dropout_tenor = tf.placeholder(tf.float32, ())
    net0 = lstm(net=bx_tenor, dropout=dropout_tenor,
                batch_size=batch_size, dimensions=dimensions)
    net1 = dnn(bx_tenor, batch_size=batch_size,
               dimensions=dimensions, timestep=timestep)
    net2 = cnn(net=bx_tenor, batch_size=batch_size, dimensions=dimensions, timestep=timestep)
    
    base_nets = [net0, net1, net2]
    net = sum(base_nets) / len(base_nets)
    return crf_layer(net, batch_size, dimensions, distances), bx_tenor, dropout_tenor

def get_loss(net, batch_size, dimensions):
    by = tf.placeholder(tf.int32, (batch_size, dimensions))
    labels = tf.one_hot(by, 2)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=labels)
    return tf.reduce_sum(loss) / batch_size, by


def main():
    pass


if __name__ == '__main__':
    main()