from merge import get_loss, lstm_only, merge
from batch import generate_batch
from read_data import get_data
import correlation
import tensorflow as tf

def train(batch_size=8, timestep=10, data_path="yfj.csv", train_net=""):
    nomalize_data  = get_data(data_path)
    _, dimensions = nomalize_data.shape
    pearson_dict = correlation.get_pearson(data_path)
    distances = [pearson_dict]
    if train_net is None:
        net, bx_tensor, dropout_tensor = merge(batch_size=batch_size, timestep=timestep,
                                            distances=distances, dimensions=dimensions)
    elif train_net == "lstm":
        net, bx_tensor, dropout_tensor = lstm_only(batch_size=batch_size, timestep=timestep,
                                            distances=distances, dimensions=dimensions)
    loss_tensor, by_tensor = get_loss(net, batch_size, dimensions)
    learning_rate_tensor = tf.placeholder(dtype=tf.float32, shape=())
    #optimizer = tf.train.AdamOptimizer(learning_rate_tensor)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_tensor, momentum=0.9)
    gradients = optimizer.compute_gradients(loss_tensor)
    cilp_gradients = [(tf.clip_by_value(g, -5.0, 5.0), v) for g, v in gradients if g is not None]
    train_step = optimizer.apply_gradients(cilp_gradients)
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", loss_tensor)
    merge_summary_op = tf.summary.merge_all()
    step = 0
    learning_rate = 0.1
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("{}log".format(train_net), graph=sess.graph)
        sess.run(init)
        recent_avgloss = 0.0
        saver = tf.train.Saver()
        former_acc = 0.0
        print("builded")
        for bx, by in generate_batch(batch_size=batch_size, timestep=timestep, dimensions=dimensions,
                                     data_path=data_path, data_set="train"):
            _, summary_string, loss = sess.run([train_step, merge_summary_op, loss_tensor],
                                                feed_dict={bx_tensor: bx,
                                                dropout_tensor: 1.0,
                                                learning_rate_tensor:
                                                learning_rate,
                                                by_tensor: by})
            summary_writer.add_summary(summary_string, step)
            step += 1
            # print loss, step
            recent_avgloss += loss
            if step % 100 == 0:
                print(recent_avgloss / 100.0, step)
                recent_avgloss = 0.0
            if step % 10000 == 0:
                saver.save(sess, "{}model/".format(train_net), global_step=step)
                # valid
                valid_num = 10000
                total = 0
                acc = 0.0
                valid = 0
                for bx, by in generate_batch(batch_size=batch_size, timestep=timestep,
                                             dimensions=dimensions, data_path=data_path,
                                             data_set="valid"):
                    q_result, = sess.run([net,], feed_dict={bx_tensor: bx,
                                                dropout_tensor: 1.0,
                                                learning_rate_tensor: learning_rate,
                                                by_tensor: by})
                    valid += 1
                    for b in range(batch_size):
                        for i in range(dimensions):
                            total += 1
                            if q_result[b][i][0] > q_result[b][i][1]:
                                if by[b][i] == 0:
                                    acc += 1
                            else:
                                if by[b][i] == 1:
                                    acc += 1
                    if valid >= valid_num:
                        break
                if float(acc) / total < former_acc:
                    learning_rate /= 10.0
                former_acc = float(acc) / total
                print("valid acc is {} after step {}".format(float(acc)/total, step))
def main():
    train(train_net="lstm")

if __name__ == '__main__':
    main()