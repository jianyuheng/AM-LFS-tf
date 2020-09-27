import tensorflow as tf
import numpy as np

def t_fun(x, theta, x_range = [-1, 1]):
    M = 6
    y = 0.0
    interval = (x_range[1] - x_range[0]) / M
    theta = tf.cast(theta, tf.float32)
    x_range = tf.cast(x_range, tf.float32)

    for m in range(M):
        if m == M - 1:
            ind = tf.cast(x >= (x_range[0] + interval * m), tf.float32)
        elif m == 0:
            ind = tf.cast(x < (x_range[0] + interval * (m + 1)), tf.float32)
        else:
            ind = tf.cast(x >= (x_range[0] + interval * m), tf.float32) * tf.cast(x < (x_range[0] + interval * (m + 1)), tf.float32)

        y += (x * theta[m] + theta[m+M]) * ind
    return y

def loss_func(y, cos_x, theta, class_num, batch_size):
    y_onehot = tf.one_hot(y, class_num)
    t1_cos_x = t_fun(cos_x, theta)
    loss = -tf.nn.log_softmax(
                t1_cos_x * y_onehot + cos_x * (1 - y_onehot), axis=1)

    line=tf.expand_dims(tf.range(0, batch_size, 1), axis=1)
    h_index = tf.expand_dims(y, axis=1)
    index = tf.concat([line, h_index], axis=1)
    loss = tf.gather_nd(loss, index)

    return tf.reduce_mean(loss)


if __name__ == '__main__':

    a = tf.random_uniform([10,5])*2-1

    theta = tf.constant([1,2,3,0,0,0, 1,2,3,0,0,0])
    print(t_fun(a, theta, tf.constant([-1,1])))

