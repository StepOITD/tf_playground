import tensorflow as tf

v1 = tf.Variable(tf.constant(3.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")

result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # init_op = tf.initialize_all_variables()
    # sess.run(init_op)
    saver.restore(sess, "/home/constantine/PycharmProjects/tf_playground/checkpoints/test/model.ckpt")

    print(sess.run(result))

#######################################################################

import tensorflow as tf

saver = tf.train.import_meta_graph("/home/constantine/PycharmProjects/tf_playground/checkpoints/test/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess,
                  "/home/constantine/PycharmProjects/tf_playground/checkpoints/test/model.ckpt")

    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))