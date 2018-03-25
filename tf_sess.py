import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')

result = a + b

sess = tf.Session()
with sess.as_default():
    print(result.eval())

print(sess.run(result))
print(result.eval(session=sess))

sess = tf.InteractiveSession()  # sess is already a default Session
print(result.eval)
sess.close()

