import tensorflow as tf
#3x3 matrix
input_data = [ 
                [1.,2.,3.],
                [1.,2.,3.],
                [2.,3.,4.] 
            ] 

x = tf.placeholder(dtype=tf.float32,shape=[None,3])
#3x1 matrix
w = tf.Variable([ [2.],[2.],[2.] ], dtype = tf.float32) 

y = tf.matmul(x,w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y,feed_dict={x:input_data})
sess.close()




# Creates a graph.
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True, this will dump 
# on the log how tensorflow is mapprint the operations on devices
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
sess.close()
