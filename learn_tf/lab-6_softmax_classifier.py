import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
 
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
#4개의 속성을 가지고 있습니다.
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
#one-hot encoding이 이미 이루어져있습니다.
 
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3 #class의 개수를 의미합니다.
 
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
 
# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# score를 logit이라고도 부른다. tf.nn.softmax를 사용하지 않고도 간단히 구현가능합니다.
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
 
# Cross entropy cost/loss 함수 구현입니다.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
 
#복잡한 미분을 직접 계산할 필요없이 tensorflow가 알아서 계산을 해줍니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
 
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
 
    print('------ test a --------')
 
    # Testing & One-hot encoding
    a = sess.run( hypothesis, feed_dict={X: [[1, 11, 7, 9]]}) #학습이 끝난뒤 hypothesis에 새로운 값을 넘겨줍니다.
    print(a, sess.run(tf.arg_max(a, 1))) #arg_max는 두번째 인자로 넘겨준 1 차원의 argument중에서 가장 큰 값을 반환합니다.
 
    print('------test b--------')
 
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1)))
 
    print('--------------')
 
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.arg_max(c, 1)))
 
    print('--------------')
 
    all = sess.run(hypothesis, feed_dict={
                   X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1))) #한 번에 여러개의 값을 확인해볼 수있습니다.

 