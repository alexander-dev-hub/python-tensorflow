# Lab 4 Multi-variable linear regression
import tensorflow as tf

 
#H(X)= 1/(1+ e^( − W'X)

# cost(W)=−1/m ∑ylog(H(x))+(1−y)(log(1−H(x)))

# W:=W−α ∂/∂W cost(W)

tf.set_random_seed(777)  # for reproducibility

tf.set_random_seed(777)  # for reproducibility
 
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
#입력 값은 두 가지 속성을 가진 값입니다.
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]
#결과 값은 binary 형태를 가집니다.
 
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2]) #shape이 이제 중요해 집니다. 갯수는 n개라고 해서 None으로 주고 2차원임으로 2를 넘겨줍니다.
Y = tf.placeholder(tf.float32, shape=[None, 1])
 
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
 
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
 
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))
 
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#cost를 최소화하는 GradientDescent를 사용합니다.
 
# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#hypothesis가 0.5보다 크거나 작거나에따라 1과 0으로 casting을 합니다.
 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#결과값과 예측값이 같은지를 비교하교 평균을 구합니다.
 
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
 
    for step in range(10001):
        cost_val,  _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)
 
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    #학습이 종료된 이후에 정확도를 출력했지만 학습 중에 출력해도 좋을거 같다는 생각이 들었습니다.
 
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


