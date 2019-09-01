import tensorflow as tf
import cv2
import numpy as np
import math
from scipy import ndimage


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted




# 뉴럴 네트워크 파라미터
n_hidden_1 = 256 # 첫번째 히든 레이어의 뉴런 개수
n_hidden_2 = 256 # 두번째 히든 레이어의 뉴런 개수
num_input = 784 # MNIST 데이터 입력( 28 x 28 이미지를 크기 784인 1차원 배열로 변환해서 사용)
num_classes = 10 # MNIST 데이터의 전체 클래스 개수 10개 ( 0 ~ 9 숫자 )

# 뉴럴 네트워크의 입력과 라벨
X = tf.placeholder(tf.float32, [None, num_input])


# 레이어(layer)의 weight와 bias를 저장할 변수를 선언합니다.
# 코드를 간결하게 하기 위해서 딕셔너리를 이용합니다.
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}



def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)
prediction = tf.nn.softmax(logits)



with tf.Session() as sess:

    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    # 저장된 모델 파라미터를 가져옵니다.
    model_path = "f:/tmp/mnist/model.saved"
    saver = tf.train.Saver()

    saver.restore(sess, model_path)
    print("Model restored from file: %s" % model_path)



    # 10개의 이미지를 위한 배열을 생성
    images = np.zeros((10, 784))

    i = 0
    for no in range(10):  # 10개의 이미지를 입력 받음

        gray = cv2.imread(str(no) + ".png", 0)
        gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        (thresh, gray) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        rows, cols = gray.shape

        cv2.imwrite("b_" + str(no) + ".png", gray)

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

        shiftx, shifty = getBestShift(gray)
        shifted = shift(gray, shiftx, shifty)
        gray = shifted

        cv2.imwrite("image_" + str(no) + ".png", gray)

        flatten = gray.flatten() / 255.0
        images[i] = flatten

        i += 1

    print(sess.run(tf.argmax(prediction, 1), feed_dict={X: images}))