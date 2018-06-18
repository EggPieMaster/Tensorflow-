import tensorflow as tf
import numpy as np

BATCH_SIZE = 8  # 一次喂入神经网络8组数据

# 基于seed长生随机数
rng = np.random.RandomState()
# 随机数产生32行2列的矩阵， 表示32组数据，一组数据两个数值
X = rng.rand(32, 2)
# 从X这32行2列矩阵中取出一行，判断如果两个数据之和小于1则给Y复制为1，如果两个数据之和不小于1给Y赋值为0，   Y作为输入数据集的标签（正确答案）
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print('X:\n', X)
print('Y:\n', Y)

# 定义神经网络输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y - y_))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  # 梯度下降算法
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)  # 自适应

# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前未经训练的参数值
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
    print('\n')
    # 训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:  # 每500次输出一次loss值
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After %d training step(s),loss on all data is %g' % (i, total_loss))

    # 输出训练后的参数数值
    print('\n')
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
