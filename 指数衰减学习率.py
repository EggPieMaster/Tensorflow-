import tensorflow as tf

LEARNING_RATE_BASE = 0.1  # 学习率初值
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 2  # 学利率更新频率

global_step = tf.Variable(0, trainable=False)  # 计数器，训练了轮数，不是训练参数，设置为不可被训练
# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=False)
# 定义损失函数
w = tf.Variable(5, dtype=tf.float32)
loss = tf.square(w + 1)
# 定义反向传播算法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(100):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('训练轮数是：%f，学习率是： %f, w值是： %f, 损失函数值是： %f' % (global_step_val, learning_rate_val, w_val, loss_val))
