# coding=gbk
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report



# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path)
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path)
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

class Config(object):
    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series?
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Trainging
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 1500
        self.clip_gradients = 10
        self.gradient_noise_scale = None



        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 128  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        # self.W = {
        #     'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),#输入
        #     'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))#输出
        # }
        # self.biases = {
        #     'hidden': tf.Variable(tf.random_normal([self.n_hidden])),#输入
        #     'output': tf.Variable(tf.random_normal([self.n_classes]))#输出
        # }
        self.W = {
            'hidden': tf.get_variable('w1_xaiver',[self.n_inputs, self.n_hidden],initializer=tf.contrib.layers.xavier_initializer()),   #输入
            'output': tf.get_variable('w2_xaiver',[self.n_hidden, self.n_classes],initializer=tf.contrib.layers.xavier_initializer())   #输出
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),#输入
            'output': tf.Variable(tf.random_normal([self.n_classes]))#输出
        }




#LSTM核心算法
def LSTM_Network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray feature matrix, shape=[batch_size,time_steps,n_inputs] 现在是(1500,128,9)
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """

    """
    Version 2,need to change test batch_size to 1500
    """
    # # transpose the inputs shape from
    # # X ==> (1500 batch * 128 steps, 9 inputs)
    # feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # # into hidden
    # # X_in = (128 batch * 28 steps, 128 hidden)
    # feature_mat = tf.matmul(feature_mat,config.W['hidden']) + config.biases['hidden']
    # # X_in ==> (128 batch, 28 steps, 128 hidden)
    # feature_mat = tf.reshape(feature_mat, [-1, config.n_steps, config.n_hidden])
    # # basic LSTM Cell.
    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    # # lstm cell is divided into two parts (c_state, h_state)
    # init_state = lstm_cell.zero_state(config.batch_size, dtype=tf.float32) # 初始化全零 state
    # # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # # Make sure the time_major is changed accordingly.
    # outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, feature_mat, initial_state=init_state, time_major=False)
    # # hidden layer for output as the final results
    # outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    # results = tf.matmul(outputs[-1], config.W['output']) + config.biases['output']
    # return  results


    """
    Version 1
    """
    #下面是根据莫烦的老版本RNN修改的代码
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    feature_mat = tf.split(0, config.n_steps, feature_mat)#此处tf.split代码有变化，请注意
    # Define a lstm cell with tensorflow
    # lstm_cell =  tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(config.n_hidden)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.8)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, feature_mat, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], config.W['output']) + config.biases['output']



    """
    Version 0
    """
    # #下面是原始版本的代码
    # # Exchange dim 1 and dim 0
    # feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    # # New feature_mat's shape: [time_steps, batch_size, n_inputs]
    #
    # # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # # feature_mat ==> (1500 batch_size * 128 steps, 9 inputs)
    # # Temporarily crush the feature_mat's dimensions
    # feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # # New feature_mat's shape: [time_steps*batch_size, n_inputs]
    #
    # # 接下来feature_mat = W*X + b
    # # Linear activation, reshaping inputs to the LSTM's number of hidden:
    # feature_mat = tf.matmul(
    #     feature_mat, config.W['hidden']
    # ) + config.biases['hidden']
    # # New feature_mat's shape: [time_steps*batch_size, n_hidden]
    #
    # # Split the series because the rnn cell needs time_steps features, each of shape:
    # feature_mat = tf.split(0, config.n_steps, feature_mat)
    # # New feature_mat's shape: a list of length "time_step" containing tensors of shape [batch_size, n_hidden]
    #
    # # Define LSTM cell of first hidden layer:
    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    #
    # # Stack two LSTM layers, both layers has the same shape
    # lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    #
    # # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
    # outputs, _ = tf.nn.rnn(lsmt_layers, feature_mat, dtype=tf.float32)
    # # outputs' shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_classes]
    #
    # # Linear activation
    # # Get the last output tensor of the inner loop output series, of shape [batch_size, n_classes]
    # return tf.matmul(outputs[-1], config.W['output']) + config.biases['output']


#将输入序列one_hot化
def one_hot(label):
    label_num = len(label)
    new_label = label.reshape(label_num)  # shape : [sample_num]
    # because max is 5, and we will create 6 columns
    n_values = np.max(new_label) + 1
    return np.eye(n_values)[np.array(new_label, dtype=np.int32)]

if __name__ == "__main__":

    #-----------------------------
    # step1: 加载数据
    #-----------------------------
    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [#加载的数据都是原始数据，raw data
                          "body_acc_x_",
                          "body_acc_y_",
                          "body_acc_z_",
                          "body_gyro_x_",
                          "body_gyro_y_",
                          "body_gyro_z_",
                          "total_acc_x_",
                          "total_acc_y_",
                          "total_acc_z_"
                          ]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    DATA_PATH = "data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    TRAIN = "train/"
    TEST = "test/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
        ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
        ]
    X_train = load_X(X_train_signals_paths)#load raw train data
    print("load X_train shape is:",X_train.shape)#(7352, 128, 9)
    X_test = load_X(X_test_signals_paths)#load raw test data
    print("load X_test shape is:",X_test.shape)#(2947, 128, 9)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"#路径为data/UCI HAR Dataset/train/y_train.txt
    y_test_path = DATASET_PATH + TEST + "y_test.txt"#路径为data/UCI HAR Dataset/train/y_test.txt
    y_train = one_hot(load_y(y_train_path)) #to the one-hot matrix
    print("y_train one-hot shape is:", y_train.shape)#(7352, 6)
    y_test = one_hot(load_y(y_test_path))
    print("y_test one-hot shape is:", y_test.shape)#(2947, 6)

    #-----------------------------------
    # step2: 定义模型参数
    #-----------------------------------
    config =Config(X_train, X_test)#此处配置了几个config

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    print(X_train[0].shape)#(128,9)
    print(X_train[0][0].shape)#(9,)
    print("\n========================================================================\n")

    #------------------------------------------------------
    # step3: 建立神经网络
    #------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])#(None,128,9)
    Y = tf.placeholder(tf.float32, [None, config.n_classes])#(None,6)

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation,计算L2参数值
    L2 = config.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Soca.neusoftmax loss and L2加入L2惩罚项
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + L2



    trainvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(Loss, trainvars),10)   # We clip the gradients to prevent explosion
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(Loss)  # 原始代码

    # global_step = tf.Variable(0)
    # learning_rate =tf.train.exponential_decay(config.learning_rate,global_step,300,0.96,staircase= True)
    # optimizer =tf.train.GradientDescentOptimizer(learning_rate).minimize(Loss,global_step=global_step)



    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned of for less console spam.
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []


    sess = tf.InteractiveSession(config = tf.ConfigProto(log_device_placement = False))  # Ture改为false可以取消在终端显示输出
    tf.global_variables_initializer().run()

    best_accuracy = 0.0
    # Start training for each batch and loop epochs

    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size), range(config.batch_size, config.train_count + 1, config.batch_size)):
            _, pred_temp, lost_temp=sess.run([optimizer, accuracy, Loss], feed_dict={X: X_train[start:end],Y: y_train[start:end]})

            train_accuracies.append(pred_temp)
            train_losses.append(lost_temp)

            # Test completely at every epoch: calculate accuracy
            pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, Loss], feed_dict={X: X_test, Y: y_test})

            test_accuracies.append(accuracy_out)
            test_losses.append(loss_out)

        print("traing iter: {},".format(i)+" test accuracy : {},".format(accuracy_out)+" loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    predictions = pred_out.argmax(1)
    y_test = load_y(y_test_path)
    print(classification_report(y_test, predictions))


