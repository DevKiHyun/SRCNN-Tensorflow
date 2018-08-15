import tensorflow as tf

class SRCNN:
    def __init__(self, config):

        self.n_channel = config.n_channel
        self.X = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channel])
        self.Y = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channel])
        self.weights = {}
        self.biases = {}
        self.global_step = tf.Variable(0, trainable=False)
        self.psnr = tf.Variable(0, dtype=tf.float32)

    def _conv2d_layer(self, inputs, filters_size, strides=[1,1], add_bias=False, name=None,
                      padding="SAME", activation=None, trainable=True):
        filters = self._get_conv_filters(filters_size, name, trainable)
        strides = [1, *strides, 1]

        conv_layer = tf.nn.conv2d(inputs, filters, strides=strides, padding=padding, name=name + "_layer")

        if add_bias != False:
            conv_layer = tf.add(conv_layer, self._get_bias(filters_size[-1], name))
        if activation != None:
            conv_layer = activation(conv_layer)

        return conv_layer

    def _get_conv_filters(self, filters_size, name, trainable=True):
        name = name+"_weights"
        initializer = tf.contrib.layers.xavier_initializer()
        conv_weights = tf.Variable(initializer(filters_size), name=name, trainable=trainable)
        self.weights[name] = conv_weights

        return conv_weights

    def _get_bias(self, bias_size, name, trainable=True):
        name = name+"_bias"
        bias = tf.Variable(tf.zeros([bias_size]), name=name)
        self.biases[name] = bias

        return bias

    def neuralnet(self):
        self.conv_1 = self._conv2d_layer(self.X, filters_size=[9, 9, self.n_channel, 128], add_bias=True, name="patch_extract",
                              padding="SAME",
                              activation=tf.nn.relu)
        self.conv_2 = self._conv2d_layer(self.conv_1, filters_size=[5, 5, 128, 64], add_bias=True, name="non_linear_map",
                              padding="SAME",
                              activation=tf.nn.relu)
        self.conv_3 = self._conv2d_layer(self.conv_2, filters_size=[5, 5, 64, self.n_channel], add_bias=True, name="reconstruct",
                              padding="SAME")
        self.output = tf.clip_by_value(self.conv_3, clip_value_min=0, clip_value_max=1)


    def optimize(self, config):
        self.cost = tf.reduce_mean(tf.pow(self.Y - self.output, 2))

        self.learning_rate = config.learning_rate
        opt1 = tf.train.AdamOptimizer(self.learning_rate)
        opt2 = tf.train.AdamOptimizer(self.learning_rate * 0.1)

        grads = opt1.compute_gradients(self.cost)

        opt1_grads = [variable for variable in grads if 'reconstruct' not in variable[1].name]
        opt2_grads = [variable for variable in grads if 'reconstruct' in variable[1].name]

        self.optimizer = tf.group(opt1.apply_gradients(opt1_grads, global_step=self.global_step),
                                  opt2.apply_gradients(opt2_grads))
        '''
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                                           beta1=config.beta_1,
                                           beta2=config.beta_2,
                                           epsilon=config.epsilon).minimize(self.cost)
        '''

    def summary(self):
        '''
        for weight in list(self.weights.keys()):
            tf.summary.histogram(weight, self.weights[weight])
        for bias in list(self.biases.keys()):
            tf.summary.histogram(bias, self.biases[bias])
        '''

        tf.summary.scalar('Loss', self.cost)
        tf.summary.scalar('Average test psnr', self.psnr)
        tf.summary.scalar('Learning rate', self.learning_rate)

        self.summaries = tf.summary.merge_all()
