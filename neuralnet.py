import time
import tensorflow as tf


class LayeredNetwork:
    def __init__(self, batch_size, x_width, y_width, hid_layers, learning_rate, global_step=None, snapshot=None,
                 rseed=None):
        self.constants = {"batch_size": batch_size, "learning_rate": learning_rate}
        self.parameters = {}
        self.tensors = {}

        self.global_step = global_step

        self.inputs = tf.placeholder(tf.float32, shape=(None,x_width), name="Inputs")
        self.tensors["x"] = self.inputs
        prevl_width = x_width
        prev_layer = "x"

        if not rseed:
            rseed = time.time()

        hid_layer_num = 0
        for layer_width, activation_func in hid_layers:
            with tf.name_scope("Layer" + str(hid_layer_num)) as scope:
                self.tensors["layer" + str(hid_layer_num)] = self.layer(prevl_width, prev_layer,
                                                                        layer_width, hid_layer_num,
                                                                        activation_func, scope, snapshot, rseed)
            prevl_width = layer_width
            prev_layer = "layer" + str(hid_layer_num)
            hid_layer_num += 1

        with tf.name_scope("OutputLayer") as scope:
            self.outputs = self.output_layer(prevl_width, prev_layer, y_width, scope, snapshot, rseed)

        with tf.name_scope("Training") as scope:
            self.labels = tf.placeholder(tf.int32, shape=(batch_size), name="Labels")
            self.tensors["y_"] = self.labels
            self.train_step = self.training(self.tensors["y_"])

        with tf.name_scope("Accuracy") as scope:
            self.testy = tf.placeholder(tf.int32, [None, ], name="Test_y")
            self.acc = self.accuracy()

    def layer(self, prevl_width, prev_layer, layer_width, layer_num, activation_func, scope, snapshot, rseed):
        if snapshot:
            wl = snapshot["w" + str(layer_num)]
            bl = snapshot["b" + str(layer_num)]
        else:
            wl = tf.random_normal([prevl_width, layer_width], dtype=tf.float32, stddev=1e-1, seed=rseed)
            bl = tf.random_normal([layer_width], dtype=tf.float32, stddev=1e-1, seed=rseed)

        self.parameters["w" + str(layer_num)] = tf.Variable(wl, name="weight")
        self.parameters["b" + str(layer_num)] = tf.Variable(bl, name="bias")
        layer_l = tf.add(tf.matmul(self.tensors[prev_layer], self.parameters["w" + str(layer_num)]),
                         self.parameters["b" + str(layer_num)])
        #layer1 = tf.nn.tanh(layer1l)
        return activation_func(layer_l, name="activation")

    def output_layer(self, prevl_width, prev_layer, y_width, scope, snapshot, rseed):
        if snapshot:
            wl = snapshot["wOutput"]
            bl = snapshot["bOutput"]
        else:
            wl = tf.random_normal([prevl_width, y_width], dtype=tf.float32, stddev=1e-1, seed=rseed)
            bl = tf.random_normal([y_width], dtype=tf.float32, stddev=1e-1, seed=rseed)

        self.parameters["wOutput"] = tf.Variable(wl, name="weight")
        self.parameters["bOutput"] = tf.Variable(bl, name="bias")

        return tf.add(tf.matmul(self.tensors[prev_layer], self.parameters["wOutput"]),
                      self.parameters["bOutput"], name="Outputs")

    def training(self, y_):
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.outputs, tf.to_int64(y_))
        cost = tf.reduce_mean(entropy)
        return tf.train.AdamOptimizer(learning_rate=self.constants["learning_rate"]).minimize(
            cost, global_step=self.global_step)

    def accuracy(self):
        self.tensors["testy"] = self.testy
        probs = tf.nn.softmax(self.outputs)
        correct_prediction = tf.equal(tf.argmax(probs, 1), tf.to_int64(self.tensors["testy"]))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#if __name__ == "__main__":
#    hog1layer(1000, .00001, datafolder, feature="hog2", snapshot=M)
