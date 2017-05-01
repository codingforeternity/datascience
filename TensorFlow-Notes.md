### [A Practical Guide for Debugging Tensorflow](https://wookayin.github.io/tensorflow-talk-debugging) (4/27/17)
* Tensor Fetching: The Bad (ii)
  * In fact, we can just perform an additional session.run() for debugging purposes, if it does not involve any side effect
    ```python
    # for debugging only, get the intermediate layer outputs.
    [fc7, prob] = session.run([net['fc7'], net['prob']],
                               feed_dict={images: batch_image})
    # Yet another feed-forward: 'fc7' are computed once more ...
    [loss_value, _] = session.run([loss_op, train_op],
                               feed_dict={images: batch_image})
    ```
  * A workaround: Use **`session.partial_run()`** (undocumented, and still experimental)
    ```python
    h = sess.partial_run_setup([net['fc7'], loss_op, train_op], [images])
    [loss_value, _] = sess.partial_run(h, [loss_op, train_op],
                                       feed_dict={images: batch_image})
    fc7 = sess.partial_run(h, net['fc7'])
    ```
* Interpose any python code in the computation graph
  * We can also embed and interpose a python function in the graph: tf.py_func() comes to the rescue!
    ```python
    tf.py_func(func, inp, Tout, stateful=True, name=None)
    ```
  * Wraps a python function and uses it as a tensorflow op.
  * Given a python function func, which takes numpy arrays as its inputs and returns numpy arrays as its outputs, the function is wrapped as an operation.
    ```python
    def my_func(x):
        # x will be a numpy array with the contents of the placeholder below
        return np.sinh(x)
    inp = tf.placeholder(tf.float32, [...])
    y = py_func(my_func, [inp], [tf.float32])
    ```
* Debugging: Summary
  * Session.run(): Explicitly fetch, and print
  * Tensorboard: Histogram and Image Summary
  * tf.Print(), tf.Assert() operation
  * Use python debugger (ipdb, pudb)
  * Interpose your debugging python code in the graph
  * The (official) TensorFlow debugger: tfdbg
* Name your tensors properly
  * The style that I much prefer:
    ```python
    def multilayer_perceptron(x):
        with tf.variable_scope('fc1'):
            W_fc1 = tf.get_variable('weights', [784, 256])  # fc1/weights
            b_fc1 = tf.get_variable('bias', [256])          # fc1/bias
            fc1 = tf.nn.xw_plus_b(x, W_fc1, b_fc1)          # fc1/xw_plus_b
            fc1 = tf.nn.relu(fc1)                           # fc1/relu
    ```
  * or use high-level APIs or your custom functions:
    ```python
    import tensorflow.contrib.layers as layers
    def multilayer_perceptron(x):
        fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu,
                                     scope='fc1')
    ```
* [Other Topics: Performance and Profiling](https://wookayin.github.io/tensorflow-talk-debugging/#82)
  * Run-time performance is a very important topic!  There will be another lecture soon.  Beyond the scope of this talk...
  * Make sure that your GPU utilization is always non-zero (and, near 100%)
    * Watch and monitor using `nvidia-smi` or `gpustat`
  * Use `nvprof` for profiling CUDA operations
  * Use CUPTI (CUDA Profiling Tools Interface) tools for TF


### https://www.tensorflow.org/extras/candidate_sampling.pdf