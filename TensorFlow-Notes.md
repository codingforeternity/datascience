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
  * https://www.tensorflow.org/extras/candidate_sampling.pdf