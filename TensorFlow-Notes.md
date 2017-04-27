### [](https://wookayin.github.io/tensorflow-talk-debugging)
* Tensor Fetching: The Bad (ii)
  * In fact, we can just perform an additional session.run() for debugging purposes, if it does not involve any side effect

```python
    # for debugging only, get the intermediate layer outputs.
    [fc7, prob] = session.run([net['fc7'], net['prob']],
                               feed_dict={images: batch_image})
    #
    # Yet another feed-forward: 'fc7' are computed once more ...
    [loss_value, _] = session.run([loss_op, train_op],
                               feed_dict={images: batch_image})
```
  * A workaround: Use session.partial_run() (undocumented, and still experimental)
```python
    h = sess.partial_run_setup([net['fc7'], loss_op, train_op], [images])
    [loss_value, _] = sess.partial_run(h, [loss_op, train_op],
                                       feed_dict={images: batch_image})
    fc7 = sess.partial_run(h, net['fc7'])
```


### https://www.tensorflow.org/extras/candidate_sampling.pdf