#### [A small introduction to boosting](https://codesachin.wordpress.com/2016/03/06/a-small-introduction-to-boosting/)
* **Using complex methods such as SVMs in Boosting usually leads to [overfitting](https://en.wikipedia.org/wiki/Overfitting).**
* To improve the bias/variance characteristics of Boosting, bagging is generally employed. What this essentially does is train each weak learner on a subset of the overall dataset (rather than the whole training set). This causes a good decrease in variance and tends to **reduce overfitting**.
* The biggest criticism for Boosting comes from its sensitivity to noisy data. Think of it this way. At each iteration, Boosting tries to improve the output for data points that haven't been predicted well 'til now. If your dataset happens to have some (or a lot) of misclassified/outlier points, the meta-algorithm will try very hard to fit subsequent weak learners to these noisy samples. As a result, **overfitting is likely to occur**. The exponential loss function used in AdaBoost is particularly vulnerable to this issue (since the error from an outlier will get an exponentially-weighed importance for future learners).
* Implementation in Scikit-Learn
  * [AdaBoost implementation](http://scikit-learn.org/stable/modules/ensemble.html#adaboost)
  * [Gradient Tree Boosting implementation](http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)
* **FWC - Reweighting based on error (e.g. PnL) will lead to fitting subsequent models to large positions with large returns.  But reweighting based on only large positions multiplied by binary, 1 or -1, for correct or incorrect, will remove the sensitivity to large returns.**

#### [AdaBoost with neural networks](http://stackoverflow.com/questions/35691636/adaboost-with-neural-networks) (12/23/16)
  1. Train your first weak classifier by using the training data
  2. The 1st trained classifier makes mistake on some samples and correctly classifier others. Increase the weight of those samples that are wrongly been classified and decrease the weight of others. Retrain your classifier with these weights to get your 2nd classifier. In your case, you first have to resample with replacement from your data with these updated weights, create a new training data and then train your classifier over these new data.
  3. Repeat the 2nd step T times and at the end of each round, calculate the alpha weight for the classifier according to the formula. 4- The final classifier is the weighted sum of the decisions of the T classifiers.
* PS: There is no guarantee that boosting increases the accuracy. In fact, so far **all the boosting methods that I'm aware of were unsuccessful to improve the accuracy with NN as weak learners** (The reason is because of the way that boosting works and needs a lengthier discussion).

#### [AdaBoost on Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
* AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers
* AdaBoost (with [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning) as the weak learners) is often referred to as the best out-of-the-box classifier.[1](https://en.wikipedia.org/wiki/AdaBoost#cite_note-1)[2](https://en.wikipedia.org/wiki/AdaBoost#cite_note-2)