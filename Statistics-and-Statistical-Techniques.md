#### [My Journey From Frequentist to Bayesian Statistics ](http://www.fharrell.com/2017/02/my-journey-from-frequentist-to-bayesian.html) (2/21/17)
* "I realized that frequentist multiplicity problems came from the chances you give data to be more extreme, not from the chances you give assertions to be true."

#### [Where priors come from](http://www.zinkov.com/posts/2015-06-09-where-priors-come-from/)
* Overview of various distributions, e.g. Bernoulli, Gamma
* "we can also do exp(x) and x2 to get distributions over positive numbers. These are the Lognormal and χ2 distributions."
* Poisson distribution "can be thought of as a discrete analog to the gamma distribution."
* "Student-T or t can be interpreted as the distribution over a subsampled population from the normal distribution. What’s going on here is that because our sample size is so small, atypical values can occur more often than they do in the general population"

#### [Bayesian Machine Learning](http://fastml.com/bayesian-machine-learning/) (12/21/16)
* Good explanation of the components of the Bayes rule, priors and posteriors and such.
* "Compare with classifiers, which usually model `P( y | x )` to discriminate between classes based on x. A generative model is concerned with joint distribution of y and x, `P( y, x )`. It’s more difficult to estimate that distribution, but it allows sampling and of course one can get `P( y | x )` from `P( y, x )`."
* "A relatively popular application of Gaussian Processes is hyperparameter optimization for machine learning algorithms."
* "The most conspicuous piece of Bayesian software these days is probably [Stan](http://mc-stan.org/). Stan is a probabilistic programming language, meaning that it allows you to specify and train whatever Bayesian models you want."
  * "automatic differentiation variational inference (ADVI). The user only provides a Bayesian model and a dataset; nothing else."
  * "In Python, the most popular package is [PyMC](https://pymc-devs.github.io/pymc3/). It is not as advanced or polished (the developers seem to be playing catch-up with Stan), but still good. PyMC has NUTS and ADVI - here’s a notebook with a [minibatch ADVI example](https://gist.github.com/taku-y/b4da34be310718a6ea02). The software uses Theano as a backend, so it’s faster than pure Python."
* "MCMC is good not just for computing probabilities but for integrating any function. Without it, scientists were limited to functions they could integrate analytically" [http://fastml.com/an-excerpt-from-the-master-algorithm/]
* "The crux of the model selection question is this: how we can quantify the difference between these models and decide which model better describes our data?" [https://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/]
  * "you can view the model selection question as fundamentally about comparing models while correcting for over-fitting of more complicated models. Let's see how this is done within the frequentist and Bayesian approaches."
  * "Frequentists consider probabilities as frequencies: that is, a probability is only meaningful in the context of repeated experiments"
  * "Bayesians consider probabilities as degrees-of-belief: that is, a probability is a way of quantifying our certainty about a particular statement: *model parameters* are uncertain quantities and thus subject to probabilistic description"
    * Frequentists look at the likelihood: `P(D | θ,M)`
    * Bayesians look at the posterior: `P(θ | D,M)`
  * Thus the sum of squares of normalized residuals about the best-fit model should follow the χ2 distribution, and so we construct our χ2 statistic: χ2(D;θ,M) = ∑n[yn−yM(xn;θ)σy,n]2
  * generally, this post is a nice IPython notebook with some nice plots (including shaded distributions)
  * Other Frequentist Approaches: [Cross-Validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29)

#### [Is the logit function always the best for regression modeling of binary data?](http://stats.stackexchange.com/questions/48072/is-the-logit-function-always-the-best-for-regression-modeling-of-binary-data)
* "Andrew Gelman [says](http://arxiv.org/pdf/0901.4011.pdf) (in a mostly unrelated context) that t7 is roughly like the logistic curve. Lowering the degrees of freedom gives you fatter tails and a broader range of intermediate values in your regression. When the degrees of freedom go to infinity, you're back to the probit model."

#### [Measuring The Memory Of Time Series Data](https://prateekvjoshi.com/2016/11/30/measuring-the-memory-of-time-series-data/)
* "Hurst exponent is basically a measure of long term memory of a given time series variable. It is also referred to as the index of dependence. It tells us how strongly the given time series data will regress to the mean."

#### [Bayesian statistics: What’s it all about?](http://andrewgelman.com/2016/12/13/bayesian-statistics-whats/) (12/15/16)
* "Bayesian statistics uses the mathematical rules of probability to combine data with prior information to yield inferences which (if the model being used is correct) are more precise than would be obtained by either source of information alone."
* "In contrast, classical statistical methods avoid prior distributions. In classical statistics, you might include in your model a predictor (for example), or you might exclude it, or you might pool it as part of some larger set of predictors in order to get a more stable estimate."
* "Except in simple problems, Bayesian inference requires difficult mathematical calculations--high-dimensional integrals--which are often most practically computed using stochastic simulation, that is, computation using random numbers. This is the so-called Monte Carlo method, which was developed systematically by the mathematician Stanislaw Ulam and others when trying out designs for the hydrogen bomb"
* "In classical statistics, improvements in methods often seem distressingly indirect: you try a new test that’s supposed to capture some subtle aspect of your data, or you restrict your parameters or smooth your weights, in some attempt to balance bias and variance. Under a Bayesian approach, all the tuning parameters are supposed to be interpretable in real-world terms, which implies--or should imply--hat improvements in a Bayesian model come from, or supply, *improvements in understanding of the underlying problem under studied*."

### * [Streaming Model for Linear Regression](http://koaning.io/bayesian-propto-streaming-algorithms.html) (12/1/16)
* **iterative, on-line/streaming approach to linear regression** (convergence of a posterior distribution--"the conditional probability distribution of the unobserved quantities of ultimate interest, given the observed data" [see [[Machine Learning Notes]]]) (kws: prior)
* nice brief explanation of Bayesian Statistics
* and really nice examples of Python plotting and likihood
* in Python and with reference to Apache Flink and Spark
* "if you've solved streaming then you've also solved batch"
* "Batch is a subset of streaming and thinking like a bayesian helps when designing models."

#### [Scaling and Normalization with Naive Bayes in Python](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html) (11/10/16)

#### [Introductory Statistics for Data Science](http://davegiles.blogspot.com/2015/04/introductory-statistics-for-data-science.html) (4/26/15)

### * [Handy Statistical Lexicon, Gelman](http://andrewgelman.com/2009/05/24/handy_statistic/) (10/3/16)

#### [A Curated List of Data Science Interview Questions](https://www.springboard.com/blog/data-science-interview-questions/) (11/8/16)

#### [When Confounding Variables Are Out Of Control](http://slatestarcodex.com/2016/04/15/links-416-they-cant-link-our-dick/) (5/4/16)
* http://blogs.discovermagazine.com/neuroskeptic/2016/04/02/confounding-variables/#.VynS9pStxuB
* [Statistically Controlling for Confounding Constructs Is Harder than You Think](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152719)
* "if you want to control for a certain variable, let’s call it C, your ability to succesfully correct for it is limited by the reliability of your measure of C. Let’s call your experimental measure (or construct) Cm. If you find that a certain correlation holds true even controling for Cm, this might be because C is not really a confound, but it might also be that Cm is a poor measure of C. “Controlling for Cm” is not the same as “Controlling for C”."

#### [Investing using Python: Kolmogorov-Smirnov test as regime switcher](http://www.talaikis.com/kolmogorov-smirnov-test-as-regime-switcher/) (4/13/16)
* kws: regime switching
* fit line to time series, residualize, and max residual indicates regime switch