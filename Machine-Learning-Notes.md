[Faster convergence than SGD: L-BFGS and neural nets](http://www.reddit.com/r/MachineLearning/comments/4bys6n/lbfgs_and_neural_nets/) (3/25/16)

[The Ultimate List of Machine Learning and Data Mining Books](http://www.reddit.com/r/MachineLearning/comments/4e79sp/the_ultimate_list_of_machine_learning_and_data/) (4/12/16)

[New technique for recommender systems](https://docs.google.com/presentation/d/19QDuPmxB9RzQWKXp_t3yqxCvMBSMaOQk19KNZqUUgYQ/edit#slide=id.g11a4ba0c5c_0_922) (4/5/16)

[Machine "Un"-learning](http://m.phys.org/news/2016-03-machine-unlearning-technique-unwanted-quickly.html) (3/18/16)
* Aggregate data at varying levels of granularity.  Then machine learn from a chosen level to speed things up.

[Two Minute Papers - How DeepMind Conquered Go With Deep Learning (AlphaGo)](http://www.reddit.com/r/MachineLearning/comments/4a6udw/two_minute_papers_how_deepmind_conquered_go_with/) (3/16/16)

TensorFlow
* [Deep-Q learning Pong with Tensorflow and PyGame](http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html) (3/16/16)
* [Deep Learning Comparison Sheet: Deeplearning4j vs. Torch vs. Theano vs. Caffe vs. TensorFlow](http://deeplearning4j.org/compare-dl4j-torch7-pylearn.html) (3/16/16)
  * "Thirdly, Java’s lack of robust scientific computing libraries can be solved by writing them, which we’ve done with ND4J."
  * "By choosing Java, we excluded the fewest major programming communities possible."
  * "We have paid special attention to Scala in building Deeplearning4j and ND4J, because we believe Scala has the potential to become the dominant language in data science."
* [How to combine CNN with CRF in python?](http://www.reddit.com/r/MachineLearning/comments/4aadvq/how_to_combine_cnn_with_crf_in_python/) (3/16/16)
* [TensorFlow Machine Learning with Financial Data on Google Cloud Platform](http://www.reddit.com/r/MachineLearning/comments/49foem/tensorflow_machine_learning_with_financial_data/) (3/10/16)
* [TensorFlow in Python](http://blog.pythonanywhere.com/126/) (3/10/16)
* Python Anywhere: Quickstart: TensorFlow-Examples on PythonAnywhere
* [Do you think TensorFlow will eventually replace Theano and Torch?](http://www.reddit.com/r/MachineLearning/comments/4amr7p/askreddit_do_you_think_tensorflow_will_eventually/) (3/18/16)

[The Promise of Artificial Intelligence Unfolds in Small Steps](http://rss.nytimes.com/c/34625/f/640377/s/4de9b663/sc/13/l/0L0Snytimes0N0C20A160C0A20C290Ctechnology0Cthe0Epromise0Eof0Eartificial0Eintelligence0Eunfolds0Ein0Esmall0Esteps0Bhtml0Dpartner0Frss0Gemc0Frss/story01.htm)
* R vs. Python

[Some Tips for Debugging in Deep Learning](http://www.lab41.org/some-tips-for-debugging-in-deep-learning-2/) (2/8/16)
* TensorFlow, Theano, Keras, Neon, PDB, TDB, TensorBoard, NVIS

[The Bias-Variance and Precision-Recall tradeoffs (simply) explained](https://cynml.wordpress.com/2016/01/18/the-bias-variance-and-precision-recall-tradeoffs-explained/) (2/9/16)

Benefit of KNN: You can model the whole distribution, all the moments of it.  This is difficult to model with other techniques.  Ask Desmond, he might know others.

[Neural GPUs Learn Algorithms](http://www.wikitract.com/neural-gpus-learn-algorithms-google/) (12/10/15)

[Self Organizing Maps (SOMs) Tutorial](http://www.ai-junkie.com/ann/som/som1.html) (12/7/15)
* Linked to from [SOMs with Google's TensorFlow](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/)

Machine Learning packages (12/4/15)
* Prototyping - R, Python
* IBM - Weka, SPSS (Desmond isn't quite sure why IBM uses these though)
* Lua/Torch - like Python, 2 GB csv limit that might be fixed by now though
* Big Data - Spark/mllib and Hadoop/Mahout, big bug less configurable, e.g. fewer metrics and less options for targeted batch GD (but this isn't really that important in practi

[Weka 3: Data Mining Software in Java](http://www.cs.waikato.ac.nz/ml/weka/)
* This software was mentioned by Desmond.
* He sounded reluctant when he had to stop using it in favor of SPSS.

[Smart Reply - Google's attempt at a deep NN that replies to an email](http://googleresearch.blogspot.com/2015/11/computer-respond-to-this-email.html) (11/3/15)

[Dissecting Bias vs. Variance Tradeoff In Machine Learning](http://prateekvjoshi.com/2015/10/20/dissecting-bias-vs-variance-tradeoff-in-machine-learning) (10/29/15)

[Visualizing popular ML algorithms](http://jsfiddle.net/wybiral/3bdkp5c0/embedded/result/) (10/25/15)

[What a Deep Neural Network thinks about your #selfie](http://karpathy.github.io/2015/10/25/selfie/) (10/26/15)

Email: Nns are only being applied to trivial problems (9/25/15)
* e.g. Image recognition, NLP
* This is because lots of people understand the goals of those problems.
* I'm in a unique position to understand the goals of financial forecasting.
* Thus suggests to get good at doing a problem the hard way first to learn how it's done, and then switch to ML
* It's almost like there's another dimension to the supervised/unsupervised dimension of ML: understanding the output/y's/range/meaning vs. not understanding--maybe this is "domain knowledge"

Email: "projections onto nonlinear subspaces (k-RBMs)" (the email contains hard copies of these papers)
* "Non-Linear Manifolds" - https://www.google.com/search?q=projection+onto+curved+subspaces&ie=utf-8&oe=utf-8#q=projection+onto+nonlinear+subspaces
* RBMs - http://deeplearning.net/tutorial/rbm.html

Email: "restricted boltzman machines, introduction"
* http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/

Paper: [A Convolutional Neural Net for Modeling Sentences](http://nal.co/papers/Kalchbrenner_DCNN_ACL14)

[The Man Who Would Teach Machines to Think](http://www.theatlantic.com/magazine/archive/2013/11/the-man-who-would-teach-machines-to-think/309529/)
* Why conquer a task if there’s no insight to be had from the victory? “Okay,” he says, “Deep Blue plays very good chess—so what? Does that tell you something about how we play chess? No. Does it tell you about how Kasparov envisions, understands a chessboard?” A brand of AI that didn’t try to answer such questions—however impressive it might have been—was, in Hofstadter’s mind, a diversion.
* “Nobody is a very reliable guide concerning activities in their mind that are, by definition, subconscious,” he once wrote. “This is what makes vast collections of errors so important. In an isolated error, the mechanisms involved yield only slight traces of themselves; however, in a large collection, vast numbers of such slight traces exist, collectively adding up to strong evidence for (and against) particular mechanisms.” Correct speech isn’t very interesting; it’s like a well-executed magic trick—effective because it obscures how it works.
* FWC - "vast collections of errors so important" - similar to studying differentials rather than levels, as in differential gene expressions--i.e. make your own errors!

[Deep Learning and the New Bayesian Golden Age](http://www.theplatform.net/2015/09/24/deep-learning-and-a-new-bayesian-golden-age/)
* Carpenter is putting his nearly one million dollars in National Science Foundation grants into work around the open source “Stan” package, which presents a scalable approach to Bayesian modeling that can incorporate larger, more complex problems than tend to fit inside other frameworks.
* working with drug maker Novartis on a drug-disease interaction clinical trial for an ophthalmology drug that spans close to 2,000 patients.

Deep Learning Book
* http://www.iro.umontreal.ca/~bengioy/dlbook/ (no pdf available, only online for now)
* Bibliography: http://www.iro.umontreal.ca/~bengioy/dlbook/front_matter.pdf (also in Downloads directory)

Deep Learning Resources (http://deeplearning.net/software_links/)
* "[Torch](http://torch.ch/) – provides a Matlab-like environment for state-of-the-art machine learning algorithms in lua"

Online Courses (YouTube)
* MIT OpenCourseWare
* Deep Learning at Oxford, 2015

Email from Bart: Fwd: Yann at Firstmark
* "Although quite long (~45 min), it's a good snapshot of where AI / deep learning is today and where it came from and why it's winning."
* [Yann Lecun, Facebook // Artificial Intelligence // Data Driven #32 (Hosted by FirstMark Capital)] (https://www.youtube.com/watch?v=AbjVdBKfkO0)
* My notes [[here|Yann Lecun, Facebook, Firstmark talk]]

Email: "Scary machine learning"
* Why You Should be a Little Scared of Machine Learning http://www.reddit.com/r/programming/comments/3l0qek/why_you_should_be_a_little_scared_of_machine/
* This link on Recurrent Neural Networks was linked to from the above: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
  * "RNNs combine the input vector with their state vector with a fixed (but learned) function to produce a new state vector"
  * "If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs."
  * "crucially this output vector's contents are influenced not only by the input you just fed in, but also on the entire history of inputs you've fed in in the past."  I.e. this links back to learning in stages (pixels, lines, shapes, faces; or regimes, sectors, instruments).
  * SEQUENCES - "model the probability distribution of the next character in the sequence given a sequence of previous characters"
  * this article contains lots of other links: [RMSProp](http://arxiv.org/abs/1502.04390), [minimal character-level RNN language model in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086), [code](https://github.com/karpathy/char-rnn), [Long-Short-Term Memory networks](https://en.wikipedia.org/wiki/Long_short_term_memory)

Email: "What is the difference between convolutional neural networks, restricted Boltzmann machines, and auto-encoders? - Cross Validated"
* http://stats.stackexchange.com/questions/114385/what-is-the-difference-between-convolutional-neural-networks-restricted-boltzma
* "So, if we already had PCA, why the hell did we come up with autoencoders and RBMs? It turns out that PCA only allows linear transformation of a data vectors."

Email: "Restricted Boltzmann machines vs multilayer neural networks - Coursera"
* http://stats.stackexchange.com/questions/40598/restricted-boltzmann-machines-vs-multilayer-neural-networks
* "Once deep (or maybe not that deep) network is pretrained, input vectors are transformed to a better representation and resulting vectors are finally passed to real classifier (such as SVM or logistic regression). In an image above it means that at the very bottom there's one more component that actually does classification."

Coursera Machine Learning Course Notes
* file:///home/fred/Documents/coursera/machine_learning/complete_notes_holehouse/Machine_learning_complete/09_Neural_Networks_Learning.html (from here: http://www.holehouse.org/mlclass/)
* http://cs229.stanford.edu/materials.html
* Email: "octave/matlab resources"
** https://www.coursera.org/learn/machine-learning/supplement/Mlf3e/more-octave-matlab-resources
* http://www.forbes.com/sites/anthonykosner/2013/12/29/why-is-machine-learning-cs-229-the-most-popular-course-at-stanford/

Interview with Cloudera CEO Mike Olson (2010)
* http://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning.html
* People still use relational databases - great if you have predictable queries over structured data
* But the data people want to work with is becoming more complex and bigger
  * Free text, unstructured data doesn't fit will into tables
  * Do sentiment analysis in SQL isn't really that good
  * So to do new kinds of processing need a new type of architecture
* Hadoop lets you do data processing - not transactional processing - on the big scale
* Increasingly things like NoSQL is being used

Email: (no subject)
* Welcome to the AI Conspiracy: The ‘Canadian Mafia’ Behind Tech’s Latest Craze http://recode.net/2015/07/15/ai-conspiracy-the-scientists-behind-deep-learning/
* "Deep-Learning AI Is Taking Over Tech. What Is It?"
* "The results of these collections are then tiled so that they overlap to obtain a better representation of the original image; this is repeated for every such layer. Because of this, they are able to tolerate translation of the input image" https://en.wikipedia.org/wiki/Convolutional_neural_network

Email: "Deep learning to understand..."
* ...anything we don't. Anything we can observe effect and input, but not transmission mechanism.
* E.g. the inner workings of the brain. Which would be very meta, b/c it would be like a machine learning a machine that learns a machine.
* E.g. machine learning to learn the structure of a neural net.  A neural net learning a neural net.  What if it could learn what the outputs and inputs are rather than being programmed to know them.

Email: "The Model Complexity Myth"
* The Model Complexity Myth https://jakevdp.github.io/blog/2015/07/06/model-complexity-myth/
* Gaussian Processes for Machine Learning (free technical book) http://www.gaussianprocess.org/gpml/
* Fill in the Blanks: Using Math to Turn Lo-Res Datasets Into Hi-Res Samples http://www.wired.com/2010/02/ff_algorithm/

Benchmarking Nearest Neighbor Searches in Python
* https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/

Email: ML
* Codementor: Building Data Products with Python: Using Machine Learning to Provide Recommendations https://www.codementor.io/python/tutorial/build-data-products-django-machine-learning-clustering-user-preferences
* "using K-means clustering as a machine learning model that makes use of user similarity in order to provide better wine recommendations"

Email: New method of learning
* 3. Chess-playing computer with an entirely new method of learning, and more here.  It is International Master strength (for now).
* http://www.technologyreview.com/view/541276/deep-learning-machine-teaches-itself-chess-in-72-hours-plays-at-international-master/

[A baseline C++ neural network library, with genetic algorithm and backpropagation training](https://github.com/SullyChen/FFANN) [X-Post from r/MachineLearning] http://www.reddit.com/r/programming/comments/3lnna7/a_baseline_c_neural_network_library_with_genetic/ 

https://www.kaggle.com/
