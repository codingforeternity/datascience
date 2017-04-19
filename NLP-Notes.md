***

[[Machine Learning Notes]]

[[Notes for Geoff Hinton's Coursera ML course]]

***

#### [Using NLP, Machine Learning & Deep Learning Algorithms to Extract Meaning from Text](https://www.infoq.com/presentations/nlp-machine-learning-meaning-text?utm_source=presentations_about_architecture-design&utm_medium=link&utm_campaign=architecture-design)
* From Roman
* At the beginning, there was search
  * Scalable & robust indexing pipeline ("put corpus into a search engine")
  * Tokenizers & analyzers
  * Synonyms, spellers & auto-suggest
  * File formats & header boosting [FWC - weighing headers higher?]
  * Rankers, link, and reputation boosting
* Then there was semantic search (for specific domains)
  * User context: location and time
  * 2 general techniques:
    * Dictionary based attribute extraction
    * Machine learned attribute extraction
* And then you need to understand language
  * positive/negative, speculative, possible, conditional, family/patient history [e.g. hypotheticals]
  * nothing, double negative, compound, speculative ("Patient denies alcohol abuse" -- the only reason for a Dr to write this is if he suspects alcohol abuse -- highly domain specific), lists
* Machine learned annotators
  * "Most of the time you're doing NLU, you're really doing ML--and earlier than you would think"
  * Sometimes easier to just code an annotators biz logic
    * Grammatical patterns ("if ... then ... else ...")
    * Direct inferences (Age <= 18 ==> child)
    * Lookups (RIDT ==> lab test)
  * But sometimes easier to learn it from examples
    * Under diagnosed conditions (e.g. flu, depression)
    * Implied by context (relevant labs normal)
* Bootstrap and then expand your vocabulary
  * Every field has its own jargon.  Even if it's just 60-100 words, those are the most used words, day to day.

#### [Dive into NLTK](http://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize) (3/20/17)
* sentence/word tokenization
  * pre-*trained* tokenization *models* for European languages!
    * "Tokenize Punkt module has many pre-trained tokenize model for many european languages, here is the list from the nltk_data/tokenizers/punkt/README file..."
  * [though they don't appear to recognize non-ASCII chars well]
* part-of-speech tagging
  * "You can find the pre-trained POS Tagging Model in nltk_data/taggers..."
* stemming and lemmatization
  * "it is usually sufficient that related words map to the same stem, even if this stem is not in itself a valid root"
  * "Many search engines treat words with the same stem as synonyms as a kind of query expansion, a process called **conflation**."
  * "*Lemmatisation is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context*, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications."
  * "You would note that the 'are' and 'is' lemmatize results are not 'be', that's because the lemmatize method default part-of-speech (`pos`) argument is 'n': `lemmatize(word, pos='n')`. So you need specified the pos for the word like these: `wordnet_lemmatizer.lemmatize('is', pos='v')`.  **We have use POS Tagging before word lemmatization[!!!]**"
  * StackOverflow: [What are the major differences and benefits of Porter and Lancaster Stemming algorithms?](http://stackoverflow.com/questions/10554052/what-are-the-major-differences-and-benefits-of-porter-and-lancaster-stemming-alg)
    * "Snowball(Porter2): Nearly universally regarded as an improvement over Porter"
    * "Lancaster: Very aggressive stemming algorithm, sometimes to a fault. **With porter and snowball, the stemmed representations are usually fairly intuitive to a reader, not so with Lancaster**, as many shorter words will become totally obfuscated. The fastest algorithm here, and **will reduce your working set of words hugely** [FWC - though I perhaps don't care about the actual mapped-to words, but rather the words that are mapped to the same word], but if you want more distinction, not the tool you would want."
    * "Honestly, I feel that Snowball is usually the way to go. There are certain circumstances in which Lancaster will hugely trim down your working set, which can be very useful, however the marginal speed increase over snowball in my opinion is not worth the lack of precision"

#### [Quora: What's the best word2vec implementation?](https://www.quora.com/Whats-the-best-word2vec-implementation-for-generating-Word-Vectors-Word-Embeddings-of-a-2Gig-corpus-with-2-billion-words) (3/7/17)
* Gensim: https://rare-technologies.com/word2vec-tutorial (seems to be a nice impl)
* Tensorflow: https://github.com/jdwittenauer/ipython-notebooks/blob/master/notebooks/tensorflow/Tensorflow-5-Word2Vec.ipynb
* Google's impl: https://code.google.com/archive/p/word2vec/

#### [Extracting Relations between Non-Standard Entities using Distant Supervision and Imitation Learning](http://www.aclweb.org/anthology/D15-1086) (2/28/17)
* "jointly training the named entity classifier and the relation extractor using imitation learning which reduces structured prediction learning to classification learning"

#### [Word embeddings; word2vec; king - man + woman is queen, but why?](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html) (1/23/17)
* Often instead of working with conditional probabilities, we use the pointwise mutual information (PMI), defined as: PMI(a,b)=log[P(a,b)/(P(a)P(b))]=log[P(a|b)/P(a)]. Its direct interpretation is how much more likely we get a pair than if it were at random. The logarithm makes it easier to work with words appearing at frequencies of different orders of magnitude. We can approximate PMI as a scalar product: PMI(a,b)=v⃗_a⋅v⃗_b
* [Matrix Factorization with TensorFlow - Katherine Bailey](http://katbailey.github.io/post/matrix-factorization-with-tensorflow/)

#### [TensorFlow documentation on word embeddings](https://www.tensorflow.org/versions/master/tutorials/word2vec/) (1/17/17) (kws: representations)
* all methods depend in some way or another on the [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis), which states that words that appear in the same contexts share semantic meaning. The different approaches that leverage this principle can be divided into two categories: count-based methods (e.g. [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)), and predictive methods (e.g. [neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models)).
* Technically, this is called [Negative Sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), and there is good mathematical motivation for using this loss function: The updates it proposes approximate the updates of the softmax function in the limit.

#### Simple feed forward and convolutional NNs for language modeling (email, 12/16/16)
* FeeD forward - fast and slow weights implemented via varying degrees of momentum
* CNN - via sliding windows over collections of words, with all the typical pooling and translation and learning of subfeatures that goes on in image recog. NNs 

#### Distributional Semantics in R: Part 2 Entity Recognition w. {openNLP}
* https://www.r-bloggers.com/distributional-semantics-in-r-part-2-entity-recognition-w-opennlp/

#### Alternative to 1-hot encodings (FWC idea, 12/14/16)
* in NLP they don't learn misspellings very well
* so, instead, use (normalized) Levenshtein (edit) distance instead; for each word in the 1-hot vector, compute LD from a given word to each of those

#### [Dependency-Based Word Embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) (12/14/16)
* "While continuous word embeddings are gaining popularity, current models are based solely on linear contexts. In this work, we generalize the skip-gram model with negative sampling introduced by Mikolov et al. to include arbitrary contexts."

#### [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/linear-classify/#softmax-classifier) (12/8/16)
* "If you’ve heard of the binary Logistic Regression classifier before, the Softmax classifier is its generalization to multiple classes"
* "we are therefore minimizing the negative log likelihood of the correct class, which can be interpreted as performing Maximum Likelihood Estimation (MLE)"
* "the Softmax classifier is never fully happy with the scores it produces: the correct class could always have a higher probability and the incorrect classes always a lower probability and the loss would always get better. However, the SVM is happy once the margins are satisfied and it does not micromanage the exact scores beyond this constraint. This can intuitively be thought of as a feature: For example, a car classifier which is likely spending most of its “effort” on the difficult problem of separating cars from trucks should not be influenced by the frog examples"

### [Sebastian Ruder, On word embeddings, Part 2: Approximating the softmax](http://sebastianruder.com/word-embeddings-softmax/) (12/18/16)
* In the following we will discuss different strategies that have been proposed to approximate the softmax.
  * hierarchical softmax (h-softmax) - words are leaves of a binary tree - speedups for word prediction tasks of at least 50x and is thus critical for low-latency tasks - Notably, we are only able to obtain this speed-up during training, when we know the word we want to predict (and consequently its path) in advance
  * Recall that the information content I(w) of a word w is the negative logarithm of its probability p(w): I(w)=−log_2[p(w)].  The entropy H of all words in a corpus is then the expectation of the information content of all words in the vocabulary.  If we manage to encode more information into the tree, we can get away with taking shorter paths for less informative words [FWC - Huffman]
    * "We can render this value more tangible by observing that a model with a perplexity of 572 is as confused by the data as if it had to choose among 572 possibilities for each word uniformly and independently. To put this into context: The state-of-the-art language model by Jozefowicz et al. (2016) achieves a perplexity of 24.2 per word on the 1B Word Benchmark. Such a model would thus require an average of around 4.60 bits to encode each word, as 24.60=24.2, which is incredibly close to the experimental lower bounds documented by Shannon. If and how we could use such a model to construct a better hierarchical softmax layer is still left to be explored."
  * differentiated softmax - not all words require the same number of parameters so use a sparse matrix arranged in blocks sorted by frequency [FWC - this could be a useful way of representing data, some sets of stats are much more likely than others] - In contrast to H-Softmax, this speed-up persists during testing (the fastest method during testing)
  * CNN-softmax - instead of storing an embedding matrix of dx|V|, we now only need to keep track of the parameters of the CNN [FWC - like factorizing the matrix] - difficult to differentiate between similarly spelled words with different meanings
* Sampling-based Approaches - only useful at training time -- during inference (at test time) the full softmax still needs to be computed to obtain a normalised probability.
  * Importance Sampling - approximate the expected value of any probability distribution using the Monte Carlo method
  * Noise Contrastive Estimation (NCE) ([TensorFlow](https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html)) - **train a model to differentiate the target word from noise** - reduce the problem of predicting the correct word to a binary classification task, where the model tries to distinguish positive, genuine data from noise samples - use **logistic regression to minimize the negative log-likelihood, i.e. cross-entropy of our training examples against the noise** - For more information on NCE, Chris Dyer has published some excellent notes [21](https://arxiv.org/abs/1410.8251).
  * ************ **Negative Sampling (NEG)** ************ - Skip Gram with Negative Sampling (SGNS aka word2vec) - the objective that has been popularised by Mikolov et al. (2013), can be seen as an approximation to NCE - **learn high-quality word representations rather than achieving low perplexity** - For more insights on the derivation of NEG, have a look at Goldberg and Levy's notes [[22](https://arxiv.org/abs/1402.3722)].  Negative Sampling does not work well for language modelling, but it is generally superior for learning word representations.
* See 'Table 1: Comparison of approaches to approximate the softmax for language modelling'
* See [TensorFlow Documentation](https://www.tensorflow.org/extras/candidate_sampling.pdf) for another comparison
* if you are looking to actually use the described methods, [TensorFlow](https://www.tensorflow.org/versions/master/api_docs/python/nn.html#candidate-sampling) has implementations for a few sampling-based approaches and also explains the differences between some of them [here](https://www.tensorflow.org/extras/candidate_sampling.pdf).

### [Sebastian Ruder, On word embeddings, Part 1](http://sebastianruder.com/word-embeddings-1/index.html) (12/7/16)
* Bengio (2003): "the final softmax layer (more precisely: the normalization term) as the network's main bottleneck, as the cost of computing the softmax is proportional to the number of words in V [FWC - the vocabulary], which is typically on the order of hundreds of thousands or millions."
* [C&W's "solution](http://sebastianruder.com/word-embeddings-1/index.html#fn:4) to avoid computing the expensive softmax is to use a different objective function: Instead of the cross-entropy criterion of Bengio et al., which maximizes the probability of the next word given the previous words, Collobert and Weston train a network to **output a higher score fθ for a correct word sequence** (a probable word sequence in Bengio's model) than for an incorrect one." ... but ... "they keep the intermediate fully-connected hidden layer (2.) of Bengio et al. around (the HardTanh layer in Figure 3), which constitutes another source of expensive computation"
* "Technically however, **word2vec** is not be considered to be part of deep learning, as its architecture is neither deep nor uses non-linearities"
  * continuous bag of words (CBOW): "use both the n words before and after the target word w_t to predict it"
  * "**skip-gram** turns the language model objective on its head: Instead of using the surrounding words, v_{w_t+j} (or v_O, "o" for output), to predict the centre word, v_{w_t} (or v_I)  as with CBOW, skip-gram uses the centre word to predict the surrounding words"
  * "As the skip-gram architecture does not contain a hidden layer that produces an intermediate state vector h, h is simply the word embedding v_{w_t} (or v_I) of the input word w_t (or I)"
  * "In the next post [see above], we will discuss different ways to approximate the expensive softmax as well as key training decisions that account for much of skip-gram's success"

#### Looks like someone's already done my 'fuzzy ESG' idea
* In [Grammar as a Foreign Language](http://arxiv.org/abs/1412.7449), the authors use a Recurrent Neural Network with attention mechanisk to generate sentence parse trees. The visualized attention matrix gives insight into how the network generates those trees [http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/]
* "Attention = (Fuzzy) Memory?"
* Trained on **Berkley Parser** and ended up being a fair amount better than the BP

### [Deep Learning, NLP, and REPRESENTATIONS](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) (11/29/16)
* "This seems to be a great strength of neural networks: **they learn better ways to represent data, automatically. Representing data well, in turn, seems to be essential to success at many machine learning problems. Word embeddings are just a particularly striking example of learning a representation**." [FWC idea - **Compute company embeddings the same way, using negative sampling on company statistics, then use those embeddings to predict returns.  This might be analogous to PCA, e.g. rather than running PCA to orthogonalize company characteristics (as mentioned in lecture 6 of Geoff Hinton's ML course), use embeddings to orthogonalize/factorize.  This might be a better approach than training LSTMs on raw data.**]
* "There’s a counterpart to this trick. Instead of learning a way to represent one kind of data and using it to perform multiple kinds of tasks, we can learn a way to map multiple kinds of data into a single representation!" [FWC - **multiple kinds of data == stats/forecasts that predict positive vs. negative future returns => combined forecast (i.e. better way to combine forecasts than mere addition)**]  "forcing two languages [forecasts] to line up at different points, they overlap and other points get pulled in the right direction"
  * **Another analog to word2vec (aka skip gram w/ negative sampling, SGNS) would be to predict company characteristics from (known) future returns--rather than the other way around.  Multiple companies with similar returns could be learned at once.  And then 'negative' examples could be generated by mutating the company characteristics in random ways.**
  * Another idea: put a T/F neuron on the end of an autoencoder to allow for negative sampling by random mutation of the decoder target. (see "denoising autoencoder")
* "The basic idea is that one classifies images [FWC - or written text, a research article or a news story] by outputting a vector in a word embedding [FWC - based on structured data used to write the article or produce the news story].  The interesting part is what happens when you test the model on new classes of images [FWC - new future scenarios never before seen]."
* "These results all exploit a sort of “these words are similar” reasoning. But it seems like much stronger results should be possible based on relationships between words. In our word embedding space, there is a consistent difference vector between male and female version of words. Similarly, in image space, there are consistent features distinguishing between male and female. Beards, mustaches, and baldness are all strong, highly visible indicators of being male. Breasts and, less reliably, long hair, makeup and jewelery, are obvious indicators of being female. **Even if you’ve never seen a king before, if the queen, determined to be such by the presence of a crown, suddenly has a beard, it’s pretty reasonable to give the male version**."
* "The **representation perspective** of deep learning is a powerful view that seems to answer why deep neural networks are so effective. Beyond that, I think there’s something extremely beautiful about it: **why are neural networks effective? Because better ways of representing data can pop out of optimizing layered models**."
* ************** FWC - so here's the model: **************
  1. collect together a bunch of related input variables, say 10-20 (this may not be necessary w/ dropout)
  2. representation: neutralize all of them to each other using a scatter-plot-smoother-like NN (don't just take ratios b/c of problems with non-linear ratio distributions)
    * consider the "banana correction" (rotation of non-linear/exponential s.t. errors are less heteroskedastic)
  3. use these ratio-like, neutralized variables as input to a negative sampling, word2vec-like algorithm to generate embeddings
    * see "denoising autoencoder" note on [[Notes for Geoff Hinton's Coursera ML course]]
    * also see "constructive distraction" and "unsupervised pretraining" notes for RBMs on the same page
    * scrap everything else mentioned: here is the **current best representation framework**, see notes on Generative Adversarial Networks (GANs) from LeCun's talk and read this paper: file:///home/fred/Documents/articles/adversarial_training/unsupervised_repr_learning_with_GANs_1511.06434v2.pdf (Q: Can one merely generate false outputs in a 1D output space (e.g. returns) and let the Discriminator decide between true and false?)
  4. use the embeddings in a forecasting model
    * use (mini-batch-induced-randomization) Bayesian MCMC approach to sample parameter space (or, if using NNs, use dropout ("An alternative to doing the correct Bayesian thing. Probably doens't work quite as well, but much more practical"); again see [[Notes for Geoff Hinton's Coursera ML course]])
  5. alternatively, identify known models in learned relationships
    * for example, CvsC works in the bottom 20% of the resid vol distribution, so it should be possible to identify a set of neurons, perhaps starting with Px and RV input neurons, that end up coding the model
    * such models can possibly be used as jumping-off points from where to start searching for other hard-coded models (or learning posterior reasons to explore futher)

#### [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/index.html)

#### [Deep or Shallow, NLP is Breaking Out](http://cacm.acm.org/magazines/2016/3/198856-deep-or-shallow-nlp-is-breaking-out/fulltext) (11/1/16)
* "Yoav Goldberg and Omer Levy, researchers at Bar-Ilan University in Ramat-Gan, Israel, have concluded much of [word2vec's] power comes from tuning algorithmic elements such as dynamically sized context windows. Goldberg and Levy call those elements hyperparameters."
* "Basically, where GloVe precomputes the large word x word co-occurrence matrix in memory and then quickly factorizes it, word2vec sweeps through the sentences in an online fashion, handling each co-occurrence separately," Rehurek, who created the open source modeling toolkit gensim and optimized it for word2vec, wrote. "So, there is a trade-off between taking more memory (GloVe) vs. taking longer to train (word2vec)."

#### [On Word Embeddings, Part 3: The Secret Ingredients of Word2Vec](http://sebastianruder.com/secret-word2vec/index.html) (10/26/16, Sebastian Ruder)
* "While GloVe is considered a "*predict*" model [FWC - as opposed to a "*count*" model, e.g. DS or LSA] by Levy et al. (2015), it is **clearly factorizing a word-context co-occurrence matrix**, which brings it close to traditional methods such as PCA and LSA. Even more, Levy et al. [4] demonstrate that word2vec implicitly factorizes a word-context PMI matrix"
* "[Levy et al. 2015](https://transacl.org/ojs/index.php/tacl/article/view/570) find that SVD -- and not one of the word embedding algorithms -- performs best on similarity tasks, while SGNS performs best on analogy datasets"
  * "**Hyperparameter settings are often more important than algorithm choice.**"
  * "SNGS [word2vec] outperforms GloVe on all tasks."
* Recommendations -- and one of the things I like most about the paper -- we can give concrete practical recommendations:
  * DON'T use shifted PPMI with SVD.
  * **DON'T use SVD "correctly"**, i.e. without eigenvector weighting (performance drops 15 points compared to with eigenvalue weighting with p=0.5).
  * DO use PPMI and SVD with short contexts (window size of 2).
  * DO use many negative samples with SGNS.
  * DO always use context distribution smoothing (raise unigram distribution to the power of α=0.75) for all methods.
  * DO use SGNS as a baseline (robust, fast and cheap to train).
  * DO try adding context vectors in SGNS and GloVe.

#### Good lecture on word2vec and GloVe (10/19/16)
* Slides from Mar 31 lecture http://cs224d.stanford.edu/syllabus.html (which are also in email)

#### [Question answering on the Facebook bAbi dataset using recurrent neural networks and 175 lines of Python + Keras](http://smerity.com/articles/2015/keras_qa.html) (7/7/16)

#### [Google Has Open Sourced SyntaxNet, Its AI for Understanding Language](http://www.wired.com/2016/05/google-open-sourced-syntaxnet-ai-natural-language/) (5/13/16)
* "According to Google, Parsey McParseface is about 94 percent accurate in identifying how a word relates the rest of a sentence, a rate the company believes is close to the performance of a human (96 to 97 percent)."
* "people use language on the web in so many different ways. When Google trains its neural nets with this kind of dataset, the accuracy rate drops to about 90 percent"

#### NLP "features" shouldn't be functions of text passages, but rather correlations to sets of terms (just like QELA).  The relationships/correlations between the pairwise text-QELA terms are the features.

#### [State of the art models for anaphora/ coreference resolution?](https://www.reddit.com/r/MachineLearning/comments/4gptkp/state_of_the_art_models_for_anaphora_coreference/) (5/2/16)
* Here's the paper: [Learning Anaphoricity and Antecedent Ranking Features for Coreference Resolution](http://people.seas.harvard.edu/~srush/acl15.pdf)

#### [Sentiment Analysis in Python with TextBlob](https://github.com/shekhargulati/52-technologies-in-2016/blob/master/11-textblob/README.md) (3/19/16)

#### [Fuzzy string matching using cosine similarity](http://blog.nishtahir.com/2015/09/19/fuzzy-string-matching-using-cosine-similarity/)

#### Naive Bayes
* https://en.wikipedia.org/wiki/Naive_Bayes_classifier
* http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning