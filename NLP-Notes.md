### [Sebastian Ruder, On word embeddings, Part 2: Approximating the softmax](http://sebastianruder.com/word-embeddings-softmax/)
* In the following we will discuss different strategies that have been proposed to approximate the softmax.
  * hierarchical softmax (h-softmax) - words are leaves of a binary tree - speedups for word prediction tasks of at least 50x and is thus critical for low-latency tasks - Notably, we are only able to obtain this speed-up during training, when we know the word we want to predict (and consequently its path) in advance
  * Recall that the information content I(w) of a word w is the negative logarithm of its probability p(w): I(w)=−log_2[p(w)].  The entropy H of all words in a corpus is then the expectation of the information content of all words in the vocabulary.  If we manage to encode more information into the tree, we can get away with taking shorter paths for less informative words [FWC - Huffman]
    * "We can render this value more tangible by observing that a model with a perplexity of 572 is as confused by the data as if it had to choose among 572 possibilities for each word uniformly and independently. To put this into context: The state-of-the-art language model by Jozefowicz et al. (2016) achieves a perplexity of 24.2 per word on the 1B Word Benchmark. Such a model would thus require an average of around 4.60 bits to encode each word, as 24.60=24.2, which is incredibly close to the experimental lower bounds documented by Shannon. If and how we could use such a model to construct a better hierarchical softmax layer is still left to be explored."
  * differentiated softmax - not all words require the same number of parameters so use a sparse matrix arranged in blocks sorted by frequency **[FWC - this could be a useful way of representing company data, some sets of company stats are much more likely than others]** In contrast to H-Softmax, this speed-up persists during testing (the fastest method during testing)
  * CNN-softmax - instead of storing an embedding matrix of dx|V|, we now only need to keep track of the parameters of the CNN [FWC - like factorizing the matrix] - difficult to differentiate between similarly spelled words with different meanings
* Sampling-based Approaches - only useful at training time -- during inference (at test time) the full softmax still needs to be computed to obtain a normalised probability.

### [Sebastian Ruder, On word embeddings, Part 1](http://sebastianruder.com/word-embeddings-1/index.html)
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

#### [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) (11/29/16)
* "This seems to be a great strength of neural networks: they learn better ways to represent data, automatically. Representing data well, in turn, seems to be essential to success at many machine learning problems. Word embeddings are just a particularly striking example of learning a representation." [FWC idea - **Compute company embeddings the same way, using negative sampling on company statistics, then use those embeddings to predict returns.  This might be analogous to PCA, e.g. rather than running PCA to orthogonalize company characteristics (as mentioned in lecture 6 of Geoff Hinton's ML course), use embeddings to orthogonalize.  This might be a better approach than training LSTMs on raw data.**]
* "There’s a counterpart to this trick. Instead of learning a way to represent one kind of data and using it to perform multiple kinds of tasks, we can learn a way to map multiple kinds of data into a single representation!" [FWC - **multiple kinds of data == stats/forecasts that predict positive vs. negative future returns => combined forecast**]  "forcing two languages [forecasts] to line up at different points, they overlap and other points get pulled in the right direction"
* "The basic idea is that one classifies images [FWC - or written text, a research article or a news story] by outputting a vector in a word embedding [FWC - based on structured data used to write the article or produce the news story].  The interesting part is what happens when you test the model on new classes of images [FWC - new future scenarios never before seen]."
* "These results all exploit a sort of “these words are similar” reasoning. But it seems like much stronger results should be possible based on relationships between words. In our word embedding space, there is a consistent difference vector between male and female version of words. Similarly, in image space, there are consistent features distinguishing between male and female. Beards, mustaches, and baldness are all strong, highly visible indicators of being male. Breasts and, less reliably, long hair, makeup and jewelery, are obvious indicators of being female. **Even if you’ve never seen a king before, if the queen, determined to be such by the presence of a crown, suddenly has a beard, it’s pretty reasonable to give the male version**."
* "The **representation perspective** of deep learning is a powerful view that seems to answer why deep neural networks are so effective. Beyond that, I think there’s something extremely beautiful about it: **why are neural networks effective? Because better ways of representing data can pop out of optimizing layered models**."
* FWC - so here's the model:
  1. collect together a bunch of related input variables, say 10-20
  2. neutralize all of them to each other using a scatter-plot-smoother-like NN (don't just take ratios b/c of problems with non-linear ratio distributions)
  3. use these ratio-like, neutralized variables as input to a negative sampling, word2vec-like algorithm to generate embeddings
  4. use the embeddings in a forecasting model

#### [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/index.html)

#### [Deep or Shallow, NLP is Breaking Out](http://cacm.acm.org/magazines/2016/3/198856-deep-or-shallow-nlp-is-breaking-out/fulltext) (11/1/16)
* "Yoav Goldberg and Omer Levy, researchers at Bar-Ilan University in Ramat-Gan, Israel, have concluded much of [word2vec's] power comes from tuning algorithmic elements such as dynamically sized context windows. Goldberg and Levy call those elements hyperparameters."
* "Basically, where GloVe precomputes the large word x word co-occurrence matrix in memory and then quickly factorizes it, word2vec sweeps through the sentences in an online fashion, handling each co-occurrence separately," Rehurek, who created the open source modeling toolkit gensim and optimized it for word2vec, wrote. "So, there is a trade-off between taking more memory (GloVe) vs. taking longer to train (word2vec)."

#### [On Word Embeddings, Part 3: The Secret Ingredients of Word2Vec](http://sebastianruder.com/secret-word2vec/index.html) (10/26/16, Sebastian Ruder)

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