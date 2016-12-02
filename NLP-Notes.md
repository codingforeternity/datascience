[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) (11/29/16)
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

[CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/index.html)

[Deep or Shallow, NLP is Breaking Out](http://cacm.acm.org/magazines/2016/3/198856-deep-or-shallow-nlp-is-breaking-out/fulltext) (11/1/16)
* "Yoav Goldberg and Omer Levy, researchers at Bar-Ilan University in Ramat-Gan, Israel, have concluded much of [word2vec's] power comes from tuning algorithmic elements such as dynamically sized context windows. Goldberg and Levy call those elements hyperparameters."
* "Basically, where GloVe precomputes the large word x word co-occurrence matrix in memory and then quickly factorizes it, word2vec sweeps through the sentences in an online fashion, handling each co-occurrence separately," Rehurek, who created the open source modeling toolkit gensim and optimized it for word2vec, wrote. "So, there is a trade-off between taking more memory (GloVe) vs. taking longer to train (word2vec)."

[On Word Embeddings, Part 3: The Secret Ingredients of Word2Vec](http://sebastianruder.com/secret-word2vec/index.html) (10/26/16, Sebastian Ruder)

Good lecture on word2vec and GloVe (10/19/16)
* Slides from Mar 31 lecture http://cs224d.stanford.edu/syllabus.html (which are also in email)

[Question answering on the Facebook bAbi dataset using recurrent neural networks and 175 lines of Python + Keras](http://smerity.com/articles/2015/keras_qa.html) (7/7/16)

[Google Has Open Sourced SyntaxNet, Its AI for Understanding Language](http://www.wired.com/2016/05/google-open-sourced-syntaxnet-ai-natural-language/) (5/13/16)
* "According to Google, Parsey McParseface is about 94 percent accurate in identifying how a word relates the rest of a sentence, a rate the company believes is close to the performance of a human (96 to 97 percent)."
* "people use language on the web in so many different ways. When Google trains its neural nets with this kind of dataset, the accuracy rate drops to about 90 percent"

NLP "features" shouldn't be functions of text passages, but rather correlations to sets of terms (just like QELA).  The relationships/correlations between the pairwise text-QELA terms are the features.

[State of the art models for anaphora/ coreference resolution?](https://www.reddit.com/r/MachineLearning/comments/4gptkp/state_of_the_art_models_for_anaphora_coreference/) (5/2/16)
* Here's the paper: [Learning Anaphoricity and Antecedent Ranking Features for Coreference Resolution](http://people.seas.harvard.edu/~srush/acl15.pdf)

[Sentiment Analysis in Python with TextBlob](https://github.com/shekhargulati/52-technologies-in-2016/blob/master/11-textblob/README.md) (3/19/16)

[Fuzzy string matching using cosine similarity](http://blog.nishtahir.com/2015/09/19/fuzzy-string-matching-using-cosine-similarity/)

Naive Bayes
* https://en.wikipedia.org/wiki/Naive_Bayes_classifier
* http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning