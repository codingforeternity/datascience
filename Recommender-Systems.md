#### [The WellDressed Recommendation Engine](https://deeplearning4j.org/welldressed-recommendation-engine) (2/17/17)
* "data feeds from merchants offer few categorization options, and on top of that, every data feed is very different from the other"
* "I decided to focus on Title, Description and Category/Keyword fields. Title and Description tend to give valuable, specific hints on what a garment actually is. Category works as a broad identification.  I do not use image data for garment identification.... The copy of the text tends to be the only true identifier of a garment type."
* "I use [word2vec](https://deeplearning4j.org/word2vec.html) to create vectors. A word needs to show at least 10 times, and I get 40 vectors for each wordset. I then proceed to sum all vectors per title, description and category, resulting in 120 vectors."
  * FWC - I think what he means by this is that he creates "40-*dimensional* vector" embeddings and then sums over the words in each of the 3 wordsets to get a 120-dimensional embedding vector for each item/garment.
* "The database has a lot of jeans and t-shirts, but very few tuxedoes and cummerbunds"
  * He has 84 garment types so he samples 1 from each of the 84 and he does this 1000 times to get an 84,000-point set.
  * So how does one count the number of URL/document types?  One possible way could be to treat each of the 120 dimensions of the item embeddings as a different type--and sample uniformly from each of those 120 *feature type* classes.
* **The algorithm described here is more for classification than for recommendations/ratings**.  It seems to be used as input to a recommendation system (i.e. first classify garment type then recommend one of the garments of a given type--filter)  It seems to predict one of 84 labels/categories given a 120-dim vector.
  * As noted, the author's first solution used "keywords with a complex rule-and weight-system" to predict category. "Title and Description tend to give valuable, specific hints on what a garment actually is."  This is the crux: figuring out what type of a garment is.

#### [Mozilla Context Graph: A recommendation system for the web](http://venturebeat.com/2016/07/06/mozilla-is-building-context-graph-a-recommender-system-for-the-web/)

#### [The truth is that people just don't dislike things online that often](http://www.benfrederickson.com/rating-set-distributions/)
* even if they did offer a dislike button, people just wouldn't use it often enough to justify the precious screen real estate it would take up.
* Most ratings are positive: more people watch good movies.
  * 15% of ratings are negative, but 30% of items rated have negative ratings.

#### Summary of techniques
  1. Information Retrival - based on various similarity metrics of word count (user) distributions for documents (purchased items)
  2. Latent Semantic Analysis - use SVD to factorize word x document (or user x item) matrix and then reconstruct with fewer dimensions
  3. Collaborative Filtering (aka Alternating Least Squares)
  4. Restricted Boltzmann Machines (see Hinton course notes)

#### ["People Who Like This Also Like ... " Part 2: Finding Similar Music using Matrix Factorization](http://www.benfrederickson.com/matrix-factorization/) (1/31/17)
* "Matrix Factorization methods can generate matches that are impossible to find with the techniques in my original [IR] post [below]."
* Covered techniques: Latent Semantic Analysis ([SVD](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/) of BM25 weights), (Implicit) Alternating Least Squares (aka Collaborative Filtering (for implicit datasets))
* Spotify uses a matrix factorization technique called [Logistic Matrix Factorization]() to generate their lists of related artists. This method has a similar idea to Implicit ALS: it's a confidence weighted factorization on binary preference data - but uses a logistic loss instead of a least squares loss

#### ["People Who Like This Also Like ... " Part 1: Distance Metrics for Fun and Profit](http://www.benfrederickson.com/distance-metrics/) (1/30/17)
* "most common way of dealing with this problem is [Jaccard distance](http://en.wikipedia.org/wiki/Jaccard_index), which normalizes the intersection size of the two sets by dividing by the total number of users that have listened to either artist."
* "Cosine based methods: The set based methods throw away a ton of useful information: how often the user has listened to the artist."
  * "Cosine distance succeeds in bringing up more relevant similar artists than the set based methods, but unfortunately there is also significantly more noise"
  * "While there are a bunch of more principled methods to overcome this ... one simple hack that I've seen used before is to smooth the cosine by the number of overlapping users
* "BM25 usually produces much better results than TF-IDF ... between the step function used in the Jaccard distance (K1 = 0) and the linear weighting used in the Cosine distance (K1 = +infinity)"
*  [Information Retrieval: Implementing and Evaluating Search Engines](https://www.amazon.ca/Information-Retrieval-Implementing-Evaluating-Engines/dp/0262026511): Learning to Rank, Relevance Feedback and Search Result Fusion. The Manning IR book is also decent and [freely available online](http://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf). [FWC - or just use [Lucene](http://lucene.apache.org/core/4_9_1/core/org/apache/lucene/search/similarities/TFIDFSimilarity.html#formula_tf)?]

#### [Wikipedia: Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize)

#### [Stackoverflow: How to Implement A Recommendation System?](https://stackoverflow.com/questions/6302184/how-to-implement-a-recommendation-system#6302223) (1/30/17)
* use keywords (also has a link to a [list of stop words](http://en.wikipedia.org/wiki/Stop_words))

#### [Scaling Recommendation Engine: 15,000 to 130M Users in 24 Months [RS Labs]](https://www.retentionscience.com/scalingrecommendations/?utm_campaign=Data%2BElixir&utm_medium=email&utm_source=Data_Elixir_116) (1/30/17)
* this company appears to have built itself on recommendation systems
* "The resultant rule: 'Select the top items from the category that the user has already bought.' ... The users' purchase categories turned out to be a very strong latent factor"
* "The 'jaccard-semantic duplication annotation scheme,' was implemented which would figure out duplicate items and categories based on their textual description."
* "This led us to tap the feedback data to figure out specifically which recs schemes were performing well in order to gain insight into items that interested people."
* "metrics to measure the diversity and novelty of our recommendations"
* "we gambled with Spark. Again, it proved to be a good decision, as Spark soon became the industry leader for Machine Learning and Big Data analytics due to its elegant APIs for manipulating distributed data, growing machine learning library, and effective fault tolerance mechanisms"
* "Scala was adapted as the language of choice and refactored all our procedural code into a single functional machine learning repository"
* "Our current architecture looks like [this](https://www.retentionscience.com/wp-content/uploads/2017/01/image8.png)"

#### [Matrix Factorization with Tensorflow](http://katbailey.github.io/post/matrix-factorization-with-tensorflow/)