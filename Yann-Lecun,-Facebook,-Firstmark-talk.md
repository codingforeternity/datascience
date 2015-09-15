https://www.youtube.com/watch?v=AbjVdBKfkO0
Yann Lecun, Facebook // Artificial Intelligence // Data Driven #32 (Hosted by FirstMark Capital) 

FirstMark Capital is a NYC venture capital firm.  Attendees included a dude from Citrix who identified himself before asking a question.

* used to be 2 big ML rivals: either Deep Learning (i.e Convolutional Neural Nets) or SVMs
* "I'm sure those of you who have done data science spend a lot of time selecting/engineering features, but then your classifier is just a standard module--e.g. logistic regression or random forests or whatever your favorite method is.  And so if you can automate the process of engineering the features, there's a lot more problems you can apply ML to."
* "DL is a conspiracy: to pick techniques that would be interesting and move away from SVMs...unbelievable success."
* "companies that have the data Google, FB, IBM--not really IBM--are in a position to take advantage relative to companies that have the technology bug not the data"
* DL - automate the process of feature engineering
  * e.g. pixels aren't very informative so you have to combine them
  * blocks of pixels are correlated  => there's a more efficient way to represent blocks
  * pixels -> patches/blocks -> motifs -> parts of objects -> objects -> etc.
  * a hierarchical structure!
* "back propagation is a practical application of the chain rule [obviously--FWC], but it took til the 80s to realize this [obviously--FWC]"
* size of typical NNs
  * hundreds of thousands of inputs
  * 1-10 billion multiply-accumulate operations (can't do this on CPUs--need GPUs)
  * 100s of millions of internal neurons
  * repeat 100s of millions of times to train properly
* timeline
  * speech recognition handled solely by DL since 2011 (Apple, Google, Microsoft)
  * image recognition since 2012
  * NLP is next
    * embedding methods
      * mapping words to vectors
      * meaning of word (1 vector) plus syntactic role (e.g. noun, verb, etc.; another vector)
    * Word2Vec
      * this is a specific technique
      * compositional properties, e.g. <Paris> - <UK> = <London>
* "We cease to be the lunatic fringe.  We're now the lunatic core."