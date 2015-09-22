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
* "back propagation is a practical application of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), but it took til the 80s to realize this"
  * FWC - the derivative of `f(g(x)` is `f'(g(x)) * g'(x)`
  * i.e. you can take the derivatives from the last layer of a NN (`g'`) and use them to compute the derivatives of the second to last layer (`f'`) ... and so on and so on backwards through the layers
  * [specifically](http://www.willamette.edu/~gorr/classes/cs449/linear2.html) we want the partial derivative of the cost function, J, w.r.t. the weights/parameters, W/Theta:
    * dJ/dW
    * using the chain rule, this can be decomposed into dJ/dy * dy/dW
    * d/dy J = d/dy 0.5*sum((Wa-y)^2) = sum(-(Wa-y)) (where Wa = h(x), the hidden/previous layer activations)
    * d/dW y = d/dW Wa = a
    * so dJ/dW = sum(-(Wa-y)) * a
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
* Industry picked up on DL faster than academia due to resistance
* Facebook AI code is mostly open source
* Yann is teaching (or taught) a course at NYU this past spring (?) that is supposedly freely available w/ lectures on the web
* the window is going to close very quickly for startups in DL due to a few reasons
  * not easy to get data
  * good people already hired by Google/FB
  * good companies already sold to Google/FB
  * there was a gold rush over 2014, but that window is closing
* Q: as a startup, better to provide a vertical (general) or horizontal (specific) solution?
  * A: vertical, b/c specifics like image recognition already solved
* Yann is a co-founder of [MuseAmi](http://www.museami.com/) - DL for music related stuff
* medical imaging not very well explored and there are still opportunities for small companies
* the major undiscovered DL principle is unsupervised learning -- e.g. learning how the brain really works
* NNs are very weakly inspired by neuroscience
  * akin to how airplanes are inspired by birds
  * need yet to discover the underlying principles of intelligence
  * just like aerodynamics are the underlying principles of birds/planes/flight
  * a bird specialist would talk a lot about feathers, if asked, but that doesn't figure at all into planes
* Jeff Inten has a Coursera course on ML, but it doesn't cover NLP (negative according to Yann)
* Yoshi Banjo has a [free text book online](http://www.iro.umontreal.ca/~bengioy/dlbook/) co-authored w/ Aaron Courville and Ian Goodfellow
* Yann has an [NVidia Webinar](https://www.facebook.com/yann.lecun/posts/10152290017742143) (watch it) and also see [these](http://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php)