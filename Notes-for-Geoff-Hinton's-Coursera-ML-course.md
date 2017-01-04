Octave source: https://ftp.gnu.org/gnu/octave/

Also see accompanying green Staples notebook.

### Week 2: Types of Neural Network Architectures
* Ilya Sutskever (2011) trained special type or RNN to predict next char in sequence
  * it generates text by predicting the probability distribution for the next char, not the highest likely next char (which would generate text like "the united states of the united states of the..."), and then sampling from that distribution
* symmetric nets are much easier to analyze than RNNs (John Hopfield)
  * but they're more restricted, e.g. can't learn cycles

### Week 2: What perceptrons can't do
* A perceptron cannot recognize patterns under translation if we allow wraparound.
  * E.g. 2 binary input vectors, A and B, each with 4 out of 10 activated "pixels."  Each of the 10 pixels will be activated by 4 translations of A and of B, so the total input received by the decision unit over all these patterns, in both cases, will be four times the sum of all the weights.  But to discriminate correctly, every single case of pattern A must provide more input to the decision unit than every single case of pattern B.
  * However, if we have 3 patterns assigned to two classes, A and B, and A contains a pattern with 4 pixels while B contains patterns with 1 and 3 pixels then a *binary* decision unit can classify if we allow translations and wraparound.  Weight = 1 from each pixel with bias of -3.5
  * Minskey and Papert's "Group Invariance Theorem" says that the part of a perceptron that learns cannot do this if the transformations form a group (e.g. translations with wraparound).
  * This result is devastating for pattern recognition (PR) because the whole point of PR is to recognize patterns that undergo transformations, like translation.
  * To deal with such transformations, a perceptron needs to use multiple feature units to recognize transformations of informative sub-patterns.
  * So the tricky part of PR must be solved by the hand-coded feature detectors, not the learning procedure.
  * Networks without hidden units are very limited in what they can learn to model.

### Week 3: Learning the weights of a linear neuron
* Instead of showing the weights get closer to a good set of weights (i.e. perceptrons, which suffer from 2 good set of weights do not average to a good set) show that actual output values get closer to the target values.
  * In perceptron learning the outputs can get farther away from the targets, even though the weights are getting closer.
* The "delta rule" for learning: delta w_i = epsilon * x_i * (t - y) ... where epsilon := "learning rate", t := target/true output, and y := estimated output
  * Derivation: Error = 0.5 * Sum_{n in training}[(t_n - y_n)^2] ... the 1/2 is only there to make the 2 in the derivative cancel
  * Differentiate the error wrt one of the weights, w_i: del E / del w_i = 0.5 * Sum_n[del y_n / del w_i * d E_n / d y_n] ... Chain Rule ("easy to remember, just cancel those 2 del y_ns ... but only when there aren't any mathematicians looking) ... = -Sum_n[x_{i,n} * (t_n - y_n)]
    * del y_n / del w_i = x_{i,n} because y_n = w_i * x_{i,n}
    * d E_n / d y_n is just the derivative of the (squared) Error function
  * Therefore: delta w_i = -epsilon * del E / del w_i

### Week 3: The error surface for a linear neuron
* Difference between "batch" and "on-line"
  * Simplest batch learning does steepest gradient descent
  * On-line/stochastic zig-zags between training case "lines" at each step moving perpendicularly to a line.  Imagine the intersection of 2 training case lines and moving perpendicularly back and forth perpendicularly to both while converging on their intersection point.
    * This is very slow if the variables are highly correlated (very elongated ellipse) because the perpendicular updates (the gradients) are also perpendicular to the intersection.
    * FWC idea - look at angle between consecutive gradients to detect correlated dimensions

### Week 3: The backpropagation algorithm
* Networks without hidden layers are very limited in the input-output mappings they can model
* Adding a layer of hand-coded features (as in perceptron) makes them much more powerful, but the hard bit is designing the features
  * We would like to find good features without requiring insights into the task or repeated trial and error of different features
  * We need to automate the trial and error feature designing loop
  * FWC - but again the trick here is to properly incorporate priors into this design
* **Reinforcement learning: learning by perturbing weights (an idea that occurs to everyone)**
  * Randomly perturb one weight and see if it improves performance--if so, save the change
  * Very inefficient--requires multiple forward passes on a set of training cases just to update one weight.  Backprop much better "by a factor of the number of weights in the network."
  * Could randomly perturb all the weights in parallel and correlate performance gain w/ weight changes, but not much better b/c requires lots of trials on each training case to "see" the effect of changing one weight through the noise created by changing all the others (FWC - reminds me of the Shapley optimization)
  * Better idea: randomly perturb the activities of the hidden units.  Once we know how we want a hidden activity to change on a training case, we can compute how to change the weights.  There are fewer activities than weights, but backprop still wins by a factor of the number of neurons.
  * Finite Difference Approximation (compute +/- epsilon changes for each weight and move in that direction) works, but backprop finds the exact gradient (**del E / del w_{i,j}** = Sum_j[del z_j / del w_{i,j} * del E / del z_j] = Sum_j[w_{i,j} * del E / del z_j]) much more quickly.
 * FWC - machine learning (backprop) is all about the learning rate!  So you can either be smart (and use backprop) or buy more computers.  You're only constrained if you need both.  Search Google for: "machine learning for the maximization of an arbitrary function"
  * *Instead of using pre-set coefficients (desired activities) to train engineered features, use error derivatives wrt hidden activities.*  We can compute error derivatives for all of the hidden units efficiently at the same time: Once we have the error derivatives for the hidden activities (hidden neuron output) it's easy to compute the (input) weights going into a hidden unit.

### Week 3: Using the derivatives computed by backprop
* 2 types of noise: unreliable target values (small worry), sampling error (big worry)
* When we fit the model it cannot tell which regularities are real and which are caused by sampling error.  FWC - So are there methods then to distinguish between the two (besides e.g. cross validation)???  See week 7.
  * Weight decay - keep weights near 0
  * Weight sharing - keep some weights similar to each other
  * Early stopping - peek at a fake test set while training and stop when performance gets decent
  * Model averaging - train lots of NNs and average then
  * Bayesian fitting - fancy averaging
  * Dropout - randomly omitting hidden units when training
  * Generative pre-training
  * FWC idea - other constraints such as monotonicity and limits on distributions

### Week 4: Neural nets for machine learning
* Obvious way to express regularities is as symbolic **rules**
  * but finding the symbolic rules involves a difficult search through a large discreet space
  * so model as a NN instead w/
    * input := person1 + relationship (both 1-hot encodings)
    * output := person2 (also 1-hot)
* **Instead of predicting the 3rd term in a relationship, [A R B], we could provide all 3 as input and predict P([A R B] is correct)**
  * for this we'd need a whole bunch of "correct" facts as well as "incorrect" ones (fwc - **negative sampling**)

### 4b: A brief diversion into cognitive science
[probably not interesting if you're an engineer]
* There has been a long debate in cognitive science between two rival theories of what it means to have a concept:
  * The feature theory : A concept is a set of semantic features.
    * This is good for explaining similarities between concepts.
    * Its convenient: a concept is a vector of feature activities.
  * The structuralist theory: The meaning of a concept lies in its relationships to other concepts.
    * So conceptual knowledge is best expressed as a relational graph.
    * Minsky used the limitations of perceptrons as evidence against feature vectors and in favor of relational graph representations.
* Both sides are wrong
  * These two theories need not be rivals. A neural net can use **vectors of semantic features to implement a relational graph**.
    * In the neural network that learns family trees, no *explicit* inference is required to arrive at the intuitively obvious consequences of the facts that have been explicitly learned.
    * The net can “intuit” the answer in a forward pass.
  * We may use explicit rules for conscious, deliberate reasoning, but we do a lot of commonsense, analogical reasoning by just “seeing” the answer with no conscious intervening steps.
    * Even when we are using explicit rules, we need to just see which rules to apply.  [FWC - i.e. just "seeing" the answer is the same as just "seeing" which rules apply]
* Localist and distributed representations of concepts
  * The obvious way to implement a relational graph in a neural net is to treat a neuron as a node in the graph and a connection as a binary relationship. But this “localist” method will not work:
    * We need many different types of relationship and the connections in a neural net do not have discrete labels.
    * We need ternary relationships as well as binary ones.  e.g. A is between B and C.
  * **The right way to implement relational knowledge in a neural net is still an open issue.**
    * But many neurons are probably used for each concept and each neuron is probably involved in many concepts. This is called a “distributed representation”.  "*A many-to-many mapping between concepts and neurons.*"
  * a "**local**" representation of a word is a 1-hot vector of length equal to the size of the vocab
  * a "**distributed**" representation is a word embedding which is distributed b/c it is based on all the other words

### 4c: The softmax output function
softmax a way of forcing the outputs to sum to 1 so that they can represent a probability distribution across discrete, mutually exclusive alternatives
* The squared error measure has some drawbacks:
  * If the desired output is 1 and the actual output is 0.00000001 there is almost no gradient for a logistic unit to fix up the error.
  * If we are trying to assign probabilities to mutually exclusive class labels, we know that the outputs should sum to 1, but we are depriving the network of this knowledge.
  * [FWC - **I wonder if the changes to the loss function that are about to be described are a *dual* of a representation of constraints as output neurons.  E.g. could the effects of this different loss function be obtained by changing the architecture of the output?**]
* The output units in a softmax group use a non-local non-linearity:
  * z_i input to final layer, Sum_i[z_i] != 1
  * then scale "softmax group(s)" so that they sum to 1: y_i = **e^**(z_i) / Sum_j[**e^**(z_j)] ... (**don't forget the e's!**)
  * **del y_i / del z_i = y_i * (1 - y_i)** ... not trivial to derive b/c of all the terms in the numerator above
* **Cross-entropy**: the *right* cost function to use with softmax
  * The right cost function is the **negative log probability of the right answer**.  [FWC - because the answer is a 1-hot vector with a 1 at the right answer or this [from Sebastian Ruder](http://sebastianruder.com/word-embeddings-softmax/): "have a look at [Karpathy's explanation](http://cs231n.github.io/linear-classify/#softmax-classifier) to gain some more intuitions about the connection between softmax and cross-entropy"]
    * C = -Sum_j[t_j ln(y_j)] ... where t_j == 1 for only one j (note the multiplication of t_j and ln(y_j) not subtraction)
    * C = -log( exp(y_i) / Sum_j[exp(y_j)] )  ... where i is the right answer (this is from Quiz 4, question 1) ... = -y_i + log(Sum_j[exp(y_j)])
  * C has a very big gradient when the target value is 1 and the output is almost zero.
    * A value of 0.000001 is much better than 0.000000001 (for a target value of 1)
    * Effectively, the steepness of dC/dy exactly balances the flatness of dy/dz
      * del C / del z_i = Sum_j[ del C / del y_i * del y_i / del z_i ] = [**y_i - t_i**](http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/)  .... (the chain rule again)

### Lecture 4d: Neuro-probabilistic language models
* Bengio's NN for predicting the next word (see green Staples notebook or pdfs)
* Information that the trigram model fails to use
  * Suppose we have seen the sentence: “the cat got squashed in the garden on friday”
  * This should help us predict words in the sentence: “the dog got flattened in the yard on monday”
  * A trigram model does not understand the similarities between:
    * cat/dog squashed/flattened garden/yard friday/monday
  * To overcome this limitation, we need to use the semantic and syntactic features of previous words to predict the features of the next word.
    * [Using a (lower dimensioned) feature representation also allows for a context that contains many more previous words (e.g. 10).]

### Lecture 4e: Dealing with a large number of possible outputs
* embed words in 2D, 2 approaches (see green Staples notebook or pdfs)
  * NN to predict logit
  * tree-based approach (Minih, Hinton 2009)
* a simpler way to learn feature vectors of words (Collobert,  Weston, 2008)
  * introduction to negative sampling
* eventually apply dimension reduction (t-SNE) to display these vectors in a 2D map
<h3>Week 4 Quiz<h3/>
* (for some reason I can't get numbered lists to work unless they are sublists)
  1. The cross-entropy cost function with an *n*-way softmax unit (a softmax unit with *n* different outputs) is equivalent to: (answer) the cross entropy cost function with n logistic units
    * FWC - reason: b/c softmax is just a scaling of logistic
    * wrong - correct answer "None of the above"
  2. A 2-way softmax unit (a softmax unit with 2 elements) with the cross entropy cost function is equivalent to: (answer) a logistic unit with the cross-entropy cost function
    * FWC - reason: b/c -t log (z) - (1 - t) log (1 - z) is equivalent to 
  3. The output of a neuro-probabilistic language model is a large softmax unit and this creates problems if the vocabulary size is large. Andy claims that the following method solves this problem: At every iteration of training, train the network to predict the current learned feature vector of the target word (instead of using a softmax). Since the embedding dimensionality is typically much smaller than the vocabulary size, we don't have the problem of having many output weights any more. Which of the following are correct? Check all that apply.
    * (check) If we add in extra derivatives that change the feature vector for the target word to be more like what is predicted, it may find a trivial solution in which all words have the same feature vector.
    * (check) The serialized version of the model discussed in the slides is using the current word embedding for the output word, but it's optimizing something different than what Andy is suggesting.
    * (not checked) In theory there's nothing wrong with Andy's idea. However, the number of learnable parameters will be so far reduced that the network no longer has sufficient learning capacity to do the task well.
    * (not checked) Andy is correct: this is equivalent to the serialized version of the model discussed in the lecture.
  4. (a) optimal -> 4; (b) greedy -> 2
  5. No
  6. In the Collobert and Weston model, the problem of learning a feature vector from a sequence of words is turned into a problem of: Learning a binary classifier.
    * FWC - (reason) kws: **dual** (see above for this word also)
  7. not worried - network doesn't care about ordering
    * 7 (take 2) - network loses the location

### Week 5a: Why object recognition is difficult
* Segmentation
* Lighting
* Deformation
* Affordances: Object classes are often defined by how they are used: Chairs are things designed for sitting on so they have a wide variety of physical shapes.
  * FWC - this suggests videos of objects being used might be useful for image recognition
* Viewpoint/transformation
  * Imagine a medical database in which the age of the patient is sometimes labeled incorrectly as the patient's weight - this is called "dimension hopping" which needs to be eliminated before applying ML

### 5c: Convolutional neural networks for hand-written digit recognition
* The replicated feature approach (currently the dominant approach for NNs)
  * Use many different copies of the same feature detector w/ diff positions
    * Could also replicate across scale and orientation (tricky and expensive)
  * Use several different features tyeps, each w/ its own map of replicated features (FWC - each with its own convolution function)
* **Backpropagation with weight constraints**
  * It's easy to modify backprop to incorporate linear constraints btw weights
    * Start with w_1 = w_2, then at every iteration ensure that delta(w_1) = delta(w_2)
  * Compute the gradients as usual, but then modify them so they satisfy constraints
    * set del E / del w_1 (and the same for w_2) to the average of the two partial derivatives
* Invariant knowledge: if a feature is useful in *some* locations during _training_, detectors for that feature will be available in all locations during *testing*.
  * "equivariance in the activities and invariance in the weights"
* Pooling
  * Achieve a small amount of translational invariance at each level by pooling (averaging or taking the max, which is slightly better) four neighboring replicated detectors to give a single output to the next level
  * Problem: after several levels of pooling, we've lost info about precise positioning of things
  * allows us to recognize if the image is a face "but if you want to recognize *whose* face it is" you need precise spatial relationships between high-level parts, which has been lost by CNNs
* Le Net
  * Yann LeCun and his collaborators developed a really good recognizer for handwritten digits by using backpropagation in a feedforward net with:
    * Many hidden layers
    * Many maps of replicated units in each layer.
    * Pooling of the outputs of nearby replicated units.
    * A wide net that can cope with several characters at once even if they overlap.
    * A clever way of training a complete system, not just a recognizer.
  * This net was used for reading ~10% of the checks in North America.
  * Look the impressive demos of LENET at **http://yann.lecun.com**
  * Architecture of Le Net
    * C1 features maps - 6 at 28x28 pixels each, each pixel in one of these maps is computed by applying 3x3 convolution function to original image, but all 3x3 pools are the same, so there are only 9 parameters per map
    * S2 feature maps - "subsampling" == "pooling" to reduce each 28x28 down to 14x14
    * C3 feature maps 16 @ 10x10
    * S4 feature maps 16 @ 5x5
    * C5 layer (i.e. no more feature map, just a straight layer) of 120 nodes
    * F6 layer 84 (fully connected)
    * output 10 (one for each digit, Gaussian(/softmax?) fully connected)
* **Priors and Prejudice**
  * **We can put our prior knowledge about the task into the network** by designing appropriate:
    * Connectivity.
    * Weight constraints.
    * Neuron activation functions
  * This is *less intrusive than hand-designing the features*.
    * But it still prejudices the network towards the particular way of solving the problem that we had in mind.
  * Alternatively, we can use our prior knowledge to create a whole lot more training data.
    * This may require a lot of work, e.g. build a simulator (Hofman&Tresp, 1993)
    * It may make learning take much longer.
    * It allows optimization to discover clever ways of using the multi-layer network that we did not think of.
    * *And we may never fully understand how it does it.*
  * The brute force approach
    * LeNet uses knowledge about the invariances to design:
      * the local connectivity
      * the weight-sharing
      * the pooling.
    * This achieves about 80 errors.
    * This can be reduced to about 40 errors by using many different transformations of the input and other tricks (Ranzato 2008)
    * Ciresan et. al. (2010) **inject knowledge of invariances** by creating a huge amount of *carefully designed* extra training data:
      * For each training image, they produce many new training examples by applying many different transformations.
        * FWC - e.g. risk factors, residual variance, and phantom factors -- allows computation of model sensitivity to assumptions
      * **They can then train a large, deep, dumb net on a GPU without much overfitting**. -- only because they have so much extra training data
      * They achieve about 35 errors.
  * How to detect a significant drop in the error rate?
    * Is 30 out of 10,000 significantly better than 40?
    * Need to look at which errors first model got right but second got wrong and vice versa.
    * McNemar test: uses ratio of model_1_wrong_model_2_right to model_1_right_model_2_wrong
      * if 30 and 40 errors this could break down into [29 shared plus 1 vs. 11] or [15 shared plus 25 vs. 15]
* A neural network for ImageNet (16% vs. 26% for all other participants in the 2012 competition)
  * Alex Krizhevsky (NIPS 2012) developed a very deep CNN with the following architecture:
    * 7 hidden layers ("deeper than usual") not counting some max pooling layers.
    * The early layers were convolutional. "Could probably get away with using local receptive fields without tying any weights, but would need a much bigger computer.  By making them convolutional, you cut down on the number of parameters a lot, and you cut down on the amount of training data a lot."
    * The last 2 layers were globally connected, which is where most parameters are ~16mm between those 2.
      * These 2 layers are looking for combinations of the features extracted by the earlier layers--and obviously there is combinatorially l
    * Activation functions:
      * **Rectified linear units** in every hidden layer, which train much faster and more expressive than logistic units (nobody uses logistic anymore)
      * **Competitive normalization** to suppress hidden activities when nearby units have stronger activities, which helps w/ variations in intensity.
    * Train on random 224x224 patches of 256x256 images to get more data in addition to left-to-right reflections.
      * At test time, combine the opinions of 10 different 224x224 patches: 4 corners + center + 5 reflections
    * Use "**dropout**" to regularize the weights in the globally connected layers (worth several percentage points of improvement)
      * half of hidden units are randomly removed for each training example -> units cannot learn to overly correct for each other (FWC - can't be too co-linear)
      * **prevents overfitting**
  
### Lecture 6a: Overview of mini-­batch gradient descent
* FWC - since the error surface lies in a space composed of pieces of quadratic bowls and the direction of steepest descent is only towards the minimum for perfect circle (cross sections), and for very skinny ellipses it is perpendicular, why not have a **normalization** procedure that attempts to make circles? (also see 'Shifting the inputs' slide in lecture 6b; also see 'separate adaptive learning rates' lecture 6d)
* SGD
  * mini-batches (10, 100, 1000) are usually better than online b/c less computation updating weights
    * mini-batches need to be sampled in a way that they approx. the full distribution to prevent "sloshing" around in the quadratic bowl
  * computing gradient for multiple cases simultaneously uses matrix mult which are very efficient on GPUs
  * turn down the learning rate when the "error" stops decreasing
    * measure the "error" on a separate validation set

### Lecture 6b: Bag of tricks for mini-batch GD
* break symmetry by initializing w/ small random values
* shift inputs - demean each input component to prevent (101,101)->2 and (99,101)->0 (a skewed elliptical error surface)
  * htan = 2 * logistic - 1
* also helps to scale inputs to prevent (0.1,10)->1 and (0.1,-10)->0 (another skewed ellipse)
* more thorough method: decorrelation (guaranteed circles every time)
  * big win for linear neurons
  * e.g. PCA
* starting w/ big learning rate risks weights becoming all very large positive or negative while error derivatives become tiny
  * people usually think they've reached a local minimum, but this usually isn't true, you're just stuck out on the plateau
  * another plateau that looks like a local minumum is in classification nets to use the "best guessing strategy" - just guess 1 with P = the proportion of 1s that are seen in the data (FWC - this is like learning the intercept but nothing else so are there other local minima where say the intercept and one coefficient are learned?)

### Lecture 6c: The momentum (viscosity) model to GD
* if the error surface is a tilted plane the ball rolling down the plane will reach a terminal velocity when the incoming error gradient exactly balances the decay/viscosity/alpha of the previous gradients
* a big learning rate by itself towards the end of learning generates big divergent oscillations across the ravine (sloshing), momentum dampens this sloshing allowing for larger learning rates
* better momentum (Sutskever, 2012 inspired by Nesterov, 1983) - first make a big jump in direction of previously accumulated gradient, then measure gradient again where you end up and make a small correction (sliding scale EAFP)
  * "much better to gamble then make a correction, than to make a correction then gamble"

### Lecture 6c: The momentum (viscosity) model to GD
* add 0.05 if consecutive gradient signs the same, multiply by 0.95 if opposite sign
* can combine adaptive learning w/ momentum: use the agreement in sign between the current gradient for a weight and the velocity for that weight (Jacobs, 1989)
* adaptive learning rates only deal with axis-aligned effects
* momentum doesn't care about alignment of the axes, it can deal with diagonal ellipses (and going in diagonal direction quickly), which adaptive learning cannot do

### Lecture 6e: rmsprop: Divide the gradient by a running average of its recent magnitude
* rmsprop is an extension of rprop to tailor it for mini-batch learning (**Hinton's current favorite model**)
* rprop: Using only the sign of the gradient
  * The magnitude of the gradient can be very different for different weights and can change during learning, which makes it hard to choose a single global learning rate--and adaptive learning rates are tricky too.
    * This escapes from plateaus with tiny 
  * rprop combines the idea of only using the sign of the gradient with the idea of adapting the step size separately for each weight.
    * Increase the step size for a weight *multiplicatively* (e.g. times 1.2) if the signs of its last two gradients agree.
    * O/w decrease multiplicatively (e.g. 0.5)
    * limit step sizes to in [1e-6, 50]
  * rprop doesn't work well (lots of people have tried) for mini-batches b/c weights can grow too much in the presence of consecutive equal sized & signed gradients (e.g. 0.1 0.1 0.1 0.1 0.1 0.1 -0.9).
* rmsprop
  * rprop is equivalent to dividing the gradient by the magnitude of the gradient: g/|g| or g/sqrt(g^2)
    * the problem w/ mini-batch rprop is that we divide by a different # foreach mini-batch
    * so why not fix the # we divide by
  * rmsprop - keep a moving avg of the squared (RMS) gradient for each weight
    * MeanSquare(w,t) = 0.9 MeanSquare(w,t-1) + 0.1 [del E / del w (t)]^2
    * dividing gradient by MeanSquare(w,t) makes learning work much better (Tijmen Tieleman, unpublished)
  * commentary on rmsprop combined w/ momentum (which doesn't seem to help as much)
* **Summary of learning methods for NNs**
  * For small datasets (e.g. 10k cases) or bigger *w/out* much redundancey, use *full-batch*
    * Conjugate gradient, LBFGS (packaged versions, simple for writing papers, no explanation of hyperparameter tweaking necessary)
    * adaptive learning rates, rprop
  * For bit, redundant datasets use *mini-batches*
    1. Try GD w/ momentum
    2. Try **rmsprop** (with momentum?)
    3. Try LeCun's latest recipe (e.g. "No more pesky learning rates" similar to rmsprop)
  * **Why is there no single answer?**
    * lots of different NNs (*esp. ones w/ narrow bottlenecks*)
    * recurrent, wide-shallow (can be optimized with not-very-accurate methods)
    * some require accurate weights, some don't
    * some have *many very rare cases* (e.g. words [FWC - stocks?])

### Quiz 6
  1. WWWwww---
  2. 2 checks
    * too small learning rate
    * large scale inputs (i.e. plateaus at logistic extremes)
    * (unchecked--wrong!) large weight inits -> this will result in large corrective gradients right away (which is wrong, i.e. should've been checked)
  3. circular cloud
  4. the two monotonically descending L shaped red curves that don't get as low as blue (one that is steep at first, crosses the blue, then levels off (correct); the other that just never gets there (wrong)) -- should have selected the one that converges at first, then diverges
  5. 2 check
    * object detection (1e6 training cases, large dataset, but maybe not redundant?)
    * speech recognition (large and redundant)
    * (unchecked) sentiment analysis w/ 100 cases (small dataset)
    * (unchecked) disease prediction (small dataset)

### [7a: Modeling sequences, a brief overview](https://www.coursera.org/learn/neural-networks/lecture/Fpa7y/modeling-sequences-a-brief-overview)
* 2 standard models (not RNNs)
  1. Kalman Filtering (engineers love them!) - efficient recursive way of updating your representation of the hidden state [e.g. covariance matrix] given a new observation
    * given an output, we can't know for sure the hidden state, but we can estimate a Gaussian distribution over the possible hidden states
  2. Hidden Markov Models (computer scientists love them) - have a discrete 1-of-N state, transitions btw states are stochastic and controlled by transition matrix, outputs are produced stochastically as well
    * this is where the term "hidden" layer comes from (coined by Hinton himself)
    * "there's an easy solution, based on dynamic programming, to take the observations we've made and from those, compute the probability distribution across the hidden states"
    * fundamental limitation of HMMs: with only only N states, they can only remember log(N) bits about what has been generated thus far [FWC - they aren't big enough; it's not possible to enumerate enough hidden states]
      * e.g. given 300 syntactic forms, 100k semantic types, and 1k combinations of voice type & intonation => 30e9 hidden states
* RNNs have a more efficient way of storing/representing information (they're also Turing-complete)
  1. distributed hidden states => efficiency
  2. non-linear => updating hidden state in complicated ways
* RNNs are hard to train though because of their computational power

### [7b: Training RNNs with backpropagation](https://www.coursera.org/learn/neural-networks/lecture/vxWDQ/training-rnns-with-back-propagation)
* "backpropagation through time algorithm"
* **feed-forward network with constrained weights*** - just "unroll" the network into a typical feed-forward net (with duplicated/constrained weights)
* remember: it's easy to modify backprop to incorporate linear constraints (compute gradients as usual, then modify them to satisfy the constraints)
* (1) forward pass to build up stack of activities at each time step followed by (2) backward pass to peel activities off the stack to compute error derivatives then (3) average derivatives from all different times to update weights
* Specifying the states of the same subset of the units at every time step is the natural way to model most sequential data.
* Specifying targets:
  1. specify desired final activities of all units
  2. activities for the last few time steps (good for learning *attractors*, i.e. to have the net "settle down" towards the end; it's easy to average in errors as we backpropagate through the final layers)
  3. specify desired activity of a subset of the outputs: natural way to train a network that should be producing "continuous output" (the other units are hidden or input)
* Q: Suppose we're training an RNN on a sequence of numbers.  After it has seen all the numbers in the sequence, we want it to tell us the sum of all the numbers in the sequence.  Which of the following statements are correct?
  * A1: To provide input, we should specify the state of one unit (say unit #1) at every time step.  Reason: There's one input at each time step, the next number in the sequence.
  * A2: We should specify a target for one unit (say unit #2) only at the final time step.  Reason: There's one output value, which occurs only at the last time step.  That's where the model is expected to produce the sum of the numbers in the sequence.

### [7c: A toy example of training an RNN](https://www.coursera.org/learn/neural-networks/lecture/44MXw/a-toy-example-of-training-an-rnn)
* We could train a *feedforward* net to do binary addition, but there are obvious regularities that it cannot capture *efficiently*
  * would have to decide in advance how many digits
  * the processing applied to the beginning of a # wouldn't generalize to the end b/c it'd be using different weights
* "A recurrent network can emulate a finite state automaton, but it is **exponentially more powerful. With N hidden neurons it has 2^N possible binary activity vectors (but only N^2 weights)**"

### [7d: Why it is difficult to train an RNN](https://www.coursera.org/learn/neural-networks/lecture/kTsBP/why-it-is-difficult-to-train-an-rnn)
* exploding and vanishing gradients
* There is a big difference between the forward and backward passes
  * In the forward pass, we use squashing functions (like the logistic) to prevent the activity vectors from exploding [FWC - these squashing functions get applied at every single layer over and over]
  * The backward pass is completely linear (which most people find surprising).  If you double the error derivatives at the final layer, all the error derivatives [at all layers throughout the net] will double.
    * backprop is a linear system which suffer from problem: when you iterate, the gradients explode or 
die
    * if small weights => they shrink exponentially [due to constraint averaging]; if big => they grow exponentially
* Typical feedforward NNs can cope with these exponential effects b/c they only have a few hidden layers.
  * In an RNN trained on long sequence (e.g. 100 time steps) the gradients can easily explode or vanish.
* 4 effective ways to learn an RNN
  1. LSTMs - compose RNN out of little modules designed to remember for long periods
  2. Hessian Free Optimization - a fancy optimizer to detect directions w/ tiny gradient but even smaller curvature (The HF Optimizer, Martens and Sutskever, 2011, is good at this)
  3. Echo State Networds - work around the problem via careful initialization, weakly coupled oscillators
  4. Good init w/ momentum - ESN initialization, but then learn w/ momentum

### [7e: Long Short-Term Memory](https://www.coursera.org/learn/neural-networks/lecture/Nk2p6/long-term-short-term-memory)
* The dynamic state of a RNN is its short term memory, but we want to make the short term memory last for a long time.
* LSTMs have been successful for cursive handwriting recognition (Graves & Schmidhuber, 2009) - input: (x,y,p) coordinates for pen tip (where p is boolean = pen up/down) - output: seq of chars
* "keep"/forget, "write" (cell influences rest of net), and "read" (rest of net influences cell) gates - all logistics (**logistics are used because they have nice derivatives**)

### Quiz 7
  1. 16 logistic hidden units can model 16 bits of information (WRONG, correct answer is ">16 bits")
  2. RNN and FFNN, both w/ 200ms (WRONG, RNN w/ 30ms of input should also be selected)
  3. see (green) notebook, A: -0.355 (CORRECT)
  4. see (green) notebook, A: 0.01392 (CORRECT)
  5. exploding (WRONG, correct "vanishing")
  6. see (green) notebook, A: c (CORRECT)

* take 2 & 3
  1. 16 HMM units -> 16 bits (WRONG, correct answer is "4 bits" for an HMM)
  3. (WRONG) asked this time for T=2 (T=1 was asked the last 2 times)

### 8A: A breif overview of "Hessian-Free" optimization (no video, but lecture slides are in lec8.pdf)
* **Good explanation of why (efficient) optimization involves multiplying by the inverse of a covariance matrix**
  * "maximum error reduction depends on the ratio of the gradient to the curvature, so a good direction to move in is one w/ high ratio of gradient to curvature, even if the gradient itself is small"
    * but if the error surface has circular cross-sections the gradient is fine
    * so apply a linear transformation to turn ellipses into circles
    * **Newton's method** multiplies the gradient vector by the **inverse of the curvature matrix (FWC - the covariance matrix?)**
      * delta(w) = -epsilon * H(w)^-1 * del E / del w
    * on a real quadratic surface this jumps to the minimum in one step
    * there are too many terms in the curvature matrix H(w) to invert it though
    * in the HF method, approximate the curvature matrix, then minimize error using *conjugate gradient*
* **Conjugate gradient**
  * ensure each next direction is "conjugate" to the previous so that you don't oscillate/thrash too much
  * Also see 'non-linear conjugate gradient' for non-quadratic error surfaces (where it still works quite well)

### [8b: Modeling character strings with multiplicative connections](https://www.coursera.org/learn/neural-networks/lecture/qGmdv/modeling-character-strings-with-multiplicative-connections-14-mins)
* The multiplicative factors described in the lecture are an alternative to simply letting the input character choose the hidden-to-hidden weight matrix.
* Modeling text: **Advantages of working with characters**
  * The web is composed of character strings.
  * Any learning method powerful enough to understand the world by reading the web ought to find it trivial to learn which strings make words (this turns out to be true, as we shall see).
  * Pre-processing text to get words is a big hassle
    * What about morphemes (prefixes, suffixes etc)
    * What about subtle effects like “sn” words?  They often have something to do w/ upper lip or nose: snot, snarl, snog.  "Many people say, 'What about snow?' but ask yourself: why is 'snow' such a good word for cocaine?"
    * What about New York?  One lexical item or 2?  "New Yorkminster Roof"?
    * What about Finnish (and [Agglutinative Language](https://en.wikipedia.org/wiki/Agglutinative_language))? This word takes 5 words in English to say the same thing: ymmartamattomyydellansakaan (FWC lots of umlaut's left off this "word")
  * It's a lot easier to predict 86 chars than 100,000 words
  * 2 ways to build a NN
    1. 1500-hidden layer RNN; requires backprop to the beginning of the string
    2. a tree where each node is a hidden state vector (exponentially many nodes), but different nodes can share structure b/c they use distributed representations
      * e.g. if we arrive at a node "fix" the hidden state can encode that this is a verb and that 'i' or 'e' often follow ("fixed" or "fixing"), so <i-next> can operate on <is-a-verb>, which can be shared by all the verbs -- and <n-next> might follow the *conjunction* of <is-a-verb> followed by <i-previous>
* Multiplicative connections
  * Use the current char to choose the whole 1500x1500 hidden-to-hidden weight matrix, but constrain the matrices to be similar for each char by using **factors**!!!
  * 1000 hidden units & 86 character units => 2086 weights (1000 from previous hidden state, 86 from incoming char, plus 1000 outgoing for next hidden state)
    * vector of inputs to group c for factor f: c_f = (b'w_f)(a'u_f)v_f
      * b'w_f : scalar input to f from group b (e.g. 86 dim)
      * a'u_f : scalar input to f from group a (e.g. 1000 dim)
      * v_f : vector of output weights to be scaled by the 2 scalars above (1000 dim)
    * rearrange: c_f = (b'w_f)(u_f * v_f')(a)
      * u_f * v_f' : outer product transition matrix w/ rank 1
      * a : current hidden state gets multiplied to determine the input that factor f gives to next hidden state
   * a can be factored out : C = sum_f[(b'w_f)(u_f * v_f')] * a where the matrix sum multiplied by a is the transition matrix
   * see page 17 of lec8.pdf at file:///home/fred/Documents/articles/geoff_hinton's_machine_learning_coursera/lec8.pdf

### [8c: Learning to predict the next character using HF](https://www.coursera.org/learn/neural-networks/lecture/buNl3/learning-to-predict-the-next-character-using-hf-12-mins)
* Ilya Sutskever used 5 million strings of 100 characters taken from wikipedia. For each string he starts predicting at the 11th character. (**FWC - attempt to recreate this**)
  * best single model for character prediction (combinations of many models do better)
  * start w/ model in default hidden state, give it a "burn-in" (FWC - ramp-in) sequence of chars and let it update its hiddens state after each char
  * See: 'How to generate character strings from the model' slide to see what it "knows" **(FWC - to generate ideas from it)**
    * tell it that whatever char it predicts is correct and let it go on generating **(FWC - this could be used to generate scenarios for monte carlo simulation.  the analogy to character learning might be learning a mean and a stdev (and skew)--the moments of a distribution--rather than different buckets of means which wouldn't have any represented order to them)**
  * Also see: 'Some completions produced by the model' slide
    * "The meaning of life is *literary recognition.*" (6th try)
  * it learns loose semantics, but humans learn these things too
    * If you have to answer this question very quickly what do you answer "What do cows drink?"
* RNNs for predicting next word (as opposed to next char)
  * Tomas Mikolov (word2vec!) has recently trained quite large RNNs on large training sets using BPTT
  * better than feed-forward NNs, better than best other models, and even better when averaged w/ others
  * RNNs **require much less training data to reach the same level of performance**
    * FWC - this is because of their constraints (& factors), the other models have too many degrees of freedom?
  * RNNs also improve faster as the datasets get bigger

### [8d: Echo State Networks](https://www.coursera.org/learn/neural-networks/lecture/s1bdp/echo-state-networks-9-min)
* Big random (fixed) expansion of input vector, then learn the output layer with a linear model.
  * Similar to Support Vector Machines (SVMs) which just do this more efficiently
* Equivalent idea for RNNs is to randomize & fix the input->hidden connections and hidden->hidden connections and just learn the hidden->output
  * Will only work if you set the random connections very carefully so that the RNN doesn't explode or die
  * Set them so that the length (L2 norm) of the activity vector stays about the same length after each iteration aka so that the spectral radius is 1 (the biggest eigenvalue of the hidden->hidden matrix is 1 or it would be 1 if it were a linear system - we want to achieve the same property in a nonlinear system)
  * This allows the input to *echo* around the network for a long time
  * Also important to use sparse connectivity (lots of 0 weights and some big weights rather than lots of medium sized weights) - this creates lots of loosely coupled oscillators (information can hang around in one part of the net and not propagate to other parts too quickly)
  * Choose input->hidden scale very carefully which need to drive the loosely coupled oscillators w/out wiping out the information from the past that they already contain
  * Learning is very fast (fortunately) so we can experiment with the scales of these connections and level of sparseness (hyperparameter tuning)
* Example - predict a sine wave from its frequency
* **Impressive modeling of 1-dimensional time series** very far into the future
  * but aren't very good for high-dimensional data like pre-processed audio or video b/c they need many more hidden units than an RNN that learns its hidden->hidden weights
* Sukskever (2012) used ESN initialization in a normal RNN (with rmsprop and momentum) - very efficient/effective

### Week 8 Quiz
  1. (checked) can use model where input char chooses whole matrix; (unchecked) can't use additive/obvious model (not sufficiently flexible); (unchecked) too many factors b/c simply choosing a multiplicative matrix for each char is at least as flexible as a factor-constrained matrix for each char; (checked WRONG - an additive model can't express a multiplicative one) can use additive with modification b/c the modification (one for each factor) acts like the multiplicative model
  2. 1 - because still selecting a single matrix that connects each hidden->hidden (WRONG - 1 per factor, so the answer is 1000, b/c the hidden->hidden weights are only constrained by the factors in this scenario, not constrained to 1 like in the former scenario)
  3. 3086000 (1500*2 + 86 for each of the 1000 factors) (see lecture notes above) (CORRECT)
  4. It should learn eventually, but that requires more compute power than is available today (WRONG - that would be overfitting) Wrong: Basic calculations about the size of the hidden state vector show that the model can never learn to reliably generate any fixed string of text that's more than 38 chars long (38 is the sqrt of 1500, the number of hidden units) Correct: That would've been overfitting, which was carefully avoided.
  5. No - they aren't at risk of overfitting b/c the hidden->hidden weights are fixed
  6. Don't always use the single most likely char next.  Do sample from the distribution.  A probability distribution is better visualized by samples from it.
  7. **In Echo State Networks, does it matter whether the hidden units are linear or logistic (or some other nonlinearity)? A: Yes. With linear hidden units, the output would become a linear function of the inputs, and we typically want to learn nonlinear functions of the input. Therefore, linear hidden units are a bad choice.**

### [Lecture 9a: Overview of ways to improve generalization](https://www.coursera.org/learn/neural-networks/lecture/R1OLs/overview-of-ways-to-improve-generalization-12-min)
* Network capacity can be controlled in several ways
  1. Architecture - limit # of hidden neurons
  2. Early stopping - assumes real relationships are learned before spurious ones; start w/ very small weights (important!) and stop training when (you're sure) validation error performance starts getting worse (and then go back to when things were best)
  3. Weight-decay - penalize large weights
  4. Noise - add noise to weights or activities
* Cross-validation - a better way to choose meta parameters
  * 3 sets of data: training, validation, test (only used once)
* Why early stopping works
  * A network with small weights has lower capacity, but why? [FWC - this is kinda like degrees of freedom.  Is there a way to combine the number of weights with their sizes to come up with a DF metric?]
  * When weights are small, if the hidden units are logistic units, they are in the linear range of the logistic function (slope = 0.25), so they behave very similarly to linear units.
  * So a small weight logistic network has no more capacity than a linear net.
  * [FWC - so then there is another point on the logistic curve where it is quadratic, cubic, etc.  The average (and stdev) across all units of this metric might be a good measure of DF.  [Sigmoid/Logistic Power Series](https://www.quora.com/Can-the-sigmoid-function-be-express-as-a-power-series)]
    * [FWC - we limit the weights when using other statistical approaches as well, e.g. SVD variants, but this seems to be a more intuitive reason for doing so.]

### [Lecture 9b: Limiting the size of the weights](https://www.coursera.org/learn/neural-networks/lecture/CD4PO/limiting-the-size-of-the-weights-6-min)
* Standard approach is to add a (constant times the) sum of the squared weights (L2) into the cost function--a penalty.
  * Large weights to occur only when there are large error derivatives (see slides)
  * Using L1 (absolute value) penalty is sometimes better b/c lots of weights are then 0, which makes a network easier to understand.  Or even more extreme, sqrt(abs), makes escape from 0 difficult but then negligible effect on really large weights (allows for a few big weights).
* **Better idea: constrain the squared length of the weight vector**
  * If an update violates the constraints, then scale down the weight vector to the allowed length [FWC - this could be used for across-the-board constraints, rather than penalties, on risk factor exposures]
  * This is much more effective at pushing irrelevant weights towards 0 b/c big weights, via the scaling, cause the small weights to get smaller.  "The penalties are then just the LaGrange multipliers required to keep the constraints satisfied."
  * Hinton finds such **constraints to work "noticeably" better than penalties**
  * FWC - this is similar to mean-variance optimization with flat-bottomed parabolas
  * It's *much easier to set a sensible constraint than a sensible weight penalty*

### [Lecture 9c: Using noise as a regularizer](https://www.coursera.org/learn/neural-networks/lecture/wbw7b/using-noise-as-a-regularizer-7-min)
* L2-weight penalty is equivalent to adding noise to the inputs (in a *linear* network)
  * Minimizing the squared error also minimizes the squared weights (because the noise *variance* gets scaled by the square of the weights), the second operand of this sum: y_j + N(0, w_i^2 * sigma_i^2)
  * E[(y - sum_i(w_i*e_i) - t)^2] = (y - t)^2 + sum_i(w_i^2 * sigma_i^2)
* Using noise in the activities as a regularizer
  * Make the units binary and stochastic on the forward pass, but then do the backward pass as if we had done the forward pass "properly"
    * In the forward pass choose 0 or 1 based on the "probabiliy" value of the logistic function.
  * This does worse on the training set (and trains considerably slower), but performance on the test set is significantly better (unpublished result)
    * [FWC - **so train with positive or negative returns, -1 or 1, but then compute cost (and propagate errors) with real-return-computed cost** -- and also note that not just tcosts, but liq constraints also, and all other *phantom* constraints will get baked into the learned function]
* [FWC - So adding noise to inputs, weights, and activations reduces overfitting (adds regularization), but what's the real mechanism for why?  Does the added noise effectively dampen actual noise, making it more difficult to overfit?  But then if you add too much noise, does it make real effects difficult to detect?  Seems like you could slowly increase the amount of noise and measure validation performance along the way]

### [Lecture 9d: Introduction to the Bayesian Approach](https://www.coursera.org/learn/neural-networks/lecture/nPahR/introduction-to-the-full-bayesian-approach-12-min)
* "The main idea behind the Bayesian Approach is that instead of looking for the most likely setting of the parameters of a model, we should consider all possible settings of the parameters and try to figure out for each of those possible settings how probable [FWC - likely] it is, given the data we observed."
* POSTERIOR = PRIOR * DATA_LIKELIHOOD
  * With enough data, the likelihood term always "wins."
* If we flip a coin 100 times and see 53 heads then what is P(head)?
  * Frequentist answer (aka maximum likelihood) = 0.53
    * Set derivative d/dp of P(D) where P(D) = p^53 * (1-p)^47 to 0 (which is where p is *maximized*) => p = 0.53
    * Problems w/ MLE
      * If we only flip the coin once and get a head, then p = 1?  No, 0.5 is still a better answer.
      * It's unreasonable to give a single answer.  Instead lets say we're unsure about p.
  * See slide 24 (of 39) of lec9.pdf 'Using a distribution over parameter values' for how the posterior is updated after one coin flip given a uniform prior.
* Bayes
  * P(W|D) = P(W) * P(D|W) / P(D)
  * P(D) is a normalizing term (to ensure the distribution integral sums to 1) equal to integral_W[P(W) * P(D|W)] but for any P(W|D) in the equation above this is the same value--it doesn't depend on W b/c it's an integral over all possible Ws

### [The Bayesian interpresation of weight decay/penalties](https://www.coursera.org/learn/neural-networks/lecture/n6TUy/the-bayesian-interpretation-of-weight-decay-11-min)
* Maximum a Posteriori Learning
  * Gives us a nice explanation of what's really going on when we use weight decay to control the capacity of a model
* Supervised Maximum Likelihood Learning
  * Finding the weight vector that minimizes the squared residuals is equivalent to finding a weight vector that *maximizes the log probability density of the correct answer*
  * see slide 30 of lec9.pdf for mathematical derivation
    * -log(P(t|y)) = k + (t-y)^2 / (2 * sigma^2)
      * note this is where the 2*sigma^2 in the denominator comes from
    * if -log(P(t|y)) is our cost function, this turns into minimizing a squared distance (the RHS)
    * *Minimizing squared error is the same as maximizing log probability under Gaussian distribution!* ("helpful to know" b/c when you're minimizing sq error you can make a probabilistic interpretation of what's going on)
* MAP: Maximum a Posteriori
  * P(W|D) = P(W) * P(D|W) / P(D)
  * Cost = -log P(W|D) = -log P(W) - log P(D|W) + log P(D)
    * log P(D) doesn't depend on W, so doesn't affect the minimization
    * log P(D|W) is the log prob of target values given weights, W (normalized by 2*sigma_D^2 of the errors, the data)
    * log P(W) is the log prob of W under the prior (normalized by 2*sigma_W^2 of the weights)
  * Minimizing the squared weights is equivalent to maximizing the log probability of the weights under a *zero-mean* Gaussian prior (the same as for minimizing the squared error!)
  * multiply through by 2*sigma_D^2 gives us:
    * Cost = MSE + sigma_D^2 / sigma_W^2 * sum_i[w_i]
    * so we have a specific number for the weight penalty, the ratio of 2 variances (FWC - like a [beta](https://en.wikipedia.org/wiki/Beta_(finance))!), it's not an arbitrary choice as in previous lecture

### [Lecture 9f: MacKay’s quick and dirty method of fixing weight costs](https://www.coursera.org/learn/neural-networks/lecture/7Q9LC/mackays-quick-and-dirty-method-of-setting-weight-costs-4-min)
* Estimating the variance of the Gaussian prior on the weights
  * After learning a model with some initial choice of variance for the weight prior, sigma_W^2, we could do a dirty trick called "empirical Bayes"
  * Set the variance of the Gaussian prior, sigma_W^2, to be whatever makes the weights that the model learned most likely.
    * i.e. use the data itself to decide what your prior is!
    * "This really violates a lot of the presuppositions of the Bayesian approach.  We're using our *data* to decide what our beliefs are."
    * Fit a zero-mean Gaussian to the 1-dimensional distribution of the learned weight vals
    * => **we could easily learn different variances for diff sets of wgts (a benefit!)**
  * We don't need a validation set!  which, to use it, would be very time consuming
* MacKay’s quick and dirty method of choosing the ratio of the noise variance, sigma_D^2, to the weight prior variance, sigma_W^2
  * Start with guesses for both the noise variance and the weight prior variance (really just guess their ratio)
  * Loop/repeat:
    1. Do some learning using the ratio of the variances as the weight penalty coefficient
    2. Reset the noise variance to be the variance of the residual errors
    3. Reset the weight prior variance to be the variance of the distribution of the actual learned weights
  * **This works quite well in practice and MacKay won several competitions this way.**
* Q: Suppose we're using MacKay's method for setting weight costs.  For one particular input unit, we decide to use the same prior variance for all of its outgoing weights to the hidden layer.  Now suppose that the state of that input unit does not contain any useful information for getting the output right.  Assuming we start with weights drawn from the prior distribution, which of the following statements is true?
  * FALSE (b/c weights are known to be 0 so variance shrinks) - After training the weights for a while without updating the weight prior variance, we expect the actual variance of the outgoing weights of that unit to be bigger than the prior variance.
  * TRUE - Every time we update the prior variance after doing some more learning, it will get smaller.
    * correct - Since that input unit contains no useful information, we expect the *error derivative to be small.  The Gaussian weight prior will always push the weights towards 0 unless it is opposed by the error derivative.*

### Week 9 Quiz
  1. the one where the training error reaches 0 is an overfit (not checked)
  2. INCORRECT - "adding weight noise" is equivalent to "L2 regularization" so it's not either of those.  it must be "L1 regularization" b/c something was used, though it should really be that a constraint on the size of the weight vector was used because a few large values happened
  3. selected the "mustache" shape in order to have lot of weights close to 0 but a few a long way from 0
  4. E = SSE, C = student-t cost = lambda/2 * sum_i[log(1 + w_i^2_], E_tot = E + C, what is d/dw E_tot
    * A: lambda * w_i in the numerator of the C derivative
  5. Different regularization methods have different effects on the learning process. For example (a) L2 regularization penalizes high weight values. (b) L1 regularization penalizes weight values that do not equal zero. Adding noise (c) to the weights during learning ensures that the learned hidden representations take extreme values. Sampling (d) the hidden representations regularizes the network by pushing the hidden representation to be binary during the forward pass which limits the modeling capacity of the network.
    * Q: Given the shown [bimodal at -10 and 10] histogram of activations (just before the nonlinear logistic nonlinearity) for a Neural Network, what is the regularization method that has been used (check all that apply)?
    * checked - 'adding noise to the weights' and 'sampling the hidden repr'
    * unchecked - L1 and L2
  6. INCORRECT - better on both training and test (the correct answer is worse on training and better on test)

### [[Week 9 programming assignment #3]]
* "The program checks your gradient computation for you, using a finite difference approximation to the gradient. If that finite difference approximation results in an approximate gradient that's very different from what your gradient computation procedure produced, then the program prints an error message. This is hugely helpful debugging information. Imagine that you have the gradient computation done wrong, but you don't have such a sanity check: your optimization would probably fail in many weird and wonderful ways, and you'd be worrying that perhaps you picked a bad learning rate or so. (In fact, that's exactly what happened to me when I was preparing this assignment, before I had the **gradient checker** working.) With a finite difference gradient checker, at least you'll know that you probably got the gradient right. It's all approximate, so the checker can never know for sure that you did it right, but if your gradient computation is seriously wrong, the checker will probably notice."

### [Lecture 10a: Why it helps to combine models](https://www.coursera.org/learn/neural-networks/lecture/pZKOF/why-it-helps-to-combine-models-13-min)
* Also see XGBoost notes on Machine Learning wiki page
* "If we have a single model we have to choose some capacity for it.  If we choose too little capacity, it won't be able to fit the regularities in the training data.  If we choose too much capacity, it will be able to fit the sampling error in the training set data.  By averaging many models we can get a better tradeoff between fitting too few regularities and overfitting the sampling error in the data.  This effect is largest when the models make very different predictions from each other."
* Combining networks: the bias-variance tradeoff
  * When the amount of training data is limited, we get overfitting.
    * Averaging the predictions of many different models is a good way to reduce overfitting.
    * It helps most when the models make very different predictions.
  * For regression, the squared error can be decomposed into a "bias" term and a "variance" term.
    * The bias term is big if the model has too little capacity to fit the data.
    * The variance term is big if the model has so much capacity that it is good at fitting the sampling error in each particular training set.
      * It's called "variance" because if we were to get another training set of the same size from our distribution, our model would fit differently to that training set because it has different sampling error, so we'll get **variance in the way the model's fit to different training sets.**
    * By averaging away the variance we can use individual models with high capacity. These models have high variance but low bias.
    * We can get low bias without getting high **variance [FWC - overfitting]** by using averaging to get rid of the high variance.
* How the combined predictor compares with the individual predictors
  * On any one test case, some individual predictors may be better than the combined predictor.
    * But different individual predictors will be better on different cases.
  * If the individual predictors *disagree* a lot, the combined predictor is typically better than all of the individual predictors when we average over test cases.
    * So we should try to make the individual predictors disagree (without making them much worse individually) [FWC - make individual predictors them uncorrelated with each other]
* Combining two networks reduces variance
  * The expected squared error we get, by picking a [single] model at random, is greater than the squared error we get by averaging the models by [exactly] the **variance of the outputs of the models**
  * E_i[(t-y_i)^2] = (t-E[y])^2 + **E[(y-E[y])^2]** - 2(t-E[y])(y-E[y])  [<- this last term vanishes because we expect the errors to be uncorrelated]
   * This only works if the noise is Gaussian
     * Don't try averaging a bunch of clocks because some may be approximately correct, but some may have stopped and be wildly wrong [skew and kurtosis matter]
     * Same applies to discrete distributions over class labels
       * if we have 2 models, i and j, that give the correct label probabilities, p_i and p_j
       * log((p_i+p_j)/2) >= (log(p_i)+log(p_j))/2 [think of the plot of the log function]
* Lots of ways to make predictors/models differ
* Making models differ by changing their training data
  * Bagging - Train diff models on diff subsets of the data (with replacement) ["bootstrapping"?]
  * Random forests - Use lots of different decision trees trained using bagging.  They work well.
  * We could use bagging w/ NNs but its *very* expensive to train
  * Boosting - Train a seq of low capacity models.  Weight the training cases differently for each model in the seq.  Up-weight cases the previous models got wrong.  [FWC - prone to overfitting]

### [Lecture 10b: Mixture of Experts](https://www.coursera.org/learn/neural-networks/lecture/stzor/mixtures-of-experts-13-min)
* **Multi-regime**, train a net on each regime
* In boosting, the mixture of models depends on particular input cases.
* Can we do better that just averaging models in a way that does *not* depend on the particular training case?
  * Maybe we can look at the input data for a particular case to help us decide which model to rely on.
  * This may allow particular models to *specialize in a subset of the training cases* [FWC - **regimes**]
  * They do not learn on cases for which they are not picked. So they can ignore stuff they are not good at modeling. Hurray for nerds!
  * The key idea is to make each expert focus on predicting the right answer for the cases where it is already doing better than the other experts.  This causes specialization.  [FWC - this is similar to k-means clustering where the clusters drift away from each other over time]
* Spectrum of models
  * Very local models (e.g. nearest neighbors)
    * very fast to fit (e.g. just store training cases)
  * Fully global models (e.g. polynomial)
    * may be slow to fit and also unstable (each param depends on all the data, so small changes to data can cause big changes to the fit)
* Multiple local models
  * Instead of using a single global model or lots of very local models, use several models of intermediate complexity.
    * Good if the dataset contains several different regimes which have different relationships between input and output.
    * e.g. financial data which depends on the state of the economy. **"But we might not know in advance what defines 'different states of the economy'--we'll have to learn that too."**
    * So how do we partition the data into regimes?
* **Partitioning based on input alone versus partitioning based on the input-output relationship**
  * We need to cluster the training cases into subsets, one for each local model.
  * The aim of the clustering is NOT to find clusters of similar input vectors.
  * We want each cluster to have a relationship between input and output that can be well-modeled by one local model.
* An error function that encourages *cooperation*
  * Loss = (t-E[y_i])^2
  * This will overfit badly because models will learn to "fix up" errors that other models make.
  * Averaging models *during training* causes cooperation, not specialization
  * If there's a model that disagrees with the average (e.g. underestimates the target while the average overestimates) then do we really want to move the disagreeing model away from the target in order to "compensate" for all of the other models (to make the average model closer to the target)?  Intuitively it seems better to move the disagreeing model towards the target.
* An error function that encourages *specialization*
  * If we want to encourage specialization we compare each predictor/model separately with the target.
  * We also use a "manager" to determine the probability of picking each expert (aka weight each model)
  * Loss = E[ p_i (t-E[y_i)^2 ]
  * Most experts/models/predictors end up ignoring most targets (due to fitted p_i)
* The mixture of experts architecture (*almost*)
  * The obvious/intuitive architecture: p_i and y_i as outputs
    * The p_i can be chosen with their very own sub-network, the manager, a *gating* network
  * p_i = exp(x_i) / sum_j[exp(x_j)]
  * Loss = E[ p_i (t-E[y_i)^2 ]
  * del Loss / del y_i = p_i (t-E[y_i)
  * If the manager, p_i, decides that there's a small probability of picking that expert for that training case, we get a very small gradient (and parameters inside that expert won't get disturbed)
  * *We want to increase p_i for all experts that give below average (weighted by p_i) squared error across all i* [see del Loss / del x_i below]
  * If we differentiate w.r.t. the outputs
of the gating network (wrt the input to the softmax, which is called the "logit") we get a signal for training the gating net.
    * del Loss / del x_i = p_i ((t-y_i)^2 - Loss)
      * This gradient will increase p_i for all experts that produce below avg SE
      * This is what causes specialization.
  * There's a better cost fn though.
* **A better cost function for mixtures of experts (Jacobs, Jordan, Nowlan & Hinton, 1991)**
  * Think of each expert as making a prediction that is a Gaussian distribution around its output (with variance 1)
  * Think of the manager as deciding on a scale for each of these Gaussians. The scale is called a "mixing proportion". e.g {0.4 0.6}
  * Maximize the log probability of the target value under this mixture [FWC - sum] of Gaussians model (i.e. the sum of the two scaled Gaussians).
  * \<P of target val on case c given Mixture of Experts\> = 
    * sum_i[ \<mixing proportion i for case c, p_ic\> exp(-0.5(t_c-y_ic)^2) / \<normalization term for Gaussian w/ sig^2=1, sqrt(2pi)\> ]
  * P(t_c | MoE) = sum_i[ p_ic exp(-0.5(t_c-y_ic)^2) / sqrt(2pi) ]

### [Lecture 10c: The idea of full Bayesian learning](https://www.coursera.org/learn/neural-networks/lecture/9MEsM/the-idea-of-full-bayesian-learning-7-min)
* Full Bayesian Learning
  * Instead of trying to find the best single setting of the parameters (as in Maximum Likelihood or MAP) compute the full posterior distribution over all possible parameter settings.
    * This is extremely computationally intensive for all but the simplest models (its feasible for a biased coin, as shown in a previous lecture).
  * To make predictions, let each different setting of the parameters make its own prediction and then combine all these predictions by weighting each of them by the posterior probability of that setting of
the parameters.
    * This is also very computationally intensive.
  * The full Bayesian approach allows us to use complicated models even when we do not have much data.
* Overfitting: A frequentist illusion?
  * **A frequentist would say: If you do not have much data, you should use a simple model, because a complex one will overfit.** [FWC - financial people are all frequentists]
    * This is true.
    * But **only if you assume that fitting a model means choosing a single best setting of the parameters.**
  * **If you use the full posterior distribution over parameter settings, overfitting disappears.**
    * When there is very little data, you get very vague predictions because many different parameters
settings have significant posterior probability.
  * Bayesian learning means learning the posterior distribution: **P(omega | data)**.  The kind of learning we saw earlier in this course is called Maximum Liklihood (ML) learning where we learn a set of parameters, omega, that maximizes P(data | omega)
* A classic example of overfitting
  * Which model do you believe, a line or a 5th order polynomial?  Clearly the line b/c the complicated model, even though it fits the training data better, is not economical and it makes silly predictions.
  * But what if we start with a reasonable prior over all fifth-order polynomials and use the full posterior distribution.
  * Now we get vague and sensible predictions.
  * From a Bayesian perspective, **there is no reason why the amount of data should influence our prior beliefs about the complexity of the model.** [FWC - there is no reason why a low signal-to-noise ratio should influence our prior beliefs...etc...]
* Approximating full Bayesian learning in a neural net
  * If the neural net only has a few parameters we could put a grid over the parameter space and evaluate p( W | D ) at each grid-point.
    * This is expensive, but it does not involve any gradient descent and there are no local optimum issues.  We're not following a path, we're only evaluating points.
  * After evaluating each grid point we use all of them to make predictions on *test data*
    * This is also expensive, but it works much better than Maximum Liklihood learning (or MAP) when the posterior is vague or multimodal (this happens when data is scarce).
  * [FWC - This seems to be a dual/transpose of a grid over the input space with a distribution for each input.  Both approaches yield a (or can be used to estimate the) distribution of outputs.  Could the two approaches be combined?  A simultaneous co-grid over both input and parameter space.  But how would one use that?]
  * p(t_test | input_test) = sum_{g in grid}[ p(W_g | D) p(t_test | input_test, W_g) ]
* An example of full Bayesian learning
  * A neural net with 2 inputs, 1 output and 6 parameters
  * Allow each of the 6 weights or biases to have the 9 possible values => 9^6 grid-points (lots but not impossible [FWC - but yeah, lots!])
  * For each grid point, compute the P of the observed outputs over all *training cases*
  * Multiply prior for each grid point by the liklihood term and renormalize to get the posterior P for each
  * Make predictions (on test data) by using the *posterior Ps* to average the predictions made by the different grid points

### [Lecture 10d: Making full Bayesian learning practical](https://www.coursera.org/learn/neural-networks/lecture/PcT3Q/making-full-bayesian-learning-practical-7-min)
* Markov Chain Monte Carlo biased in the direction of the gradient
* Sample weight vectors in proportion to their probability in the posterior distribution
* What can we do if there are too many parameters for a grid?
  * Only a tiny fraction of the grid points make a significant contribution to the predictions--lots of 0 posterior Ps.
  * An idea that makes Bayesian learning feasible: It might be good enough to just sample weight vectors according to their posterior probabilities.
  * \<P output_test given D\> = sum_{i in weight space}[ \<posterior P_i\> \<probability distribution of output_test given W_i\>]
  * p(y_test | input_test, D) = sum_i[ p(W_i | D) p(y_test | input_test, W_i) ]
  * Instead of adding up all terms in this sum, sample weight vectors according to p(W_i | D)
* Sampling weight vectors
  * In standard backpropagation we keep moving the weights in the direction that decreases the cost.
  * Eventually, the weights settle into a local minimum (or get stuck on plateau, or just move so slowly we run out of patience)
* One method for sampling weight vectors
  * Suppose we add some Gaussian noise to the weight vector after each update.
  * So the weight vector never settles down.
  * It keeps wandering around, but it tends to prefer low cost regions of the weight space.
  * Can we say anything about how often it will visit each possible setting of the weights?
  * Save the weights after every 10,000 steps.
* The wonderful property of Markov Chain Monte Carlo
  * *Amazing fact:* If we use just the right amount of noise, and if we let the weight vector wander around for long enough before we take a sample, we will get an unbiased sample from the true posterior over weight vectors.
  * This is called a "**Markov Chain Monte Carlo**" (MCMC) method.
  * **MCMC makes it feasible to use full Bayesian learning with thousands of parameters.**
  * L'orangian (sp?) Method, not the most efficient
  * There are related MCMC methods that are more complicated but more efficient: We don’t need to let the weights wander around for so long before we get samples from the posterior.
* Full Bayesian learning with mini-batches
  * If we compute the gradient of the cost function on a random mini-batch we will get an unbiased estimate with sampling noise.
  * Maybe we can **use the sampling noise to provide the noise that an MCMC method needs!** (very clever) 
  * Ahn, Korattikara & Welling (ICML 2012) showed how to do this fairly efficiently.
  * So full Bayesian learning is now possible with lots of parameters.

### [Lecture 10e: Dropout: an efficient way to combine neural nets](https://www.coursera.org/learn/neural-networks/lecture/Sc5AW/dropout-9-min)
* Method of combining a very large number of NN models w/out having to train them all.
* Two ways to average models: (1) mixture (arithmetic mean) and (2) product (geometric mean renormalized to sum to 1)
* **Dropout**: An efficient way to average many large neural nets (http://arxiv.org/abs/1207.0580)
  * **An alternative to doing the correct Bayesian thing**.  Probably doens't work quite as well, but much more practical [FWC - assuming we're starting with NNs to begin with]
  * Consider a neural net with one hidden layer.
  * Each time we present a training example, we randomly omit each hidden unit with probability 0.5.
  * So we are randomly sampling from 2^H different architectures. All architectures share weights.
* Dropout as a form of model averaging
  * We sample from 2^H models. So *only a few of the models ever get trained, and they only get one training example*.  This is as *extreme as bagging can get*.
  * The sharing of the weights means that every model is very strongly regularized.
    * It’s a *much better regularizer than L2 or L1 penalties that pull the weights towards zero*.
* But what do we do at test time?
  * We could sample many different architectures and take the geometric mean of their output distributions.
  * It better to use all of the hidden units, but to *halve their outgoing weights.*
  * This exactly computes the geometric mean of the predictions of all 2^H models (provided we're using a softmax output group).
* What if we have more hidden layers?
  * Use dropout of 0.5 in every layer.
  * At test time, use the "mean net" that has all the outgoing weights halved.
  * This is not exactly the same as averaging all the separate
dropped out models (in a multi-layer net), but it’s a pretty good approximation, and it's fast.
  * Alternatively, run the (lots of) stochastic model several times on the same input (with dropout and then average across those stochastic models). [FWC - does he mean this is similar to "sampling many different architectures and taking the geometric mean" as mentioned above?]
    * **Benefit: This gives us an idea of the uncertainty in the answer.**
* What about the input layer?
  * It helps to use dropout there too, but with a higher probability of keeping an input unit.
  * This trick is already used by the "**denoising autoencoders**" developed by Pascal Vincent, Hugo Larochelle and Yoshua Bengio "and it works very well."