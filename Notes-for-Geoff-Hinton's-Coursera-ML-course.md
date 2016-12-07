Octave source: https://ftp.gnu.org/gnu/octave/

Also see accompanying green Staples notebook.

### Week 2: Types of Neural Network Architectures
* Ilya Sutskever (2011) trained special type or RNN to predict next char in sequence
  * it generates text by predicting the probability distribution for the next char, not the highest likely next char (which would generate text like "the united states of the united states of the..."), and then sampling from that distribution
* symmetric nets are much easier to analyze than RNNs (John Hopfield)
  * but they're more restricted, e.g. can't learn cycles

<h3>Week 2: What perceptrons can't do</h3>
* A perceptron cannot recognize patterns under translation if we allow wraparound.
  * E.g. 2 binary input vectors, A and B, each with 4 out of 10 activated "pixels."  Each of the 10 pixels will be activated by 4 translations of A and of B, so the total input received by the decision unit over all these patterns, in both cases, will be four times the sum of all the weights.  But to discriminate correctly, every single case of pattern A must provide more input to the decision unit than every single case of pattern B.
  * However, if we have 3 patterns assigned to two classes, A and B, and A contains a pattern with 4 pixels while B contains patterns with 1 and 3 pixels then a *binary* decision unit can classify if we allow translations and wraparound.  Weight = 1 from each pixel with bias of -3.5
  * Minskey and Papert's "Group Invariance Theorem" says that the part of a perceptron that learns cannot do this if the transformations form a group (e.g. translations with wraparound).
  * This result is devastating for pattern recognition (PR) because the whole point of PR is to recognize patterns that undergo transformations, like translation.
  * To deal with such transformations, a perceptron needs to use multiple feature units to recognize transformations of informative sub-patterns.
  * So the tricky part of PR must be solved by the hand-coded feature detectors, not the learning procedure.
  * Networks without hidden units are very limited in what they can learn to model.

<h3>Week 3: Learning the weights of a linear neuron</h3>
* Instead of showing the weights get closer to a good set of weights (i.e. perceptrons, which suffer from 2 good set of weights do not average to a good set) show that actual output values get closer to the target values.
  * In perceptron learning the outputs can get farther away from the targets, even though the weights are getting closer.
* The "delta rule" for learning: delta w_i = epsilon * x_i * (t - y) ... where epsilon := "learning rate", t := target/true output, and y := estimated output
  * Derivation: Error = 0.5 * Sum_{n in training}[(t_n - y_n)^2] ... the 1/2 is only there to make the 2 in the derivative cancel
  * Differentiate the error wrt one of the weights, w_i: del E / del w_i = 0.5 * Sum_n[del y_n / del w_i * d E_n / d y_n] ... Chain Rule ("easy to remember, just cancel those 2 del y_ns ... but only when there aren't any mathematicians looking) ... = -Sum_n[x_{i,n} * (t_n - y_n)]
    * del y_n / del w_i = x_{i,n} because y_n = w_i * x_{i,n}
    * d E_n / d y_n is just the derivative of the (squared) Error function
  * Therefore: delta w_i = -epsilon * del E / del w_i

<h3>Week 3: The error surface for a linear neuron</h3>
* Difference between "batch" and "on-line"
  * Simplest batch learning does steepest gradient descent
  * On-line/stochastic zig-zags between training case "lines" at each step moving perpendicularly to a line.  Imagine the intersection of 2 training case lines and moving perpendicularly back and forth perpendicularly to both while converging on their intersection point.
    * This is very slow if the variables are highly correlated (very elongated ellipse) because the perpendicular updates (the gradients) are also perpendicular to the intersection.
    * FWC idea - look at angle between consecutive gradients to detect correlated dimensions

<h3>Week 3: The backpropagation algorithm</h3>
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

<h3>Week 3: Using the derivatives computed by backprop</h3>
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

<h3>Week 4: Neural nets for machine learning</h3>
* Obvious way to express regularities is as symbolic **rules**
  * but finding the symbolic rules involves a difficult search through a large discreet space
  * so model as a NN instead w/
    * input := person1 + relationship (both 1-hot encodings)
    * output := person2 (also 1-hot)
* **Instead of predicting the 3rd term in a relationship, [A R B], we could provide all 3 as input and predict P([A R B] is correct)**
  * for this we'd need a whole bunch of "correct" facts as well as "incorrect" ones (fwc - **negative sampling**)

<h3>4b: A brief diversion into cognitive science</h3>
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

<h3>4c: The softmax output function</h3>
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
  * The right cost function is the **negative log probability of the right answer**.  [FWC - because the answer is a 1-hot vector with a 1 at the right answer]
    * C = -Sum_j[t_j ln(y_j)] ... where t_j == 1 for only one j (note the multiplication of t_j and ln(y_j) not subtraction)
    * C = -log( exp(y_i) / Sum_j[exp(y_j)] )  ... where i is the right answer (this is from Quiz 4, question 1) ... = -y_i + log(Sum_j[exp(y_j)])
  * C has a very big gradient when the target value is 1 and the output is almost zero.
    * A value of 0.000001 is much better than 0.000000001 (for a target value of 1)
    * Effectively, the steepness of dC/dy exactly balances the flatness of dy/dz
      * del C / del z_i = Sum_j[ del C / del y_i * del y_i / del z_i ] = y_i - t_i  .... (the chain rule again)

<h3>Lecture 4d: Neuro-probabilistic language models</h3>
* Bengio's NN for predicting the next word (see green Staples notebook or pdfs)
* Information that the trigram model fails to use
  * Suppose we have seen the sentence: “the cat got squashed in the garden on friday”
  * This should help us predict words in the sentence: “the dog got flattened in the yard on monday”
  * A trigram model does not understand the similarities between:
    * cat/dog squashed/flattened garden/yard friday/monday
  * To overcome this limitation, we need to use the semantic and syntactic features of previous words to predict the features of the next word.
    * [Using a (lower dimensioned) feature representation also allows for a context that contains many more previous words (e.g. 10).]

<h3>Lecture 4e: Dealing with a large number of possible outputs</h3>
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

<h3>Week 5a: Why object recognition is difficult<h3/>
* Segmentation
* Lighting
* Deformation
* Affordances: Object classes are often defined by how they are used: Chairs are things designed for sitting on so they have a wide variety of physical shapes.
  * FWC - this suggests videos of objects being used might be useful for image recognition
* Viewpoint/transformation
  * Imagine a medical database in which the age of the patient is sometimes labeled incorrectly as the patient's weight - this is called "dimension hopping" which needs to be eliminated before applying ML

<h3>5c: Convolutional neural networks for hand-written digit recognition<h3/>
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
  
<h3>Lecture 6a: Overview of mini-­batch gradient descent<h3/>
* FWC - since the error surface lies in a space composed of pieces of quadratic bowls and the direction of steepest descent is only towards the minimum for perfect circle (cross sections), and for very skinny ellipses it is perpendicular, why not have a **normalization** procedure that attempts to make circles? (also see 'Shifting the inputs' slide in lecture 6b; also see 'separate adaptive learning rates' lecture 6d)
* SGD
  * mini-batches (10, 100, 1000) are usually better than online b/c less computation updating weights
    * mini-batches need to be sampled in a way that they approx. the full distribution to prevent "sloshing" around in the quadratic bowl
  * computing gradient for multiple cases simultaneously uses matrix mult which are very efficient on GPUs
  * turn down the learning rate when the "error" stops decreasing
    * measure the "error" on a separate validation set

<h3>Lecture 6b: Bag of tricks for mini-batch GD<h3/>
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

<h3>Lecture 6c: The momentum (viscosity) model to GD<h3/>
* if the error surface is a tilted plane the ball rolling down the plane will reach a terminal velocity when the incoming error gradient exactly balances the decay/viscosity/alpha of the previous gradients
* a big learning rate by itself towards the end of learning generates big divergent oscillations across the ravine (sloshing), momentum dampens this sloshing allowing for larger learning rates
* better momentum (Sutskever, 2012 inspired by Nesterov, 1983) - first make a big jump in direction of previously accumulated gradient, then measure gradient again where you end up and make a small correction (sliding scale EAFP)
  * "much better to gamble then make a correction, than to make a correction then gamble"

<h3>Lecture 6c: The momentum (viscosity) model to GD<h3/>
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
  * start w/ model in default hidden state, give it a "burn-in" (FWC - ramp-in) sequence of chars and let it update its hiddens tate after each char
  * See: 'How to generate character strings from the model' slide to see what it "knows" **(FWC - to generate ideas from it)**
  * Also see: 'Some completions produced by the model' slide
    * "The meaning of life is *literary recognition.*" (6th try)
