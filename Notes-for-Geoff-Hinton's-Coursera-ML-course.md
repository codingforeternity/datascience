Octave source: https://ftp.gnu.org/gnu/octave/

Also see accompanying green Staples notebook.

<h3>Week 2: Types of Neural Network Architectures</h3>
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


