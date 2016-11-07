Octave source: https://ftp.gnu.org/gnu/octave/

Week 2: Types of Neural Network Architectures
* Ilya Sutskever (2011) trained special type or RNN to predict next char in sequence
  * it generates text by predicting the probability distribution for the next char, not the highest likely next char (which would generate text like "the united states of the united states of the..."), and then sampling from that distribution
* symmetric nets are much easier to analyze than RNNs (John Hopfield)
  * but they're more restricted, e.g. can't learn cycles

Week 2: What perceptrons can't do
* A perceptron cannot recognize patterns under translation if we allow wraparound.
  * E.g. 2 binary input vectors, A and B, each with 4 out of 10 activated "pixels."  Each of the 10 pixels will be activated by 4 translations of A and of B, so the total input received by the decision unit over all these patterns, in both cases, will be four times the sum of all the weights.  But to discriminate correctly, every single case of pattern A must provide more input to the decision unit than every single case of pattern B.
  * However, if we have 3 patterns assigned to two classes, A and B, and A contains a pattern with 4 pixels while B contains patterns with 1 and 3 pixels then a *binary* decision unit can classify if we allow translations and wraparound.  Weight = 1 from each pixel with bias of -3.5
  * Minskey and Papert's "Group Invariance Theorem" says that the part of a perceptron that learns cannot do this if the transformations form a group (e.g. translations with wraparound).
  * This result is devastating for pattern recognition (PR) because the whole point of PR is to recognize patterns that undergo transformations, like translation.
  * To deal with such transformations, a perceptron needs to use multiple feature units to recognize transformations of informative sub-patterns.
  * So the tricky part of PR must be solved by the hand-coded feature detectors, not the learning procedure.
  * Networks without hidden units are very limited in what they can learn to model.

Week 3: Learning the weights of a linear neuron
* Instead of showing the weights get closer to a good set of weights (i.e. perceptrons, which suffer from 2 good set of weights do not average to a good set) show that actual output values get closer to the target values.
  * In perceptron learning the outputs can get farther away from the targets, even though the weights are getting closer.
* The "delta rule" for learning: delta w_i = epsilon * x_i * (t - y) ... where epsilon := "learning rate", t := target/true output, and y := estimated output
  * Derivation: Error = 0.5 * Sum_{n in training} (t_n - y_n)^2 ... the 1/2 is only there to make the 2 in the derivative cancel
  * Differentiate the error wrt one of the weights, w_i: del E / del w_i = 0.5 * Sum_n (del y_n / del w_i * d E_n / d y_n) ... Chain Rule ("easy to remember, just cancel those 2 del y_ns ... but only when there aren't any mathematicians looking) ... = -Sum_n (x_{i,n} * (t_n - y_n)
    * del y_n / del w_i = x_{i,n} because y_n = w_i * x_{i,n}
    * d E_n / d y_n is just the derivative of the (squared) Error function
  * Therefore: delta w_i = -epsilon * del E / del w_i

Week 3: The error surface for a linear neuron
* Difference between "batch" and "on-line"
  * Simplest batch learning does steepest gradient descent
  * On-line/stochastic zig-zags between training case "lines" at each step moving perpendicularly to a line.  Imagine the intersection of 2 training case lines and moving perpendicularly back and forth perpendicularly to both while converging on their intersection point.
    * This is very slow if the variables are highly correlated (very elongated ellipse) because the perpendicular updates (the gradients) are also perpendicular to the intersection.
    * FWC idea - look at angle between consecutive gradients to detect correlated dimensions

Week 3: The backpropagation algorithm
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
  * Finite Difference Approximation (compute +/- epsilon changes for each weight and move in that direction) works, but backprop finds the exact gradient much more quickly.
 * FWC - machine learning (backprop) is all about the learning rate!  So you can either be smart (and use backprop) or buy more computers.  You're only constrained if you need both.  Search Google for: "machine learning for the maximization of an arbitrary function"