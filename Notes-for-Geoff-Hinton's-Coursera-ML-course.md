Octave source: https://ftp.gnu.org/gnu/octave/

Week 2: Types of Neural Network Architectures
* Ilya Sutskever (2011) trained special type or RNN to predict next char in sequence
  * it generates text by predicting the probability distribution for the next char, not the highest likely next char (which would generate text like "the united states of the united states of the..."), and then sampling from that distribution
* symmetric nets are much easier to analyze than RNNs (John Hopfield)
  * but they're more restricted, e.g. can't learn cycles

Week 2: What perceptrons can't do
* A perceptron cannot recognize patterns under translation if we allow wrap around.
  * E.g. 2 binary input vectors, A and B, each with 4 out of 10 activated "pixels."  Each of the 10 pixels will be activated by 4 translations of A and of B, so the total input received by the decision unit over all these patterns, in both cases, will be four times the sum of all the weights.  But to discriminate correctly, every single case of pattern A must provide more input to the decision unit than every single case of pattern B.