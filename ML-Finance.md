Email: 2 things
* Just like image recognition starts with pixels, then lines, then shapes, then objects.
* Stock forecasting should happen in the same granular/machine to general/human sort of way.
* How those forecasts are best used though is another problem going in the opposite direction. Like an hourglass. From the human/trader to the market/machine.
* Perhaps there can be different dimensions of generalization happening at the same time. Factor models are somewhat orthogonal to the dimension described in the previous paragraphs, for example. Orthogonal neural nets (which might just be closer to how a brain actually works where different sensory systems interact with each other).

Email: Rather than learning...
* ...optimal stocks to trade and then learning how to predict those, instead learn optimal factors (risk and forecast), and tcosts, and residual stock positions.
* Can the factor model be learned also?

Idea: Anomaly Detection (9/16/15)
* had this idea while watching https://www.coursera.org/learn/machine-learning/lecture/LSpXm/choosing-what-features-to-use
* perhaps anomaly detection can be used to red-flag stocks w/ something weird going on (e.g. accounting irregularities)
* use it to identify stocks with future (in the expected holding horizon window) 1-day large returns, cases where getting it wrong would be really bad (or just really risky)
* the trick would of course be to have the features engineer themselves
* Andrew Ng - "unusual combinations of values of features"
* of course this might be better learned w/ a neural net because there are a lot of examples, but on the other hand, they all happen for different reasons

Email: How would u get a neural net... (8/14/15)
* ...to consider time. To evolve with time. Both regime breaks, as well as slow transitions, as well as nothing.
* Multiple nets. One for each possible regime. And a regime is defined by which strategies work well for prediction during that time.
* This is just like the autonomous car video at the end of week 4 of the ML course.  It has one NN for driving one-lane roads and another for driving two-lane roads--and it switches between the two based on some algorithm, perhaps its own NN.
  * This could all be combined into a multi-layer NN where new input features are introduced at later-on mixed hidden/input layers.  So the early input layers are used to choose the regime (e.g. correlations of common variables to future outcomes).  This can boil down to one or two output neurons, to represent two to four regimes.  Then, from there, individual instrument variables can be introduced and combined with the previous output layer to another hidden layer and then on to an output layer for individual instrument forecasts.
  * This perhaps can be generalized even more to build nets of generalized universe-wide variables, to less general not quite as universal variables (sector forecasts? long-term forecasts?), to very specific outputs (individual instrument, short-term forecasts).  Perhaps the brain works like this; not every neuron needs to code the same specificity of information.  Not every neuron needs to be on equal footing (connected to) every other neuron.  But typically this seems how NNs are designed, at least as far as it doesn't seem to make sense to have multiple hidden layers because they can be combined into a single hidden layer via matrix multiplication.
* Another thought: regimes may often be defined by what forecasts work during which periods.  So the accuracy of a NN may define it's regime, to some extent.  This means that the output--the cost, the error--of the NN may feed back around as an input, creating a sort of cyclicality.
  * Of course this could also be done by doing the thing with PCA that I discuss in another thread: i.e. looking for variables that define regimes first based on their correlations to future returns.  And then using some sort of unsupervised learning to tease apart regimes from the PCs.
* See the hierarchy in [this post](http://stats.stackexchange.com/questions/114385/what-is-the-difference-between-convolutional-neural-networks-restricted-boltzma) for how multi-layer RBMs are used to identify pixels, then lines, then shapes, then faces (see the graphics)
* "The apparent randomness of the markets, unlike a strong pseudo-random number generator, appear to be affected by time dimension. In other words, certain window sizes cause markets to appear less random. This may indicate the presence of cyclical non-random behaviours in the markets e.g. regimes." http://www.turingfinance.com/hacking-the-random-walk-hypothesis/

Email: how to use news data (8/25/15)
* don't look for "sentiment", look for multi-year trends, starting and stopping.  e.g. china has been hot shit for ever since the last downturn.  but now it's crashing and so you'd expect that to effect us markets.  last time around it was housing, and the time before that it was the internet.  in all of these cases the prevailing trend came to an end, which i'm sure is detectable in news
* again, these sorts of trends can probably be discovered using neural nets.  you only need to discover the regime.  let the forecasters decide what happens in what regime