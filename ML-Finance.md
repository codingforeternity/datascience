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