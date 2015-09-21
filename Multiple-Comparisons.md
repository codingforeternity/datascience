Email: "Average correlation..."
* ...Of multiple comparisons can perhaps be used to determine little n's fraction of big N. One way to test this is with 3 variables of varying correlations and then 3 more with different corrs but same avg. Do they have the same FDRs? Then try w 4 vars, then 5, etc.
* Or perhaps the distribution of the correlations matters, eg stdev 
* The actual PCs don't matter, they're only used to count the number of variables via their singular values.  Consider 2 almost perfectly colinear variables.  Regardless of the dimensionality of the space, the ratio of the singular values will be very large, i.e. they're really more like one variable than two.  Now consider 2 almost perfectly orthogonal variables.  The ratio of their singular values will be close to 1.  Divide by the largest SV in both scenarios and sum across adjusted SVs and you get something close to 1 in the first case (e.g. 1 + 0.01) and something close to 2 in the second scenario (e.g. 1 + 0.99).
* here's a similar question on the subject with a data set similar to Johanna's:
http://stats.stackexchange.com/questions/13739/can-we-combine-false-discovery-rate-and-principal-components-analysis
  * "simple M" - A Multiple Testing Correction Method for Genetic Association Studies Using Correlated Single Nucleotide Polymorphisms - the number of tests is the number of principal components to explain 99.5% of the variance
  * This isn't right though.  Imagine 2 dimensions with no correlation but where one dimension has variance that is 200x (i.e. 100 / (100 - 99.5)) the variance of the other.  this would indicate only one comparison is being made, but there isn't any correlation between the 2 dimensions so 2 comparisons are actually being made.  One solution: normalize all dimensions before performing PCA to the same variance.  Note though, that my procedure would have the same issue as one PC would be much larger than the second, so it would also determine there to be a single comparison.

Email: "can you get this paper too?"
* http://bioinformatics.oxfordjournals.org/content/19/3/368.short It's mentioned in the biostathandbook.com/multiplecomparisons.html link I sent earlier.
* Read "Discussion" beginning on page 373 for more having to do with what's discussed here http://www.biostathandbook.com/multiplecomparisons.html
* and a few more (see email for hard copies of papers)
  * http://link.springer.com/article/10.1007/BF02294317
  * http://epm.sagepub.com/content/early/2014/09/12/0013164414548894.abstract
  * http://www.sciencedirect.com/science/article/pii/S0167947302000725
* "A data based algorithm for the generation of random vectors" http://www.sciencedirect.com/science/article/pii/0167947386900137 (referenced here (with code!): http://stackoverflow.com/questions/19061997/simulating-correlated-multivariate-data)
* "Correlated, Uniform, Random Values" http://www.acooke.org/random.pdf (also see email for hard copy)

Email: "good description of Bonferroni-Hochberg Procedure
* http://www.biostathandbook.com/multiplecomparisons.html
* From Johanna: "this is really great -- i plan to try this with my correlation data (all other data have been corrected for multiple comparisons). it seems like even though my results may no longer be significant, i should likely report them anyway (without correction, but using terms such as 'possible effect') given that false negatives are costly in my case (?)..."
* My response:
  * I think you were instructed to do that precisely :)
  * What are you planning to report exactly?  The correlations between your measurements and DR?  Or the differences in group means for your measurements, which have their own P stats?
  * Note also that you want the Benjamini–Hochberg–Yekutieli procedure (http://en.wikipedia.org/wiki/False_discovery_rate) which attempts to control for dependency (Benjamini-Hochberg does not).
* More from Johanna:
  * i was going to report it all -- correlations between measures and DR, and differences in group means (with corresponding p values).
  * will check out the B-H-Y procedure shortly -- does this version of the test essentially do the same thing as narrowing my dependent variable list by choosing the most representative variable for a set of inter-dependent variables (i.e. those variables which i know vary together either biologically, or numerically) as per our discussion? what do you mean by controlling for dependency? maybe i should read first. yes, i'll do that.
* And me again:
  * yes, that's what BHY does. BH isn't exactly what you want because it assumes complete independence between the tests so BH it too conservative (i.e. it thinks you have more tests than you really do).  for example, consider the case where all of the tests are perfectly correlated (i.e. all the same), that's the same thing as one test, not many.  then there's the case where there isn't any dependence, that's when you use BH.  then there are all the cases in between (i.e. yours) so you have to control for that dependence by either using BHY or using permutations or using simulated data (hence the Vale-Maurelli article) to determine the percentiles of the distribution that you're dealing with.

Email: "multiple comparisons (Bonferroni and Benjamini-Hochberg)"
* if an industry adopts the new patented procedure, then it should be possible to test whether the usage of the procedure is being obeyed.  simple collect a set of studies that all have a given alpha, e.g. 0.05, and approximately 5% of them should be false positives on their post-publish data. (http://www.biostathandbook.com/multiplecomparisons.html)
* https://www.google.com/search?q=manova&ie=utf-8&oe=utf-8&aq=t&rls=org.mozilla:en-US:official&client=firefox-a&channel=fflb
* http://bioinformatics.oxfordjournals.org/content/19/3/368.short
* BH vs BHY (which controls for dependence) vs Storey's pFDR, which says that BH is still too conservative when there is dependence: http://stats.stackexchange.com/questions/63441/what-are-the-practical-differences-between-the-benjamini-hochberg-1995-and-t
* "THE CONTROL OF THE FALSE DISCOVERY RATE INMULTIPLE TESTING UNDER DEPENDENCY" http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_yekutieli_ANNSTAT2001.pdf
* "The control of FDR assumes that when many of the tested hypotheses are rejected it may be preferable to control the proportion of errors rather than the probability of making even one error." http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_yekutieli_ANNSTAT2001.pdf

General Statistics ("for hackers")
* Statistics for Hackers (slides) http://www.reddit.com/r/programming/comments/3lk5y6/statistics_for_hackers_slides/
  * "In general, computing the sampling distribution is hard, but simulating the sampling distribution is easy."  E.g. P(>22 heads out of 30 tosses) => just write a for loop: `randint(2, size=30).sum() >= 22`
  * Four Recipes for Hacking Statistics
    1. Direct Simulation
    2. Shuffling
    3. Bootstrapping
    4. Cross Validation
  * Shuffling:
    * "Simulate the distribution by shuffling the labels repeatedly and computing the desired statistic.  If the labels really don't matter, then switching them shouldn't change the result."
    * "Works when the _Null Hypothesis_ assumes two groups are equivalent"
    * "Like all methods, it will only work if your samples are representative--always beware selection biases"
    * For more discussion & references see: _Statistics is Easy_ by Shasha & Wilson
  * Bootstrapping
    * We need a way to simulate samples, but we don't have a generating model. Solution: Bootstrap Resampling (with replacement)
```python
for i in range(10000):
  sample = N[randint(20, size=20)]
  xbar[i] = mean(sample)
mean(xbar), std(xbar)
```