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