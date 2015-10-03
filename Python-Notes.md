[Straightening Loops: How to Vectorize Data Aggregation with pandas and NumPy](http://blog.datascience.com/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/)
* To summarize in terms of best performance at summing a list, NumPy ndarray `sum` > pandas Series `sum` > standard library `sum` > for loop > standard library `reduce`.
* DataFrame methods for aggregation and grouping are typically faster to run and write than the equivalent standard library implementation with loops. For instances where performance is a serious consideration, NumPy ndarray methods offer as much as one order of magnitude increases in speed over DataFrame methods and the standard library.

Email: OOC Dataframes
* Out-of-Core Dataframes in Python: Dask and OpenStreetMap https://jakevdp.github.io/blog/2015/08/14/out-of-core-dataframes-in-python/

Email: (no subject)
* Design Patterns Explained http://www.reddit.com/r/programming/comments/3lkcis/design_patterns_explained/
* http://www.pysnap.com/design-patterns-explained/