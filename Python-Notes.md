[Why Python is Slow](http://blog.kevmod.com/2016/07/why-is-python-slow/) (7/5/16)
* it's the C code that's slow, not the JIT interpreter

[Asynchronous Programming with Python 3](https://community.nitrous.io/tutorials/asynchronous-programming-with-python-3) (5/6/16)
* Good explanation of `async` and `await` keywords introduced in Python 3.5 (similar to `synchronized` and Future in Java)

[Jamal Moir: An Introduction to Scientific Python (and a Bit of the Maths Behind It) - Matplotlib](http://feedproxy.google.com/~r/JamalMoirBlogPython/~3/J4BvLPu8J1g/scientific-python-matplotlib.html) (4/28/16)

[A Speed Comparison Of C, Julia, Python, Numba, and Cython on LU Factorization](https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization?lang=en) (4/15/16)
* bottom line: use scipy

[How does Python compare to C#?](https://www.quora.com/How-does-Python-compare-to-C) (1/11/16)

[Machine Learning in Python](https://www.dataquest.io/blog/getting-started-with-machine-learning-python/)
* Includes an intro to Pandas, Matplotlib, and Scikit-Learn

[Learn Python interactively with IPython - A Complete Tutorial!](https://github.com/rajathkumarmp/Python-Lectures)

[Probability, Paradox, and the Reasonable Person Principle](http://nbviewer.ipython.org/url/norvig.com/ipython/Probability.ipynb) (in iPython Notebook, by Peter Norvig)

[Straightening Loops: How to Vectorize Data Aggregation with pandas and NumPy](http://blog.datascience.com/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/)
* To summarize in terms of best performance at summing a list, NumPy ndarray `sum` > pandas Series `sum` > standard library `sum` > for loop > standard library `reduce`.
* DataFrame methods for aggregation and grouping are typically faster to run and write than the equivalent standard library implementation with loops. For instances where performance is a serious consideration, NumPy ndarray methods offer as much as one order of magnitude increases in speed over DataFrame methods and the standard library.

Email: OOC Dataframes
* Out-of-Core Dataframes in Python: Dask and OpenStreetMap https://jakevdp.github.io/blog/2015/08/14/out-of-core-dataframes-in-python/

Email: (no subject)
* Design Patterns Explained http://www.reddit.com/r/programming/comments/3lkcis/design_patterns_explained/
* http://www.pysnap.com/design-patterns-explained/