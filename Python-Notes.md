#### [Pandas Cheat Sheet](http://www.kdnuggets.com/2017/01/pandas-cheat-sheet.html) (1/30/17)

#### [How Python Makes Working With Data More Difficult in the Long Run](https://www.jeffknupp.com/blog/2016/11/13/how-python-makes-working-with-data-more-difficult-in-the-long-run/)
* "Let's consider two definitions of "good code" so we can be clear what we mean by better.
  1. Code that is short, concise, and can be written quickly
  2. Code that is maintainable
* If we're using the first definition, the Python version is "better". If we're using the second, it's far, far worse."
* "I've painted a rather bleak picture of using Python to manipulate complex (and even not-so-complex) data structures in a maintainable way. In truth, however, it's a shortcoming shared by *most* dynamic languages. In the second half of this article, I'll describe what various people/companies are doing about it, from simple things like the movement towards 'live data in the editor' all the way to the Dropboxian 'type-annotate all the things' (**[Static Typing in Python](https://www.dropbox.com/s/efatwr0pozsargb/PyCon%20mypy%20talk%202016.pdf?dl=0)**). In short, there's a lot of interesting work going on in this space and lot's of people are involved (notice the second presenter name [Guido] in that Dropbox deck)."

#### [Practical Machine Learning Tutorial with Python Introduction](https://pythonprogramming.net/machine-learning-tutorial-python-introduction/)
* From Reddit - [What is the best python based ML course available online?](https://www.reddit.com/r/MachineLearning/comments/4thirl/what_is_the_best_python_based_ml_course_available/)

#### [Data Mining in Python: A Guide](https://www.springboard.com/blog/data-mining-python-tutorial/)
* Good overview of the tools and IPython Notebook

#### [Vladimir Iakolev: Abusing annotations with dependency injection](https://nvbn.github.io/2016/08/07/annotations-injector/) (8/7/16)

#### [Why Python is Slow](http://blog.kevmod.com/2016/07/why-is-python-slow/) (7/5/16)
* it's the C code that's slow, not the JIT interpreter

#### [Asynchronous Programming with Python 3](https://community.nitrous.io/tutorials/asynchronous-programming-with-python-3) (5/6/16)
* Good explanation of `async` and `await` keywords introduced in Python 3.5 (similar to `synchronized` and Future in Java)

#### [Jamal Moir: An Introduction to Scientific Python (and a Bit of the Maths Behind It) - Matplotlib](http://feedproxy.google.com/~r/JamalMoirBlogPython/~3/J4BvLPu8J1g/scientific-python-matplotlib.html) (4/28/16)

#### [A Speed Comparison Of C, Julia, Python, Numba, and Cython on LU Factorization](https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization?lang=en) (4/15/16)
* bottom line: use scipy

#### [How does Python compare to C#?](https://www.quora.com/How-does-Python-compare-to-C) (1/11/16)

#### [Machine Learning in Python](https://www.dataquest.io/blog/getting-started-with-machine-learning-python/)
* Includes an intro to Pandas, Matplotlib, and Scikit-Learn

#### [Learn Python interactively with IPython - A Complete Tutorial!](https://github.com/rajathkumarmp/Python-Lectures)

#### [Probability, Paradox, and the Reasonable Person Principle](http://nbviewer.ipython.org/url/norvig.com/ipython/Probability.ipynb) (in iPython Notebook, by Peter Norvig)

#### [Straightening Loops: How to Vectorize Data Aggregation with pandas and NumPy](http://blog.datascience.com/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/)
* To summarize in terms of best performance at summing a list, NumPy ndarray `sum` > pandas Series `sum` > standard library `sum` > for loop > standard library `reduce`.
* DataFrame methods for aggregation and grouping are typically faster to run and write than the equivalent standard library implementation with loops. For instances where performance is a serious consideration, NumPy ndarray methods offer as much as one order of magnitude increases in speed over DataFrame methods and the standard library.

#### Email: OOC Dataframes
* Out-of-Core Dataframes in Python: Dask and OpenStreetMap https://jakevdp.github.io/blog/2015/08/14/out-of-core-dataframes-in-python/

#### Email: (no subject)
* Design Patterns Explained http://www.reddit.com/r/programming/comments/3lkcis/design_patterns_explained/
* http://www.pysnap.com/design-patterns-explained/