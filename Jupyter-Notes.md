#### [Jupyter Kernels](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)
* [StackOverflow: Choosing a Spark/Scala kernel for Jupyter/IPython](http://stackoverflow.com/questions/32858203/choose-spark-scala-kernel-for-jupyter-ipython) (2/14/17)
* I can't speak for all of them, but I use Spark Kernel and it works very well for using both Scala and Spark.
  * Spark Kernel is now [Apache Toree](https://developer.ibm.com/open/openprojects/apache-toree/) which possibly comes from IBM (i.e. red flag)
  * [jove-scala](http://people.duke.edu/~ccc14/sta-663/Jupyter.html) is no more.  The GitHub page says use jupyter-scala instead
  * Zeppelin looks pretty well developed.  [Zeppelin Notebook - big data analysis in Scala or Python in a notebook, and connection to a Spark cluster on EC2](http://christopher5106.github.io/big/data/2015/07/03/iPython-Jupyter-Spark-Notebook-and-Zeppelin-comparison-for-big-data-in-scala-and-python-for-spark-clusters.html) is a nice explanation of installing it on AWS.  Zeppelin is a JVM-based [alternative](https://github.com/alexarchambault/jupyter-scala) to Jupyter.
  * [This](https://developer.ibm.com/hadoop/2016/05/04/install-jupyter-notebook-spark/) IBM link says to use jupyter-scala
  * [Scala Notebook](https://github.com/Bridgewater/scala-notebook) (last commit 2015; from Bridgewater) - An alternative to Jupyter.
  * [IScala](https://github.com/mattpap/IScala) (no commits since 2014)
  * [jupyter-scala](https://github.com/alexarchambault/jupyter-scala) - This looks like the one to use (last commit 1/17)

#### [6 points to compare Python and Scala for Data Science using Apache Spark](https://datasciencevademecum.wordpress.com/2016/01/28/6-points-to-compare-python-and-scala-for-data-science-using-apache-spark/)
* Python is more analytical oriented while Scala is more engineering oriented

#### Example Notebooks (12/6/16)
* [Matplotlib Tutorial](http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)
* [Nice basic data sciencey, data cleaning blog post generated from a notebook](http://danielfrg.com/blog/2013/03/07/kaggle-bulldozers-basic-cleaning/)
* [The Importance of Preprocessing in Data Science and the Machine Learning Pipeline tutorial series](https://www.datacamp.com/community/tutorials/the-importance-of-preprocessing-in-data-science-and-the-machine-learning-pipeline-i-centering-scaling-and-k-nearest-neighbours#gs.nPFcZ2s)

#### [Python Data Science Handbook from O'Reilly by Jake VanderPlas](https://github.com/jakevdp/PythonDataScienceHandbook)
* Run `jupyter notebook` inside `~/code/PythonDataScienceHandbook/notebooks/` to see the book
  * Local version here: http://localhost:8888/notebooks/01.00-IPython-Beyond-Normal-Python.ipynb
* First chapter is a nice background on Jupyter

#### [Jupyter Notebook Tutorial: The Definitive Guide](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook?utm_campaign=Data%2BElixir&utm_medium=email&utm_source=Data_Elixir_107#gs.JClrSzA)
