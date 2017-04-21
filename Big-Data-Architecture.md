### Out-of-core data options (4/21/17)
* Why not just use TensorFlow for everything?
  * Dask creates its own execution graphs, but why is this necessary when TF already has them?
  * In particular, TF even has [support for reading from files](https://www.tensorflow.org/programmers_guide/reading_data).  So if that is the case, then why not just construct the files and start the TF graph there?
* [Dask](https://jakevdp.github.io/blog/2015/08/14/out-of-core-dataframes-in-python/)
  * Out-of-core functional/numpy/dataframes promoted by @jakevdp--so it must be good.
* [Xray + Dask: Out-of-Core, Labeled Arrays in Python](https://www.continuum.io/content/xray-dask-out-core-labeled-arrays-python)
  * Xray seems to have a clunky interface.
  * And doesn't Dask have the same functionality?

### [Big Data Architecture Patterns](https://www.youtube.com/watch?v=-N9i-YXoQBE) (10/3/16)
* Good YouTube talk describing all of the differences and the history of relational dbs (SQL) -> semi-structured -> document stores (NoSQL) along with a description of Hadoop (an architecture paradigm) along the way