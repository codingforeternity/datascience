[Beyond shuffling: Tips and tricks for scaling Spark jobs - Holden Karau (IBM)](https://www.safaribooksonline.com/library/view/the-spark-video/9781491970355/video256090.html)
* If using non-JVM language (Python or R), then you should really use DataFrames.  The cost of using RDDs in any non JVM language is quite high because the data needs to be copied from the JVM to worker processes (running Python or R) and then back to JVM.  The data has to be serialized twice, e.g. in Python it gets Pickled.
* Where does DataFrame explode?
  * Iterative algorithms - large execution plans
  * Default shuffle size is sometimes too small for big data (200 partitions)
  * Default partition size when reading is also sad

[SparkNet: Training deep networks in Spark - Robert Nishihara (UC Berkley)](https://www.safaribooksonline.com/library/view/the-spark-video/9781491970355/video256080.html) (11/22/16)
* Why do we need SparkNet (built on top of Caffe and TensorFlow) when we already have MLLib?
  * Because MLLib doesn't support construction of different network architectures.
  * It only supports models where the model specification is the same every time, e.g. LogisticRegression, RandomForest.
  * Caffe and TensorFlow are C++ projects so they use JavaCPP to connect SparkNet to them.
* How to average weights from models trained on different Spark nodes:
```scala
val weights = workers.map(_ => net.getWeights())
  .reduce((a, b) => WeightCollection.add(a, b))
WeightCollection.scale(weights, 1F / numWorkers)
val broadcastWeights = sc.broadcast(weights)
workers.foreach(_ => net.setWeights(broadcastWeights.values))
```
* Code: https://github.com/amplab/SparkNet
* Paper: http://arxiv.org/abs/1511.06051

[Performing Advanced Analytics on Relational Data with Spark SQL](https://www.safaribooksonline.com/library/view/performing-advanced-analytics/9781491908297/part00.html?autoStart=True)
* "**Similar to typical ETL**, except doing it all in one program!"
```scala
// define schema using a case class (similar to POJO or Java Bean)
case class Person(name: String, age: Int)
// create RDD of person objects and register as a table
val people = sc.textFile("examples/src/main/resources/people.txt")
               .map(_split(","))
               .map(p => Person(p(0), p(1).trim.toInt))
// "invoking reflection at runtime to figure out the names and types of columns in this table"
people.registerAsTable("people") // FWC - this is the key!  Spark already has registration!!!
```
* SparkSQL caches tables using an in-memory columnar format (which is also the storage mechanism provided by Parquet)
  * scan only required columns
  * fewer allocated objects (GC) - same data in a Python RDD 4GB vs. 2GB for a Spark RDD
  * automatically select best compression
* Disk caching
```scala
people.saveAsParquetFile("people.parquet")
// Parquet files are self-describing so the schema is preserved
val parquetFile = sqlContext.parquetFile("people.parquet")
// parquet files can also be registered as tables (and then used in SQL statements)
parquetFile.registerAsTable("parquetFile")
```
* Data from multiple sources--all join'able via SparkSQL
* Conclusion: Big data analytics is evolving to include:
  * more *complex* analytics (e.g. machine learning)
  * more *interactive* ad-hoc queries, including SQL
  * more *real-time* stream processing
  * Spark is a fast platform that *unifies* these apps

[Exception Handling in Apache Spark](https://www.nicolaferraro.me/2016/02/18/exception-handling-in-apache-spark/) (9/14/16)
* import it.nerdammer.spark.additions._
* https://github.com/nerdammer/spark-additions

[A Tale of Three Apache Spark APIs: RDDs, DataFrames, and Datasets; When to use them and why](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html) (9/14/16)
* "When to use RDDs? ...you want to manipulate your data with functional programming constructs than domain specific expressions"
* "Dataset, by contrast, is a collection of strongly-typed JVM objects, dictated by a case class you define in Scala or a class in Java"
* "Whereas the Dataset[T] typed API is optimized for data engineering tasks, the untyped Dataset[Row] (an alias of DataFrame) is even faster and suitable for interactive analysis."

[Real-time data analysis using Spark](http://blog.scottlogic.com/2013/07/29/spark-stream-analysis.html)

[Spark Streaming DStream (Significance Testing)](http://spark.apache.org/docs/latest/mllib-statistics.html#streaming-significance-testing)
* peacePeriod - The number of initial data points from the stream to ignore, used to mitigate novelty effects.  [This seems akin to a ramp-up period.]

[[Official] Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html) (9/7/16)

[[Very short] Spark Tutorial](http://www.tutorialspoint.com/apache_spark/apache_spark_rdd.htm) (9/6/16)
* "There are two ways to create RDDs − parallelizing an existing collection in your driver program, or referencing a dataset in an external storage system, such as a shared file system"

[Flink vs. Spark](http://www.slideshare.net/sbaltagi/flink-vs-spark) (9/6/16)
* and more here: http://sparkbigdata.com/

[The Essential Guide to Streaming-first Processing with Apache Flink](https://www.mapr.com/blog/essential-guide-streaming-first-processing-apache-flink)
* "Until now, data streaming technology was lacking in several areas, such as performance, correctness, and operability, forcing users to roll their own applications to ingest and analyze these continuous data streams, or (ab)use batch processing tools to simulate continuous ingestion and analysis pipelines."
* Note the word "pipeline."  I wonder if this sentence hints at the origin of the term.  A pipeline, by definition (or originally intended definition) really means a streamed application.  The intent of UIMA may have been this originally, but it sure has diverged from that since then reverting back to more of a batch processing infrastructure.

[SparkML](http://web.cs.ucla.edu/~mtgarip/) (7/7/16)

[Spark Custom Streaming Sources](https://medium.com/@anicolaspp/spark-custom-streaming-sources-e7d52da72e80#.gk1plv86q) (5/3/16)

[Hadoop or AWS more useful for Machine Learning careers?](http://www.reddit.com/r/MachineLearning/comments/4e81ne/hadoop_or_aws_more_useful_for_machine_learning/) (4/13/16)

[The Future of Hadoop is Misty](https://www.linkedin.com/pulse/future-hadoop-misty-haifeng-li) (3/10/16)
* Why use Hadoop when you can just spin up nodes on AWS?
* Lots of the components of Hadoop have been surpassed by newer technologies, e.g. Spark.
* Cloudera possibly in trouble.

[Demystifying the complexities of Spark and Hadoop](https://www.reddit.com/r/programming/comments/46l6ao/demystifying_the_complexities_of_spark_and_hadoop/) (2/20/16)

[Dataflow/Beam & Spark: A Programming Model Comparison](https://www.reddit.com/r/programming/comments/441qop/dataflowbeam_spark_a_programming_model_comparison/) (2/3/16)

[How to check hypotheses with bootstrap and Apache Spark?](https://www.reddit.com/r/programming/comments/43fnb4/how_to_check_hypotheses_with_bootstrap_and_apache/) (1/30/16)

[Analyzing Real-time Data With Spark Streaming In Python](http://prateekvjoshi.com/2015/12/22/analyzing-real-time-data-with-spark-streaming-in-python/) (1/20/16)