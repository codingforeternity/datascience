[Scala Implicits](http://googlyadventures.blogspot.com/2016/03/today-i-taught-someone-scala-implicits.html) (9/14/16)
[Implicit Conversions and Parameters](http://www.artima.com/pins1ed/implicit-conversions-and-parameters.html)
* Converting Receivers: "Implicit conversions also apply to the receiver of a method call, the object on which the method is invoked. This kind of implicit conversion has two main uses. First, receiver conversions allow smoother integration of a new class into an existing class hierarchy. And second, **they support writing domain-specific languages (DSLs)** within the language."

[Exception Handling in Apache Spark](https://www.nicolaferraro.me/2016/02/18/exception-handling-in-apache-spark/) (9/14/16)
* import it.nerdammer.spark.additions._
* https://github.com/nerdammer/spark-additions

[A Tale of Three Apache Spark APIs: RDDs, DataFrames, and Datasets; When to use them and why](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html) (9/14/16)

[Real-time data analysis using Spark](http://blog.scottlogic.com/2013/07/29/spark-stream-analysis.html)

[Spark Streaming DStream (Significance Testing)](http://spark.apache.org/docs/latest/mllib-statistics.html#streaming-significance-testing)
* peacePeriod - The number of initial data points from the stream to ignore, used to mitigate novelty effects.  [This seems akin to a ramp-up period.]

[[Official] Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html) (9/7/16)

[[Very short] Spark Tutorial](http://www.tutorialspoint.com/apache_spark/apache_spark_rdd.htm) (9/6/16)
* "There are two ways to create RDDs − parallelizing an existing collection in your driver program, or referencing a dataset in an external storage system, such as a shared file system"

Scala IDE (Eclipse plugin)
* "more than one scala library found in the build path" - http://stackoverflow.com/questions/13682349/how-to-use-just-one-scala-library-for-maven-eclipse-scala
* "is cross-compiled with an incompatible version of Scala - http://scala-ide.org/docs/current-user-doc/faq/index.html
* "plugin execution not covered by lifecycle configuration" - https://www.eclipse.org/m2e/documentation/m2e-execution-not-covered.html
  * this means that Eclipse's "lifecycle configuration" isn't going to run this Maven plugin because they aren't hooked up to each other

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