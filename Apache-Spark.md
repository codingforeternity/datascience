#### [Real-time Streaming ETL with Structured Streaming in Apache Spark 2.1](https://databricks.com/blog/2017/01/19/real-time-streaming-etl-structured-streaming-apache-spark-2-1.html)

#### Problem w/ Spark
* Assumes everything is embarrassingly parallel
* Path dependence is an issue (e.g. back-test simulations or anything w/ ramp-up required for each datapoint)
* But what if many sims could be accomplished in the same proc?  Each variable computed simultaneously for all sims, then merely different rollups at the end (or at the point of divergence)
* Still, how do subequest sims benefit from prev? Parquet file caching? 

#### Links
* Flattening JSON w/ complex types: http://stackoverflow.com/questions/28332494/querying-spark-sql-dataframe-with-complex-types
* Reading JSON data in Spark DataFrames: http://xinhstechblog.blogspot.com/2016/05/reading-json-nested-array-in-spark.html (and older: http://xinhstechblog.blogspot.be/2015/06/reading-json-data-in-spark-dataframes.html)
* Match case with regex: https://www.safaribooksonline.com/library/view/scala-cookbook/9781449340292/ch01s09.html
* Use [java.time](http://stackoverflow.com/questions/3614380/whats-the-standard-way-to-work-with-dates-and-times-in-scala-should-i-use-java) (formerly JodaTime) for date-times in Java/Scala

#### [Pro Spark Streaming: The Zen or Real-time Analytics Using Spark](https://www.safaribooksonline.com/library/view/pro-spark-streaming/9781484214794/A367671_1_En_1_Chapter.html)
* In reference to Hadoop/MapReduce (not Spark): "*Iterative applications that perform the same computation multiple times are also a bad fit for Hadoop. Many machine-learning algorithms belong to this class of applications*. For example, k-means clustering in Mahout refines its centroid location in every iteration. This process continues until a threshold of iterations or a convergence criterion is reached. It runs a driver program on the user’s local machine, which performs the convergence test and schedules iterations. This has two main limitations: the output of each iteration needs to be materialized to HDFS, and the driver program resides in the user’s local machine at an I/O cost and with weaker fault tolerance."

#### [The future of streaming in Spark: *Structured Streaming* (new in Spark 2.0)](https://www.safaribooksonline.com/library/view/the-spark-video/9781491970355/video256085.html)
* "The simplest way to perform streaming analytics is not having to **reason** about streaming at all."
  * Spark should be smart enough to do all that complicated reasoning about fault tolerance, end-to-end guarantees
  * DStream API exposes batch time, hard to incorporate event time
  * RDD/DStream has similar API, but still requires translation
  * Reasoning about end-to-end guarantees
    * Requires carefully constructing sinks that handle failures correctly
    * Data consistency in the storage while being updated [FWC - atomicity]
* New Model
  * Think of the data coming in as being inserted into a database table.
  * Users can configure to receive the full table every interval, the deltas (new rows and modified rows), or appends (only new rows).
* Dataset, single unified API!  (bounded or unbounded)
  * Based on (and interoperable with) R/Pandas
  * 2 interfaces:

```sql
SELECT type, avg(signal)
FROM devices
GROUP BY type
```

```scala
ctxt.table("devices")
    .groupBy("type")
    .agg("type", avg("signal"))
    .map(lambda ...)
    .select
```

  * statically typed

```scala
case class DeviceData(type: String, signal: Int)
// convert data to Java objects
val ds: Dataset[DeviceData] = ctx.read.json("data.json").as[DeviceData]
// compute histogram of age by name
val hist - ds.groupBy(_.type).mapGroups {
  case (type, data: Iter[DeviceData]) =>
    val buckets = new Array[Int](10)
    data.map(_.signal).foreach { a =>
      buckets(a/10) += 1
    }
    (type, buckets)
  }
```

* Batch ETL with DataFrame

```scala
input = ctxt.read.format("json").load("source-path")
result = input.select("device", "signal").where("signal > 15")
result.write.format("parquet").save("dest-path")
```

* Streaming ETL with DataFrame

```scala
// read from Kafka or json stream
input = ctxt.read.format("json").stream("source-path") // <- stream!!!
result = input.select("device", "signal").where("signal > 15")
// write to parquet file stream (sequence of files)
result.write.format("parquet").startStream("dest-path") // <- startStream!!!
```

* [Structured Streaming Demo notebook](https://docs.databricks.com/spark/latest/structured-streaming/index.html)

* Convert log messages into structured data

```scala
// read files from a directory either once or as a stream!
val input = sqlContext.read
  .format("text")
  .load("/logs") // <- change 'load' to 'stream' to convert to streaming dataset!
  .as[String] // https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html
input.isStreaming
```

```scala
case class LogMessage(timeStamp: String, success: boolean, fullMsg: String)
val parsed = input.flatMap(_.split(" ").toList match {
  case date :: time :: msg =>
    val fullMsg = msg.mkString(" ")
    val success = fullMsg.contains("succsessful")
    val timeStamp = s"$date $time"
    LogMessage(timeStamp, success, fullMsg) :: Nil
  case _ => Nil
}}
```

```scala
import org.apahce.spark.sql.functions._
val stats = parsed
  .groupBy(window($"timestamp", "10 seconds"), $successful")
  .agg(count("*").as("counts"))
  .select(date_format($"window.end", "hh:mm:ss").as("window"), $"counts", $"successful")
```

* Plots/graphs in a notebook will be updated automatically when using streaming
* `input` and `stats` are handles to running streams which can be stopped, status, get error or get terminated

```scala
stats.stop()
stats.awaitTermination()
stats.exception() // if there was an exception, get it
stats.sourceStatuses()
stats.sinkStatus()
```

* End-to-end, exactly-once guarantees
  * **Fast**, *fault-tolerant*, **exactly-once** *stateful stream processing* without having to **reason** about streaming.
  * offset tracking in WAL + state management + fault-tolerant sources and sinks

#### 3 Spark Links
* https://spark.apache.org/docs/2.0.0-preview/mllib-linear-methods.html
* http://spark.apache.org/docs/latest/ml-pipeline.html
* http://spark.apache.org/docs/latest/ml-features.html#tf-idf

#### [Use cases and design patterns for Spark Streaming - Vida Ha (Databricks)](https://www.safaribooksonline.com/library/view/the-spark-video/9781491970355/video256086.html)
* Receiving data
  * Driver runs Receivers as long-running tasks
  * Receiver divides stream into _blocks_ and keeps in memory
  * Blocks also replicated to other executors
  * Every batch interval the driver launches tasks to process the blocks (and, e.g., write them to data store)
* Word count over a time window
  * `wordStream.reduceByKeyAndWindow((x: Int, y: Int) => x+y, windowSize, slidingInterval)`
  * windowSize is a multiple of the batch size
  * or you can use slidingInterval to do something like every 2 seconds rather than w/ windowSize
  * For performance
    * Increase the batch interval, if possible,
    * Incremental aggregations with inverse reduce function
    * Checkpointing - e.g. if batch interval is 30 minutes, set up checkpoints every 10 to ensure that you don't have to go back 30 if there's a (driver) failure - `wordStream.checkpoint(checkpointInterval)`
    * Keep processing time < 80% of batch interval (o/w batches start queueing up)

#### [Beyond shuffling: Tips and tricks for scaling Spark jobs - Holden Karau (IBM)](https://www.safaribooksonline.com/library/view/the-spark-video/9781491970355/video256090.html)
* If using non-JVM language (Python or R), then you should really use DataFrames.  The cost of using RDDs in any non JVM language is quite high because the data needs to be copied from the JVM to worker processes (running Python or R) and then back to JVM.  The data has to be serialized twice, e.g. in Python it gets Pickled.
* Where does DataFrame explode?
  * Iterative algorithms - large execution plans
  * Default shuffle size is sometimes too small for big data (200 partitions)
  * Default partition size when reading in data is also sad
    * Can read in data using RDD API and then convert to DF afterwards ("known'ish thing")
* Avoid lineage explosions: Cut the lineage of a DataFrame that has too long of an execution plan.  "It's kinda silly that we have to do this, but this is where we are."  Don't want to do this in Python though (for same reasons mentioned above).
```scala
def cutLineage(df: DataFrame): DataFrame = {
  val sqlCtx = df.sqlContext
  val rdd = df.rdd
  rdd.cache()
  sqlContext.createDataFrame(rdd, df.schema)
}
```
* Spark testing resources
  * Scala: spark-testing-base (scalacheck & unit), sscheck (scalacheck), example-spark (unit)
  * 'Unit Testing Spark with Java' by Jesse Anderson
  * 'Making Apache Spark Testing Easy with Spark Testing Base'
  * 'Unit Testing Apache Spark with py.test'

#### [SparkNet: Training deep networks in Spark - Robert Nishihara (UC Berkley)](https://www.safaribooksonline.com/library/view/the-spark-video/9781491970355/video256080.html) (11/22/16)
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
* Update 1/3/17 - Or here may be an alternative: [BigDL: Distributed Deep learning Library for Apache Spark](https://github.com/intel-analytics/BigDL)

#### [Performing Advanced Analytics on Relational Data with Spark SQL](https://www.safaribooksonline.com/library/view/performing-advanced-analytics/9781491908297/part00.html?autoStart=True)
* "**Similar to typical ETL**, except doing it all in one program!"
```scala
// define schema using a case class (similar to POJO or Java Bean)
case class Person(name: String, age: Int)
// create RDD of person objects and register as a table
val people = sc.textFile("examples/src/main/resources/people.txt")
               .map(_.split(","))
               .map(p => Person(p(0), p(1).trim.toInt))
// "invoking reflection at runtime to figure out the names and types of columns in this table"
people.registerAsTable("people") // FWC - this is the key!  Spark already has registration!!!
```
* **WARNING: `registerAsTable` has been replaced by `createOrReplaceTempView` in newer versions of Spark**
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

#### [Exception Handling in Apache Spark](https://www.nicolaferraro.me/2016/02/18/exception-handling-in-apache-spark/) (9/14/16)
* import it.nerdammer.spark.additions._
* https://github.com/nerdammer/spark-additions

#### [A Tale of Three Apache Spark APIs: RDDs, DataFrames, and Datasets; When to use them and why](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html) (9/14/16)
* "When to use RDDs? ...you want to manipulate your data with functional programming constructs than domain specific expressions"
* "Dataset, by contrast, is a collection of strongly-typed JVM objects, dictated by a case class you define in Scala or a class in Java"
* "Whereas the Dataset[T] typed API is optimized for data engineering tasks, the untyped Dataset[Row] (an alias of DataFrame) is even faster and suitable for interactive analysis."

####[Real-time data analysis using Spark](http://blog.scottlogic.com/2013/07/29/spark-stream-analysis.html)

#### [Spark Streaming DStream (Significance Testing)](http://spark.apache.org/docs/latest/mllib-statistics.html#streaming-significance-testing)
* peacePeriod - The number of initial data points from the stream to ignore, used to mitigate novelty effects.  [This seems akin to a ramp-up period.]

#### [[Official] Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html) (9/7/16)

#### [[Very short] Spark Tutorial](http://www.tutorialspoint.com/apache_spark/apache_spark_rdd.htm) (9/6/16)
* "There are two ways to create RDDs − parallelizing an existing collection in your driver program, or referencing a dataset in an external storage system, such as a shared file system"

#### [Flink vs. Spark](http://www.slideshare.net/sbaltagi/flink-vs-spark) (9/6/16)
* and more here: http://sparkbigdata.com/

#### [The Essential Guide to Streaming-first Processing with Apache Flink](https://www.mapr.com/blog/essential-guide-streaming-first-processing-apache-flink)
* "Until now, data streaming technology was lacking in several areas, such as performance, correctness, and operability, forcing users to roll their own applications to ingest and analyze these continuous data streams, or (ab)use batch processing tools to simulate continuous ingestion and analysis pipelines."
* Note the word "pipeline."  I wonder if this sentence hints at the origin of the term.  A pipeline, by definition (or originally intended definition) really means a streamed application.  The intent of UIMA may have been this originally, but it sure has diverged from that since then reverting back to more of a batch processing infrastructure.

#### [SparkML](http://web.cs.ucla.edu/~mtgarip/) (7/7/16)

#### [Spark Custom Streaming Sources](https://medium.com/@anicolaspp/spark-custom-streaming-sources-e7d52da72e80#.gk1plv86q) (5/3/16)

#### [Hadoop or AWS more useful for Machine Learning careers?](http://www.reddit.com/r/MachineLearning/comments/4e81ne/hadoop_or_aws_more_useful_for_machine_learning/) (4/13/16)

#### [The Future of Hadoop is Misty](https://www.linkedin.com/pulse/future-hadoop-misty-haifeng-li) (3/10/16)
* Why use Hadoop when you can just spin up nodes on AWS?
* Lots of the components of Hadoop have been surpassed by newer technologies, e.g. Spark.
* Cloudera possibly in trouble.

#### [Demystifying the complexities of Spark and Hadoop](https://www.reddit.com/r/programming/comments/46l6ao/demystifying_the_complexities_of_spark_and_hadoop/) (2/20/16)

#### [Dataflow/Beam & Spark: A Programming Model Comparison](https://www.reddit.com/r/programming/comments/441qop/dataflowbeam_spark_a_programming_model_comparison/) (2/3/16)

#### [How to check hypotheses with bootstrap and Apache Spark?](https://www.reddit.com/r/programming/comments/43fnb4/how_to_check_hypotheses_with_bootstrap_and_apache/) (1/30/16)

#### [Analyzing Real-time Data With Spark Streaming In Python](http://prateekvjoshi.com/2015/12/22/analyzing-real-time-data-with-spark-streaming-in-python/) (1/20/16)