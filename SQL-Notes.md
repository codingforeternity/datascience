[How to configure postgresql for the first time?](http://stackoverflow.com/questions/1471571/how-to-configure-postgresql-for-the-first-time)
* "Note that if you do a mere psql, it will fail since it will try to connect you to a default database having the same name as you (ie. whoami). template1 is the admin database that is here from the start."

#### [Pro Spark Streaming (Safari Books)](https://www.safaribooksonline.com/library/view/pro-spark-streaming/9781484214794/A367671_1_En_1_Chapter.html)
* ACID to BASE
  * Old relational model: Atomicity, Consistency, Isolation, Durability
  * New model: Basically Avaiable, Soft state, Eventual consistency
  * BASE prioritizes availability over consistency
    * Formally: The Consistency, Availability, Partitioning (CAP) theorem
      * "only two of the three CAP properties can be achieved at the same time"
* "Examples of popular NoSQL stores include key-value stores, such as Amazon’s DynamoDB and Redis; column-family stores, such as Google’s BigTable (and its open source version HBase) and Facebook’s Cassandra; and document stores, such as MongoDB."

[Great DB article](http://arstechnica.com/information-technology/2016/03/to-sql-or-nosql-thats-the-database-question/) (4/13/16)