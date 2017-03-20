### Week 1
* sbt tutorial
  * sbt console
    * You can start the Scala interpreter inside sbt using the console task. The interpreter (also called REPL, for "read-eval-print loop") is useful for trying out snippets of Scala code. Note that the interpreter can only be started if there are no compilation errors in your code.
  * sbt compile
  * sbt test
    * The directory src/test/scala contains unit tests for the project. In order to run these tests in sbt, you can use the test command.
  * sbt run
    * If your project has an object with a main method (or an object extending the `App` trait), then you can run the code in sbt easily by typing run. In case sbt finds multiple main methods, it will ask you which one you'd like to execute.
  * sbt submit
    * The sbt task submit allows you to submit your solution for the assignment. The submit tasks takes two arguments: your Coursera e-mail address and the submission password. NOTE: the submission password is not your login password.