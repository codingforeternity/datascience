Java Resources
* https://github.com/cxxr/better-java

Coursera Java
* https://www.coursera.org/course/algs4partI and https://www.coursera.org/course/algs4partII
* Java book: http://introcs.cs.princeton.edu/java/home/

[Java Design Patterns](http://javarevisited.blogspot.sg/2012/06/20-design-pattern-and-software-design.html)
* abstract class vs. interface: "Interface are used to represent adjective or behavior e.g. Runnable, Clonable, Serializable etc"
* "classes to provide Market Data": "MarketData should be composed with a MarketDataProvider by using dependency injection. So when you change your MarketData provider Client won't get affected because they access method form MarketData interface or class."
* "What is main benefit of using factory pattern ? Where do you use it? Factory pattern’s main benefit is increased level of encapsulation while creating objects. If you use Factory to create object you can later replace original implementation of Products or classes with more advanced and high performance implementation without any change on client layer. See my post on [Factory pattern](http://javarevisited.blogspot.com/2011/12/factory-design-pattern-java-example.html) for more detailed explanation and benefits."
  * "The factory methods are typically implemented as virtual methods, so this pattern is also referred to as the 'Virtual Constructor'."

Email: Java
* Top 40 Core Java Interview Questions Answers from Telephonic Round http://www.reddit.com/r/programming/comments/3kvoy9/top_40_core_java_interview_questions_answers_from/

Email: Java sites
* 10 websites that help Java developers daily
* http://codeinventions.blogspot.com/2014/12/Most-Useful-websites-that-help-Java-developers.html

Email: C++ to Java
* This tutorial is intended for students who are already familiar with C++ and with data structures, and are interested in learning Java.  http://pages.cs.wisc.edu/~hasti/cs368/JavaTutorial/
* Difference between Java and C++ Constructor - Interview Question http://javarevisited.blogspot.com/2015/09/difference-between-java-and-c-constructor.html
  * [Why multiple inheritances are not supported in Java](http://javarevisited.blogspot.sg/2011/07/why-multiple-inheritances-are-not.html)
  * [Why String is Immutable or Final in Java](http://javarevisited.blogspot.com/2010/10/why-string-is-immutable-in-java.html)
  * [Top 20 Core Java Interview Questions and Answers asked on Investment Banks](http://javarevisited.blogspot.sg/2011/04/top-20-core-java-interview-questions.html)
  * [When a class is loaded and initialized in JVM - Java](http://javarevisited.blogspot.sg/2012/07/when-class-loading-initialization-java-example.html)
* BTW, if you are from C++ background and looking for a good book to learn Java then check out [Core Java, Volume 1](http://www.amazon.com/Core-Volume-I-Fundamentals-Edition-Series/dp/0137081898) by Cay S. Horstmann.

Exceptions
* https://docs.oracle.com/javase/tutorial/essential/exceptions/index.html
* "Here's the bottom line guideline: If a client can reasonably be expected to recover from an exception, make it a checked exception. If a client cannot do anything to recover from the exception, make it an unchecked exception."

Java Varargs
* http://docs.oracle.com/javase/1.5.0/docs/guide/language/varargs.html

Functional Programming in Java
* [Functional programming: A step backward](http://www.javaworld.com/article/2078610/java-concurrency/functional-programming--a-step-backward.html)
* "The better argument for functional programming is that, in modern applications involving highly concurrent computing on multicore machines, state is the problem. All imperative languages, including object-oriented languages, involve multiple threads changing the shared state of objects. This is where deadlocks, stack traces, and low-level processor cache misses all take place. If there is no state, there is no problem."
* "Unlike imperative code, functional code doesn't map to simple language constructs. Rather, it maps to mathematical constructs."
* "After decades of progress in making programming languages easier for humans to read and understand, functional programming syntax turns back the clock."
  * This goes along with Josh's oft stated point about premature optimization.
  * "premature optimization is the root of all evil" because it leads to less maintainable code
  * For example, how maintainable does this look? http://sebastian-millies.blogspot.de/2015/09/cartesian-products-with-kleisli.html

jOOQ (Java 8 Streams and Functional Programming)
* [Comparing Imperative and Functional Algorithms in Java 8](http://blog.jooq.org/2015/09/17/comparing-imperative-and-functional-algorithms-in-java-8/)
* [How to Use Java 8 Streams to Swiftly Replace Elements in a List](http://blog.jooq.org/2015/04/02/how-to-use-java-8-streams-to-swiftly-replace-elements-in-a-list/)
* [Common SQL Clauses and Their Equivalents in Java 8 Streams](http://blog.jooq.org/2015/08/13/common-sql-clauses-and-their-equivalents-in-java-8-streams/)
* [How to use Java 8 Functional Programming to Generate an Alphabetic Sequence](http://blog.jooq.org/2015/09/09/how-to-use-java-8-functional-programming-to-generate-an-alphabetic-sequence/)

[Lambda Expressions in Java 8](http://www.drdobbs.com/jvm/lambda-expressions-in-java-8/240166764)
* "Recently, functional programming has risen in importance because it is well suited for concurrent and event-driven (or "reactive") programming."
* "You can supply a lambda expression whenever an object of an interface with a single abstract method is expected. Such an interface is called a _functional interface_."
* "In fact, conversion to a functional interface is the only thing that you can do with a lambda expression in Java."
* "The expression `System.out::println` is a _method reference_ that is equivalent to the lambda expression `x -> System.out.println(x)`."
* "A lambda expression has three ingredients:"
  1. "A block of code"
  2. "Parameters"
  3. "Values for the free variables; that is, the variables that are not parameters and not defined inside the code... We say that these values have been _captured_ by the lambda expression."
* "interface methods with concrete implementations (called default methods). Those methods can be safely added to existing interfaces."
* https://docs.oracle.com/javase/tutorial/java/javaOO/lambdaexpressions.html

Singleton in Java
* http://javarevisited.blogspot.in/2012/12/how-to-create-thread-safe-singleton-in-java-example.html
* http://javarevisited.blogspot.gr/2012/07/why-enum-singleton-are-better-in-java.html