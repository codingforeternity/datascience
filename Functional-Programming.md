[Structural Pattern Matching in Java](http://blog.higher-order.com/blog/2009/08/21/structural-pattern-matching-in-java/) (7/5/16)
* "structural pattern matching on algebraic data types. Once you’ve used this feature, you don’t ever want to program without it. You will find this in languages like Haskell and Scala."

[Efficiency of Functional Programming](http://stackoverflow.com/questions/1990464/efficiency-of-purely-functional-programming) (4/12/16)

[OOP is embarrassing](https://www.youtube.com/watch?v=IRTfhkiAqPw&app=desktop) (3/7/16)
* 2:50 - Let data just be data
  * Let actions just be actions (don't nounify verbs)

[MapReduce is not functional programming](https://medium.com/@jkff/mapreduce-is-not-functional-programming-39109a4ba7b2#.c42eic180) (3/2/16)
* map is a trivial concept. It’s basically SELECT from SQL
* reduce in MapReduce is a GROUPBY operator: it groups the output of map by key and applies a second, different map on each (key, [stream of values]) pair
* I should also note that, though classical MapReduce consists of 3 stages (parallel apply, group-by a.k.a. shuffle, and another parallel apply), this is merely an arbitrary restriction of the original implementation.
People quickly realized that you can assemble more complex networks of parallel applications and group-by’s, which you can see in the FlumeJava paper, Spark, Apache Crunch etc., and finally of course in Dataflow.

[Awk, Unix, and functional programming](http://trevorjim.com/awk-unix-and-functional-programming/) (2/6/16)

[OOP is bad](https://www.youtube.com/watch?v=QM1iUe6IofM) (1/26/16)
* procedural programming is ideal
* good discussion of 4 types of programming at beginning of talk (imperative, functional, procedural, and ?)
* 18:08 - Why does OOP not work?  A: encapsulation
* 23:00 - "half-assed encapsulation actually gets us something
  * wrangling the object zoo
  * sub-system hierarchies of objects -- the improper OOP way -- require introductiong "sub-god objects" [fwc - [sobjects"]
* 25:00 - the proper OOP way and the improper OOP way both suck
* 33:00 - Write methods only when the exclusive association with the data type is not in doubt -- main example of this: abstract data types (ADTs), e.g. lists and queues
  * the minute you start hemming and hawing over whether a function has a primary association with a data type is the moment you say "screw it" and just make it a plain function
  * because most things we tend do do in code are cross-cutting concerns-- they don't have special obvious relationships with particular data types
* principles
  1. when in doubt, parameterize (no globals, or shared implicit state)
    * want data access in our programs to flow through the call graph
  2. bundle globals into structs/records/classes
  3. favor pure functions (easier when efficiency is not a priority)
  4. encapsulate (loosely) at the level of namespaces/packages/modules
  5. don't be afraid of long functions (do be afraid of needlessly shared state)
    * of course if you want to execute code from multiple places you have to break things into functions
    * but don't break things into functions merely for documentation purposes -- too much code floating around, too much API to search through
      * plus its tough to name functions/variables well
      * next best thing: make it private or nested function
      * constrain scope of local variables in sub-scopes or nested anonymous funcs
      * what we really want though (that doesn't exist in any language) is an anonymous function that doesn't see any of its enclosing scope (FWC - write a language! or just check out Groovy)
        * but this requires variables from local scope to be passed into functions as params
        * what we really want is a scope-limiter like "use x, y { ... }" that restricts scope to a few local variables without having to pass them in
        * such blocks should return a value also
        * makes it clear that this is a piece of code used only in this one place (and don't have to give it a name)
* 43:55 - books to not read

[3 Reasons why You Shouldn’t Replace Your for-loops by Stream.forEach()](http://blog.jooq.org/2015/12/08/3-reasons-why-you-shouldnt-replace-your-for-loops-by-stream-foreach/) (12/11/15)

[From 2001: "I will eat a week's pay if OOP is still in vogue in 2015."](https://mobile.twitter.com/fernozzle/status/672133043037929472) (12/2/15)

[Beware of Functional Programming in Java](http://blog.jooq.org/2015/11/10/beware-of-functional-programming-in-java/) (11/13/15)
* The lambda style will encourage using higher-order functions in Java, all over the place. Which is generally good. But only when the higher-order function is a static method, whose resulting types will not enclose any state.
* So, be careful, and follow this rule: (“Pure”) Higher order functions MUST be static methods in Java!

[Google Guava (Java library): Functional Explained](https://github.com/google/guava/wiki/FunctionalExplained) (11/9/15)

Also see [[Java Notes]] for functional programming in Java.

Good Functional Programming Article
* https://github.com/fcrimins/fcrimins.github.io/wiki/Better-Java
* There is no `User` but it’s very likely there is `SignupUser`. There is no `Order` but you definitely can deal with `PlaceOrder`. And when you see classes ending with `Manager`--just run.

[Why do some functional programmers criticize design patterns in OOP languages as a sign of language deficiency](https://www.quora.com/Why-do-some-functional-programmers-criticize-design-patterns-in-OOP-languages-as-a-sign-of-language-deficiency-while-Monad-is-also-a-design-pattern)

[Functional Programming in Python](http://www.pysnap.com/functional-programming-in-python/)
* "Q2. Can we do Functional Programming in Python? Ans. Yes, we can do FP in Python but its NOT a “Pure” functional language and still youcan program in functional style but be careful. The only reason to ever do so is for readability. If the algorithm is more elegantly expressed functionally than imperatively, and it doesn’t cause performance problems (it usually doesn’t), then go right ahead."
* [The fate of reduce() in Python 3000](http://www.artima.com/weblogs/viewpost.jsp?thread=98196)
  * "So now reduce(). This is actually the one I've always hated most, because, apart from a few examples involving + or *, almost every time I see a reduce() call with a non-trivial function argument, I need to grab pen and paper to diagram what's actually being fed into that function before I understand what the reduce() is supposed to do. So in my mind, the applicability of reduce() is pretty much limited to associative operators, and in all other cases it's better to write out the accumulation loop explicitly."

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
