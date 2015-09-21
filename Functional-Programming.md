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
