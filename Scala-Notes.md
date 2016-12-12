[Unboxed union types via the Curry-Howard isomorphism](http://milessabin.com/blog/2011/06/09/scala-union-types-curry-howard/) (12/2/16)
* unions, like in C++ or like Either[A,B] in Scala (which is apparently considered "boxed"), not like tuples

[Scala by Example (Martin Odersky)](http://www.scala-lang.org/docu/files/ScalaByExample.pdf)
* includes nice explanation of tail recursion (see gcd/factorial example, p. 18)
  * "A production quality Scala implementation is therefore only required to re-use the stack frame of a directly tail-recursive function whose last action is a call to itself."
* "Scala uses call-by-value (FWC - parameters are evaluated before calling) by default, but it switches to call-by-name (FWC - unevaluated references to parameters are passed and only evaluated upon use) evaluation if the parameter type is preceded by =>" (p. 14)
* Indeed, the anonymous function `(x1: T1, ..., xn: Tn) => E` is equivalent to the block `{ def f (x1: T1, .., xn: Tn) = E ; f _ }` where `f` is  fresh  name  which  is  used  nowhere  else  in  the  program.   We  also  say, anonymous functions are 'syntactic sugar'.
* Currying (FWC - multiple parameter lists reduce to curried functions) "The style of function-returning functions is so useful that Scala has special syntax for it." (p. 24)
  * "Generally, a curried function definition `def f (args_1) ... (args_n) = E` expands to `def f args_1) ... (args_{n-1}) = (args_n) => E`"
  * Also note, that functions that do not take any parameters are a special case of this.
* Really neat example of _fixed point functions_: the square root function can be expressed as follows `def sqrt(x: Double) = fixedPoint(averageDamp(y => x/y))(1.0)` (p. 27)
* "Note that, unlike in Java, redefining definitions need to be preceded by an **override** modifier." (p. 33) [FWC - this is something that a lot of OO languages are missing; it prevents bugs]
* "Dynamic  method  dispatch  is  analogous  to  higher-order  function  calls.    In  both
cases, the identity of code to be executed is known only at run-time. This similarity
is not just superficial.   Indeed,  Scala represents every function value as an object
(see Section 8.6)." (p. 36)

[Ways to pattern match generic types in Scala](http://www.cakesolutions.net/teamblogs/ways-to-pattern-match-generic-types-in-scala) (9/21/16)
* See [Type Tags](http://www.cakesolutions.net/teamblogs/ways-to-pattern-match-generic-types-in-scala#type-tags) section in particular

[Scala Mixins](http://www.scala-lang.org/old/node/117) (9/21/16)
* `class X extends B`
* `class Y extends B`
* `class D extends X with Y`
* Only the members of Y that are explicitly defined inside Y are "mixed in" to D.
* D receives all of its B functionality via X.
* This prevents the diamond problem of multiple class inheritance.

Q: Why are the elements of [Scala tuples](http://www.scala-lang.org/files/archive/spec/2.11/03-types.html#tuple-types) indexed starting from 1 rather than 0?<br/>
A: Because they aren't indexes.  They're elements in a basket--class members, fields--that just don't have names (yet).  Thinking about them as indexes is the wrong way of thinking about them.  So starting from 1 discourages this way of thinking. [FWC]

[Scala Implicits](http://googlyadventures.blogspot.com/2016/03/today-i-taught-someone-scala-implicits.html) (9/14/16)
[Implicit Conversions and Parameters](http://www.artima.com/pins1ed/implicit-conversions-and-parameters.html)
* **Implicit Receiver Conversion**: "Implicit conversions also apply to the receiver of a method call, the object on which the method is invoked. This kind of implicit conversion has two main uses. First, receiver conversions allow smoother integration of a new class into an existing class hierarchy. And second, **they support writing domain-specific languages (DSLs)** within the language."
* This "rich wrappers" pattern is common in libraries that provide syntax-like extensions to the language, so you should be ready to recognize the pattern when you see it. *Whenever you see someone calling methods that appear not to exist in the receiver class, they are probably using implicits.* Similarly, if you see a class named RichSomething, e.g., RichInt or RichString, that class is likely adding syntax-like methods to type Something.
* As you can now see, these rich wrappers apply more widely, often letting you get by with an internal DSL defined as a library where programmers in other languages might feel the need to develop an external DSL.

Scala IDE (Eclipse plugin)
* "more than one scala library found in the build path" - http://stackoverflow.com/questions/13682349/how-to-use-just-one-scala-library-for-maven-eclipse-scala
* "is cross-compiled with an incompatible version of Scala - http://scala-ide.org/docs/current-user-doc/faq/index.html
* "plugin execution not covered by lifecycle configuration" - https://www.eclipse.org/m2e/documentation/m2e-execution-not-covered.html
  * this means that Eclipse's "lifecycle configuration" isn't going to run this Maven plugin because they aren't hooked up to each other

[Abstract Type Members versus Generic Type Parameters in Scala](http://www.artima.com/weblogs/viewpost.jsp?thread=270195) (9/20/16)
* "My observation so far about abstract type members is that they are primarily a better choice than generic type parameters when you want to let people mix in definitions of those types via traits. You may also want to consider using them when you think the explicit mention of the type member name when it is being defined will help code readability."

[Scala Tutorial](http://www.tutorialspoint.com/scala/)

[Scala on Wikipedia](https://en.wikipedia.org/wiki/Scala_%28programming_language%29) (6/7/16)
* "By convention, a method should be defined with empty-parens when it performs side effects."