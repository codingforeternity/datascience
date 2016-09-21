[Scala Mixins](http://www.scala-lang.org/old/node/117) (9/21/16)
* `class X extends B`
* `class Y extends B`
* `class D extends X with Y`
* Only the members of Y that are explicitly defined inside Y are "mixed in" to D.
* D receives all of its B functionality via X.
* This prevents the diamond problem of multiple class inheritance.

[Abstract Type Members versus Generic Type Parameters in Scala](http://www.artima.com/weblogs/viewpost.jsp?thread=270195) (9/20/16)
* "My observation so far about abstract type members is that they are primarily a better choice than generic type parameters when you want to let people mix in definitions of those types via traits. You may also want to consider using them when you think the explicit mention of the type member name when it is being defined will help code readability."

[Scala Tutorial](http://www.tutorialspoint.com/scala/)

[Scala on Wikipedia](https://en.wikipedia.org/wiki/Scala_%28programming_language%29) (6/7/16)
* "By convention, a method should be defined with empty-parens when it performs side effects."