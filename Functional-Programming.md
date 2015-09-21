Also see [[Java Notes]] for functional programming in Java.

Good Functional Programming Article
* https://github.com/fcrimins/fcrimins.github.io/wiki/Better-Java
* There is no `User` but it’s very likely there is `SignupUser`. There is no `Order` but you definitely can deal with `PlaceOrder`. And when you see classes ending with `Manager`--just run.

[Why do some functional programmers criticize design patterns in OOP languages as a sign of language deficiency](https://www.quora.com/Why-do-some-functional-programmers-criticize-design-patterns-in-OOP-languages-as-a-sign-of-language-deficiency-while-Monad-is-also-a-design-pattern)

[Functional Programming in Python](http://www.pysnap.com/functional-programming-in-python/)
* "Q2. Can we do Functional Programming in Python? Ans. Yes, we can do FP in Python but its NOT a “Pure” functional language and still youcan program in functional style but be careful. The only reason to ever do so is for readability. If the algorithm is more elegantly expressed functionally than imperatively, and it doesn’t cause performance problems (it usually doesn’t), then go right ahead."
* [The fate of reduce() in Python 3000](http://www.artima.com/weblogs/viewpost.jsp?thread=98196)
  * "So now reduce(). This is actually the one I've always hated most, because, apart from a few examples involving + or *, almost every time I see a reduce() call with a non-trivial function argument, I need to grab pen and paper to diagram what's actually being fed into that function before I understand what the reduce() is supposed to do. So in my mind, the applicability of reduce() is pretty much limited to associative operators, and in all other cases it's better to write out the accumulation loop explicitly."