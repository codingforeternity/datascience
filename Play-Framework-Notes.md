### [Main Concepts for Scala](https://www.playframework.com/documentation/2.5.x/ScalaHome)

#### [Forms](https://www.playframework.com/documentation/2.5.x/ScalaForms)

#### [Templates](https://www.playframework.com/documentation/2.5.x/ScalaTemplates)
* "A template is like a function, so it needs parameters, which must be declared at the top of the template file."
* [Template Common Use Cases](https://www.playframework.com/documentation/2.5.x/ScalaTemplateUseCases)
* `views.html.[OptionalDir.]index` is a Scala class generated by Twirl from the `views/[OptionalDir/]index.scala.html` template (note how `views.html.index(testRelativity)` is called in Application.scala

#### [Session and Flash scopes](https://www.playframework.com/documentation/2.5.x/ScalaSessionFlash)
* "keep data across multiple HTTP requests"
* 

#### [Cookies](https://www.playframework.com/documentation/2.5.x/ScalaResults#setting-and-discarding-cookies)

#### [Routing](https://www.playframework.com/documentation/2.5.x/ScalaRouting)

#### Database
* `heroku config` will show the `DATABASE_URL` connection string
* but it has to be set locally (as an environment variable) to this: `export DATABASE_URL=postgres://postgres:password@localhost:5432/default`
* http://stackoverflow.com/questions/35635485/error-running-heroku-locally-using-the-scala-example

#### [Actions, Controllers and Results](https://www.playframework.com/documentation/2.5.x/ScalaActions)
* "A Controller is nothing more than an object that generates Action values. Controllers can be defined as classes to take advantage of Dependency Injection or as objects."
* "Note: Keep in mind that defining controllers as objects will not be supported in future versions of Play. Using classes is the recommended approach."
* Use the `Action` companion object to construct `Action` instances, all of which return `Result`s.
* TODO
  * `def fredtest(name: String) = TODO`
* `Ok(out)` is equivalent to:

```scala
Result(
  header = ResponseHeader(200, Map.empty),
  body = HttpEntity.Strict(ByteString(out), Some("text/plain"))
)
```