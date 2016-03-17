[Microservices Theory](http://martinfowler.com/articles/microservices.html) (3/4/16)
* "To start explaining the microservice style it's useful to compare it to the monolithic style: a monolithic application built as a single unit. Enterprise Applications are often built in three main parts: a client-side user interface (consisting of HTML pages and javascript running in a browser on the user's machine) a database (consisting of many tables inserted into a common, and usually relational, database management system), and a server-side application"
* "Design for Failure"

[Bruno Rocha: Microservices with Python, RabbitMQ and Nameko](http://brunorocha.org/python/microservices-with-python-rabbitmq-and-nameko.html) (3/4/16)

[RESTful API Best Practices and Common Pitfalls](https://medium.com/@schneidsDotNet/restful-api-best-practices-and-common-pitfalls-7a83ba3763b5#.sii1bf1qe) (2/25/16)
* "Respect the change management process. Avoid introducing break changes to existing endpoints that people are using."
* Also discusses asynchronous query response.

[How RESTful is Your API?](http://www.bitnative.com/2012/08/26/how-restful-is-your-api/) (2/25/16)
* "Pragmatic REST"
* Really good, succinct article that specifically covers REST sans discovery and what that means for an API.
* Also discusses pragmatic versioning in place of discovery.

[Rescuing REST From the API Winter](http://intercoolerjs.org/2016/01/18/rescuing-rest.html) (2/1/16)
* Basically, JSON-based REST isn't REST because it doesn't have native support for links.
* Schema/Structure needs to be discovered at runtime which is what HATEOS is for.
* HATEOAS - Hypermedia as the Engine of Application State.
* HATEOAS is a bigger part of REST than anyone ever really realized.

[Testing REST clients](https://www.kenneth-truyers.net/2016/01/29/testing-rest-clients/) (2/1/16)

[REST Introduction](http://www.infoq.com/articles/rest-introduction) 1/9/16
* The beauty of the link approach using URIs is that the links can point to resources that are provided by a different application, a different server, or even a different company on another continent
* the representations of a resource should be in standard formats -- if a client "knows" both the HTTP application protocol and a set of data formats, it can interact with any RESTful HTTP application in the world in a very meaningful way
* i.e. the idea that every resource should respond to the same methods. But REST doesn't say which methods these should be, or how many of them there should be
* HTTP "instantiates" the REST uniform interface with a particular one, consisting of the HTTP verbs

[REST Misconceptions Video](http://www.infoq.com/presentations/rest-misconceptions) (1/9/16)
* UriApi 25:00
  * As a client I have to have some knowledge about the structure of the URIs so that I can build them myself
  * Code is full of getting a customer and then appending slash-something to get to the orders--to get to the orders
  * URIs that I document publicly now become the API.  I've documented a UriAPi
  * It's perfectly fine not to do Rest, I'm just a fan of calling tings what they are--of useful terminology--so don't call it rest if it's not rest--that is not restful.
  * one of the reasons this is not restful is because assumptions about server details become facts
  * there's a an assumption here that if i have a customer and i append "orders" that i get a list of orders for that customer
  * Client is now relying on exact URI structure which is something i would like to avoid if in any way possible
  * I hate version number URIs--everybody does them--i don't mind, i hate them nonetheless.  you change the uris for no good reason
* versioning 27:00
  * ok to version data, documentation, and formats--just not APIs
  * use version numbers in apis to version the resource itself
  * create new resources for new aspects and reserve space for links (so new resources can be discovered from existing ones)
* Postel's Law 36:20
  * "be liberal in what you accept and conservative in what you send"
  * high chance of what you send will be recognized by others
* Client and Server rules 38:00
  * more dynamic 41:00
  * change something on server side and new client doesn't have to be rolled out again b/c it learns about it at runtime
  * e.g. identifiers pointing to suburls
  * pass state transition information from server to client (43:40)
  * Rule #1: Don't have clients build URIs using string concatenation (45:15) ...instead: provide recipes
* Summary (53:30)
  * Link context over pretty uris
  * hypermedia over uri apis
  * hypermedia flows over static links
  * generalized formats over services

[A Brief Introduction to REST](http://www.infoq.com/articles/rest-introduction) (12/21/15 and 1/8/16)

[Dr. Dobb's - RESTful Web Services: A Tutorial](http://www.drdobbs.com/web-development/restful-web-services-a-tutorial/240169069) (1/8/16)
* Don't use query parameters (except for "parameters to an operation that needs the data items")

[Is REST Best in a Microservices Architecture?](http://capgemini.github.io/architecture/is-rest-best-microservices/) (12/21/15)

[Why REST is important even for your internal API](https://medium.com/@_reneweb_/why-rest-is-important-even-for-your-internal-api-ab08a40d01d3#.o8uyilkxr) (12/9/15)