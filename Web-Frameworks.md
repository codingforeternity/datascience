#### [Twitter Moving to JVM](https://www.infoq.com/articles/twitter-java-use)
* FWC - **really nice overview of Twitter architecture**
* "One of the overall observations one can make from looking at Twitter's architecture is that many of the design decisions are admirably pragmatic."
* "Twitter also uses HDFS in Hadoop extensively for off-line computation"
* "Communication between front-end and back-end services uses the Facebook developed Thrift as the RPC mechanism, and **JSON over REST** [FWC - more evidence they aren't compatible] as the public RPC"
* "As the back-end code is being pulled towards the JVM, the front end client code, in common with many contemporary web based applications, is gradually making heavier and heavier use of browser-based **JavaScript**."
* "I wouldn't say that Rails has served as poorly in any way, it's just that we outgrew it very quickly."
* "Rather, the move to JVM is driven as much by a need for better developer productivity as it it for better performance."
  * "The primary driver is honestly encapsulation, so we can iterate faster as a company. Having a single, monolithic application codebase is not amenable to quick movement on a per-team basis."
* "I would say about half of the productivity gain is purely because of accumulated technical debt in the search Rails stack. And *the other half [of the productivity gain] is that, as search has moved into a Service Oriented Architecture and exposes various APIs,* **static typing** *becomes a big convenience in enforcing coherency across all the systems. You can guarantee that your dataflow is more or less going to work*, and focus on the functional aspects. Whereas for something like building a web page you don't want to recompile all the time, you don't really want to worry about whether in some edge condition you are going to get a type you didn't expect. But *as we move into a light-weight Service Oriented Architecture model, static typing becomes a genuine productivity boon*. And **Scala gives you the same thing**."

#### [The state of Ruby and Rails: Opportunities and obstacles; Ruby on Rails still shines for Web development, but not for speed or scalability](http://www.javaworld.com/article/2945136/scripting-jvm-languages/the-state-of-ruby-and-rails-opportunities-and-obstacles.html) (from 6/6/15)
* FWC - **probably the best written and (seemingly) least biased article I've read**
* "Few sites would experience the same extreme demand as Twitter, so not every Rails-driven site is a candidate for a ground-up rewrite."
* "In light of this, one possible reason for the growth in demand for Ruby on Rails and Ruby is to preserve or maintain -- or even replace -- existing Ruby or Rails infrastructure, rather than building new objects with it."

#### [Ruby on Rails vs Groovy on Rails](https://acadgild.com/blog/ruby-on-rails-vs-groovy-on-rails/) (from 12/15)
* "So choosing among the framework depends upon what skill set you have. If you want to maximize advantage of your existing Java skills or have developers experienced in Java than go for Grails. But if you are more accustomed with Ruby, HTML, CSS and JavaScript then go for Rails."

#### [What are the Benefits of Ruby on Rails? After Two Decades of Programming, I Use Rails](https://www.toptal.com/ruby-on-rails/after-two-decades-of-programming-i-use-rails) (from 3 years ago)
* "Rails has been around the block. In a hipster kind of way, it’s not even that cool anymore."

#### [The Languages And Frameworks You Should Learn In 2016](http://tutorialzine.com/2015/12/the-languages-and-frameworks-you-should-learn-in-2016/)
* In 2016, things are different. You pay nothing to set up a web page with Weebly or Strikingly or Wix or Squarespace. The cost is zero.  You just need to pick colors for your site these days. Most of them will have a widget that can take user input and send that information your way. Do you want a blog? you can create a free one on Ghost or Tumblr or Jekyll or LinkedIn or Medium. You need a blog and a website together? WordPress will do the trick. **If I started a business today, I would host it on Bluehost or GoDaddy with WordPress underneath**. In fact, This site runs on Bluehost with WordPress underneath. Domains are cheap today. Once your site is ready, you can point your domain to sites I mentioned above and you have an online business. We are reaching maturity in web applications now. Freelancing is not the same anymore. There are out of the box solutions for most things you would hire a developer for back in 2002. Need to sell shirts? Go to Shopify. Need to sell e-books or any software? Go to amazon or sell it on your site using stripe or gumroad or e-junkie or e-bay. Need to roll out your own design? You have cheap options in Fiverr or 99designs or Upwork. With dedicated graphic design firms, you get premium quality. With dribbble, You get quality work and a ton of options.

#### [Spring vs. Rest of world](http://springtutorials.com/spring-framework-vs-rest-of-the-world/)
* "Play builds on Scala and looks promising. You can use either Java or Scala to build with it. I am new to Scala as well as Play. I know that Play Framework versions aren’t backward compatible."
* "Grails builds on Groovy and allows you to build your code fast. It is the groovy version of ruby on rails. I see that latest Grails builds on Spring 4 and Spring Boot. As a Java Developer, you will have to learn Groovy. You may need to understand GSP – groovy server pages vs JSP concept. You may need to understand  GORM – Groovy ORM vs Java ORM. This is not a steep learning curve but takes time regardless."

#### [Which Technologies Do Startups Use? An Exploration of AngelList Data](https://codingvc.com/which-technologies-do-startups-use-an-exploration-of-angellist-data)

#### [Rails vs Django vs Play: Battle of frameworks](http://www.diogonunes.com/blog/rails-vs-django-vs-play-frameworks/)
* Python/Django - "developing on Django is an exhausting symphony of sighs"
* "[Zentasks](http://www.playframework.com/documentation/2.3.x/JavaGuide1) is a tutorial that guides you on the process of creating a website for task management"
* "So on Play I had to inherit a base template and implement every empty block – hence, the 'baggage' of the base template always came attached. Rails uses composition"
* Java/Play - "perfect if you know Java and want to develop web applications, it just works."
* Ruby/Rails - "Rails enables fast development great for prototyping. The more you use it the more you love it."

#### [Is Play framework scalable while Ruby on Rails isn't?](https://www.quora.com/Why-is-Play-framework-scalable-but-Ruby-on-Rails-isnt)
* "Rails is a great prototyping application. It's fast, developers can build great applications on top of it. But you don't have to build your entire application on top of it. **You're going to be able to build a lot faster with Rails than you would with Play**. You're going to be able to convert processes that are slow in Rails into SOA's."

#### [Why do people use Heroku when AWS is present? What's distinguishing about Heroku](http://stackoverflow.com/questions/9802259/why-do-people-use-heroku-when-aws-is-present-whats-distinguishing-about-heroku)
