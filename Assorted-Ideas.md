#### Train a NN to predict a word based on its definition (from a dictionary)
* or use such a mapping with negative sampling, to construct word embeddings
* aren't such things already trained on Wikipedia though?

#### Machine Learning of Neural Net Architecture
* Eg detect the effectiveness of individual nodes, and allow them to die off or multiply based on how effective they are.
* Allow tumors to form similar to how LSTMs have cell state plus regular CNN. How might a CNN detect that it needs a cell state for example?
* This all still requires a predefined cost function. ie, what to learn. How could that be learned? seems like theres a finite set of these tho, squared error, cross entropy, logisti
* This idea seems similar to dropout, but deletion is only one of the forms of mutation, and plus its mainly done to prevent overfitting, not to learn the architecture of the net
* [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) (in TensorFlow)

#### [[Linguistic Networks]]

#### Chem Collection (8/17/16)
* There are tons of fertilizers flowing into the Gulf of Mexico out the mouth of the Mississippi River.
* Harvest them to (1) remove the chemicals and (2) purify the Gulf.

#### My reality show idea (4/27/16)
* Talkshow is texting in public: http://kottke.org/16/04/talkshow-is-texting-in-public

#### A programming language...
* ...where every function can think for itself, decide when it wants to run based on patterns of use
* every menu option can show itself when it wants (e.g. if the user navigates to it many times, then it can make itself easier to navigate to)
* e.g. I so often send email messages w/out a body, and I don't want my phone to ask me if that's really what I want to do every time

#### All software should adapt
* this means building on top of an *adaptive platform*
* i go to 3 options in every eclipse pop-up menu
* i only "share" from my phone to gmail and text messaging
* when i type "joseph" i typically only mean one person
* expected-next-word language modeling is where this might work correctly, but this should be a feature of something lower down than nlp that should percolate up to all these other places also

#### [*How the Internet Became Commercial* - Counter argument to government invented the internet](http://marginalrevolution.com/marginalrevolution/2015/10/how-the-internet-became-commercial.html) (4/15/16)

#### [Elasticsearch as a Time Series Database - Getting data in with StatsD](https://www.reddit.com/r/programming/comments/46pgpz/elasticsearch_as_a_time_series_database_getting/) (2/20/16)

#### [The UNIX School: awk & sed tutorial](https://www.reddit.com/r/programming/comments/44gnm1/the_unix_school_awk_sed_tutorial/) (1/25/16)

#### [Getting Started with Git (Channel 9)](https://www.reddit.com/r/programming/comments/42n0s6/getting_started_with_git_channel_9/) (1/25/16)

#### [Rust and the Blub Paradox](http://www.jonathanturner.org/2016/01/rust-and-blub-paradox.html) (1/28/16)

#### [Manager Strategies: People can read their manager's mind](http://yosefk.com/blog/people-can-read-their-managers-mind.html) (1/19/16)

#### [Finely-Grained Management](http://mikehadlow.blogspot.cl/2014/06/heisenberg-developers.html) (2/7/16)
* Heisenberg Developers: You can not observe a developer without altering their behavior.

#### [Nice git cheatsheet](http://luisbg.blogalia.com//historias/76017) (12/7/15)

#### [Overconfidence](http://econlog.econlib.org/archives/2015/11/ram_on_overconf.html) (1/2/16)
* RCTs for public policy. Why is introspection all that is currently required?  People accept that the human body is a complex system, requiring RCTs, but they don't accept that an economy is.

#### [Why GNU grep is fast](https://lists.freebsd.org/pipermail/freebsd-current/2010-August/019310.html) (12/10/15)

#### [Why use REST inside the company?](https://medium.com/@_reneweb_/why-rest-is-important-even-for-your-internal-api-ab08a40d01d3) (12/9/15)

#### Email: Use anomaly detection for SIDS (10/6/15)
* SIDS is an anomaly detection scenario because there are very few cases (e.g. 0.1%).

#### Email: Don't regulate pay (10/6/15)
* It's been tried and it doesn't work.  Pay in finance just keeps going up.
* Pay that's earned should be taxed less than pay that isn't.  Another way to think about this is pay that's acquired by chance should be taxed more than pay that isn't.
* So then we just need a metric of chance for each industry/sector/other set of factors.  One idea: the ratio of successful to unsuccessful members of an industry.

#### Book: [The Warmth of Other Suns](http://kottke.org/15/10/the-warmth-of-other-suns) (10/10/15)
* "The Warmth of Other Suns is about the Great Migration, the mass movement of African Americans from the Southern US to the Northeast, Midwest, and West between 1910 and 1970. During that time, roughly 6 million African Americans moved north and west to escape Jim Crow laws, discrimination, low wages, the threat of physical violence & death, and everyday humiliation & lack of freedom in the South."
* Sound like anything going on today, e.g. Mexicans perhaps.  So why isn't it being celebrated like it is in this book?
* What's more, hasn't this been a constant in the history of America?

#### Charter Prisons (Email: "Why don't", 10/17/15)
* There are private prisons but are there non profit prisons?  Charter prisons?
* Wouldn't it be possible to massively increase prisoner welfare by just attaching GPS and/or listening/recording devices to their ankles that would set off alarms if tampered with.
* This could protect them against other prisoners and guards.
* Could even make it voluntary, if prisoners objected on privacy grounds--not that there's any of that in prison--not that I know.

#### Facial Biases (10/20/15)
* Email: Thin upper lip bias
* It seems like folks with thin upper lips are assumed to be more dishonest or shifty than others.  I had this idea while watching episode 39 of House of Cards.
* What other kinds of facial feature biases are there?  Seems easy enough to learn.  Would mean understanding features more than lines though.
* There are obviously race and sex biases.  And there are biases against age and height.  There are surely biases against all sorts of things we aren't aware of.

#### 2 Problems with Maps Apps
  1. They aren't predictive.  E.g. what will the traffic be like on a summer Friday afternoon leaving the city.  I.e. they aren't forward looking.  But they're not backwards looking either.  If there's unexpected traffic (e.g. due to an accident) and all of a sudden traffic begins to flow freely again, that should be accounted for very quickly.
  2. The graph being traversed shouldn't have intersections as its nodes; rather nodes should be points before and after intersections because it might take longer to turn left than go straight.  Take for example, exiting off of Rte 91 North onto Rte 84 East.

#### Email: The only way to learn is to mess up (10/1/15)
* Even theory-based learning requires the testing of theories
* i.e. using controls where on one side of the "line of control" lies the theory and the other side is "messed up" (i.e. the anti-theory)

#### 9/25/15
DATA!
* IDEA: http://www.data.gov/

#### Email: Create a company...
* ...that rather than trying to always make things better, like google, tries to prevent things from becoming worse, by stifling innovation for example.  Every cost against rent seeking

#### Email: ML for aggregated knowledge
* On the Web for example. How many sites support a particular point of view
* Internetal Consensus!

#### Idea: Feature Engineering for input and output features
* Not only do we want systems to learn the input features automatically, we should also want them to learn the output features automatically.
* We want them to learn what types of questions might be asked.  They should act as a sort of database query language.
* E.g. given data.gov, we should be able to ask questions about any entity (the economy, specific companies or countries) that might have a relationship to that sort of data

#### Email: data support via ML for spoken/written words (9/17/15)
* People say all sorts of shit, but its very rarely backed up by data.  It would be really cool if given some statement, one could use ML to look for support for that statement in all the data.

#### Email: back data out of published results (9/17/15)
* just like in the ML course where features are backed out of ratings (low rank matrix factorization)
* if data could be backed out of published results, then the data could be combined with data from other studies and "meta-data studies" could be born

#### Email: ML for myths
* Eg diets
* Things with tons of unproven "knowledge"
* Can myths/diets be debunked using ML?

#### Email: ML for understanding babies
* How many theories are there?  How many don't have any supporting evidence--like psychology?
* Understand what their cries and motions mean

#### Email: Babies whack themselves in their faces (10/1/15)
* Because they're learning, just like back propagation.

#### Email: Newest cyber threat will be data manipulation, US intelligence chief says | Technology | The Guardian
* ML cyber security. All data must be publicly confirmable--like with Bitcoin.
* http://www.theguardian.com/technology/2015/sep/10/cyber-threat-data-manipulation-us-intelligence-chief

#### Email: Did the Chinese stock market crash BECAUSE US unemployment has dropped so low? Ie risk of rising rates? (9/17/15)
* I.e. contrary to what the following post suggests
* [Why the Federal Reserve Decided to Wait a Few Months Before Messing With the Economy](http://www.slate.com/blogs/moneybox/2015/09/17/federal_reserve_september_decision_janet_yellen_decides_not_to_mess_with.html)