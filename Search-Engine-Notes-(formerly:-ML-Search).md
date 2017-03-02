#### [PageRank on Wikipedia](https://en.wikipedia.org/wiki/PageRank)
* "The PageRank algorithm outputs a probability distribution used to represent the likelihood that a person randomly clicking on links will arrive at any particular page... It is assumed in several research papers that the distribution is evenly divided among all documents in the collection at the beginning of the computational process."  The algorithm iterates to update the distribution a bit in each pass.
  * This still has the problem that it is based on "random clicking" as opposed to "clicking according to some prior distribution."  Could a MCMC approach be used here to improve the "random"/uniform prior?
  * "The formula uses a model of a random surfer who gets bored after several clicks and switches to a random page. The PageRank value of a page reflects the chance that the random surfer will land on that page by clicking on a link. It can be understood as a Markov chain in which the states are pages"

#### [How to build a search engine from scratch](https://www.quora.com/How-to-build-a-search-engine-from-scratch) (2/2/17)
* See Agapiev's answer in particular.
* [CommonCrawl](http://commoncrawl.org/big-picture/frequently-asked-questions/) - the web, crawled for us

#### Another problem with Google search (10/17/15)
* It leads to a lot of anecdotal evidence.
* Which just happens to be a massive percentage of the evidence for and against baby products of all kinds.
* Take for example: http://www.candokiddo.com/news/rocknplay
* This is a well read site.  But it's nothing but anecdotal.
* It would be great to see a reliability score for it.

#### Email: Watson for reviews (10/6/15)
* Filter reviews and return a re-aggregated list of relevant ones, according to an individual's preferences and their validity.
* Use collaborative filtering (CF) for ailments/problems rather than human-understandable problems.  I.e., use our 8 ailments and CF back to EMRA problems.
* Use Watson for any user-specific record, like EMRs.

#### Email: "Internet decay"
* http://www.vox.com/2015/8/6/9099357/internet-dead-end
* "What links these seemingly dissimilar stories is a very basic fear — the idea that the internet as we knew it, the internet of five or 10 or 20 years ago, is going away as surely as print media, replaced by a new internet that reimagines personal identity as something easily commodified, that plays less on the desire for information or thoughtfulness than it does the desire for a quick jolt of emotion."
* I.e. like reality TV.  So why can't machine learning (e.g. Watson) correct the internet back to it's previous state?  People experience the internet through search.  If the searching mechanism doesn't point to reality TV, then [reality TV] might as well not exist, as far as a user is concerned.

#### Email: Phonetics search and spellcheck (9/17/15)
* Why does typing into a search box only suggest word completions that have the same characters as I'm typing?
* Why not suggest word completions that match the phonetics of what I'm typing?
* This might be a poor example, but I was searching for the Neue Gallerie, but I couldn't remember how it was spelled: Noye Gallery?
* Also, suggested spelling don't seem to go beyond having a single character mispelled, nor do they seem to know about how close characters are to each other on the (phone) keyboard.
* Spellcheck should work the same way, using phonetics.

#### Idea: Watson seems to have required rules to play _Jeopardy!_
* I.e. rules had to be programmed such as the relationship of the category headings to the questions.
* Why couldn't the relationship have been learned?
* I suppose spatial relatedness (rows/columns) would have to be learned first.
* But wouldn't this be the similar to learning faces with a hierarchical network?
  * pixels -> lines -> motifs-> sub-objects -> objects
  * spatial relations -> rows/columns -> categories -> question hints

#### Email: Is there a watson search engine (6/12/15)
* And if not why? It should be configurable along any dimension, eg trustworthiness, that it's user specifies
* http://searchengineland.com/goodbye-blekko-search-engine-joins-ibms-watson-team-217633
* http://searchengineland.com/google-forecloses-on-content-farms-with-farmer-algorithm-update-66071
* Google has a "search quality team"
* Reliability of results isn't exactly what this would want to target, but rather it should target "scientific basis."  Which content has basis in science vs. not?  Can this be assessed using NLP?  AI?  Comparison to scientific research? Can NLP be used to distinguish between what is science and what is hearsay?
* Also see: FactCheck.org and SciCheck.org, the first (and maybe the second) of which has a page that lists "trusted sources" such as the CBO.
* Such a search engine could be constructed to _cross-check_ (potential name?) against any set of sources.  Scientific sources could be one thing to cross-check against.  PDFs could be another.  Religious sources could be another. (set-differences between sources might be interesting: <science> - <religion> = <?>)