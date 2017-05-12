#### [Google's Optical Character Recognition (OCR) software works for 248+ languages]
(https://opensource.com/life/15/9/open-source-extract-text-images) (5/11/17)
* Google's OCR is probably using dependencies of [Tesseract](https://en.wikipedia.org/wiki/Tesseract_(software)), an OCR engine released as free software, or [OCRopus](https://en.wikipedia.org/wiki/OCRopus), a free document analysis and optical character recognition (OCR) system that is primarily used in Google Books.
* [Document layout analysis](https://en.wikipedia.org/wiki/Document_layout_analysis)
  * "It is a common assumption in both document layout analysis algorithms and optical character recognition algorithms that the characters in the document image are oriented so that text lines are horizontal. Therefore, if there is skew present then it is important to rotate the document image so as to remove it.  *It follows that the first steps in any document layout analysis code are to remove image noise and to come up with an estimate for the skew angle of the document.*"
* Tesseract
  * "Since version 3.00 Tesseract has supported output text formatting, hOCR[9] positional information and page-layout analysis."
  * "**Tesseract is suitable for use as a backend and can be used for more complicated OCR tasks including layout analysis by using a frontend such as OCRopus.**"
  * "Tesseract's output will have very poor quality if the input images are not preprocessed to suit it: Images (**especially screenshots**) must be scaled up such that the text x-height is at least 20 pixels,[13] any rotation or skew must be corrected or no text will be recognized, low-frequency changes in brightness must be high-pass filtered, or Tesseract's binarization stage will destroy much of the page, and dark borders must be manually removed, or they will be misinterpreted as characters"
* OCRopus ([[OCRopus installation and examples]])
  * "A free document layout analysis and OCR system, implemented in C++ and Python and for FreeBSD, Linux, and Mac OS X. This software supports a plug-in architecture which allows the user to select from a variety of different document layout analysis and OCR algorithms"
* [OCRFeeder](https://en.wikipedia.org/wiki/OCRFeeder)
  * "OCRFeeder is an optical character recognition suite for GNOME, which also supports virtually any command-line OCR engine, such as CuneiForm, GOCR, Ocrad and Tesseract."
* pyocr
  * Mentioned on [Quora](https://www.quora.com/What-are-the-best-open-source-OCR-libraries)

#### [A short guide to learn NNs](https://chatbotslife.com/a-short-guide-to-learn-neural-networks-and-get-famous-and-rich-then-bf7da3cba76f#.bsa5v9ekx)
* http://cs231n.stanford.edu/
* http://www.cs.utoronto.ca/~fidler/teaching/2015/CSC2523.html
* curated "awesome" lists
  * https://github.com/ChristosChristofidis/awesome-deep-learning
  * https://github.com/kjw0612/awesome-rnn
