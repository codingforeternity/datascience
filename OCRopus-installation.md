```bash
$ git clone https://github.com/tmbdev/ocropy.git
$ cd ocropy
$ virtualenv -p /usr/bin/python2 venv-ocropus
$ source venv-ocropus/bin/activate
$ pip install -r requirements.txt
$ wget -nd http://www.tmbdev.net/en-default.pyrnn.gz
$ mv en-default.pyrnn.gz models/
$ python setup.py install  # requires python2
$ ./run-test
ImportError: No module named _tkinter, please install the python-tk package
$ sudo apt-get install python-tk
$ ./run-test

$ mkdir fred
$ mv ~/Pictures/fred_ocropy_test.png fred
$ ocropus-nlbin fred/fred_ocropy_test.png -o fred

$ ocropus-gpageseg 'fred/????.bin.png'
ERROR:  fred/0001.bin.png SKIPPED too many connnected components for a page image (3889 > 1021) (use -n to disable this check)

$ ocropus-gpageseg -n 'fred/????.bin.png'
INFO:  scale 7.48331
ERROR:  fred/0001.bin.png: scale (7.48331) less than --minscale; skipping

$ identify fred_ocropy_test.png
fred_ocropy_test.png PNG 918x1001 918x1001+0+0 8-bit sRGB 180KB 0.000u 0:00.000

$ convert fred_ocropy_test.png -resize 1836x2002 bigger.png  # exact proportional resizing not required: "Resize will fit the image into the requested size. It does NOT fill, the requested box size." [http://www.imagemagick.org/Usage/resize/]

$ ocropus-nlbin fred/bigger.png -o fred

$ ocropus-gpageseg 'fred/????.bin.png'
INFO:  scale 14.966630
INFO:  number of lines 53
INFO:  finding reading order
INFO:  writing lines
INFO:      33  fred/0001.bin.png 15.0 34

$ ocropus-rpred -Q 4 -m models/en-default.pyrnn.gz 'fred/????/??????.bin.png'
INFO:  
INFO:  ########## /home/fred/Documents/code/ocropy/venv-ocropus/bin/ocropus-rp
INFO:  
INFO:  #inputs: 34
# loading object ././models/en-default.pyrnn.gz
INFO:  fred/0001/010002.bin.png:=.
INFO:  fred/0001/010003.bin.png:ocropy
INFO:  fred/0001/010001.bin.png:READMEAmd
INFO:  fred/0001/010005.bin.png:you may need to do some image preprocessing, and possibly also traIn new models.
INFO:  fred/0001/010006.bin.png:In addhton to the recogntlon scnpts themsehes, there are a number of scnpts for ground truth edting and correcton,
INFO:  fred/0001/010004.bin.png:OCRopus s a collecton of document analysis programs, not a turn-key OCR system. In order to apply t to your documents,
INFO:  fred/0001/010009.bin.png:since t seems to confuse too many users).
INFO:  fred/0001/010007.bin.png:measuring error rates, determming confuson matnsces, etc. OCRopus commands will generally pnint a stack trace akong wth
INFO:  fred/0001/01000a.bin.png:Installing

$ ocropus-hocr 'fred/????.bin.png' -o fred/out.html
writing to fred/out.html
median_xheight 16.0
=== fred/0001.bin.png

$ firefox !$ &
```