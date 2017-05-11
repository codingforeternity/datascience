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
```