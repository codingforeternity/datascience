### [MongoDB Installation](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
1. `sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 0C49F3730359A14518585931BC711F9BA15703C6`
2. `echo "deb [ arch=amd64 ] http://repo.mongodb.org/apt/ubuntu trusty/mongodb-org/3.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.4.list`
  * I'm running Ubuntu version 15.10.  The above command is for version 14.04.  The equivalent command for 16.04 didn't work with step #4 below.
3. `sudo apt-get update`
4. `sudo apt-get install -y mongodb-org`
5. follow these instructions: https://w0rldart.com/installing-mongodb-on-ubuntu-15-04/