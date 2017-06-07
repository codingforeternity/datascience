### In-Memory MongoDB (6/7/17)
* [Embedded MongoDB when running integration tests - StackOverflow](https://stackoverflow.com/questions/6437226/embedded-mongodb-when-running-integration-tests), 2 options:
  1. [Embedded MongoDB](https://github.com/flapdoodle-oss/de.flapdoodle.embed.mongo)
  2. [fakemongo/Fongo](https://github.com/fakemongo/fongo) (formerly a Foursquare project?)
* [Integration Testing Done Right With Embedded MongoDB](https://dzone.com/articles/integration-testing-done-right)

### Notes
* Terminology: a "collection of documents" is analogous to a "table of rows."
  * "The `grades` array contains embedded *documents* as its elements."  So a "document" is also analogous to a hash or a dict, if you will.
* "If you attempt to add documents to a collection that does not exist, MongoDB will create the collection for you." [https://docs.mongodb.com/getting-started/shell/insert/]
* "All queries in MongoDB have the scope of a single collection."

### [MongoDB Installation](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
1. `sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 0C49F3730359A14518585931BC711F9BA15703C6`
2. `echo "deb [ arch=amd64 ] http://repo.mongodb.org/apt/ubuntu trusty/mongodb-org/3.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.4.list`
  * I'm running Ubuntu version 15.10.  The above command is for version 14.04.  The equivalent command for 16.04 didn't work with step #4 below.
3. `sudo apt-get update`
4. `sudo apt-get install -y mongodb-org`
  * This produced the following warning message, which the next step is meant to address: "invoke-rc.d: mongod.service doesn't exist but the upstart job does. Nothing to start or stop until a systemd or init job is present."
5. follow these instructions: https://w0rldart.com/installing-mongodb-on-ubuntu-15-04/