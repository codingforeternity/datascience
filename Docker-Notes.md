#### Installation
* Don't use `sudo apt-get install docker docker-compose docker-engine` to install Docker.  Instead follow the steps below.
  1. Remove all existing Docker packages using the following commands: `apt list --installed | grep docker` and `sudo apt-get remove <docker_package>`
  2. Follow the 'Install using the repository' instructions [here](https://docs.docker.com/engine/installation/linux/ubuntu/).
    * "Before you install Docker for the first time on a new host machine, you need to set up the Docker repository."
  4. Continue on the same page through the 'Install Docker' section instructions, which will install `docker-engine` (version 1.12.6 at time of writing, 2/1/17)
  5. Per step #3-4 on [this](https://docs.docker.com/compose/install/) page, go to [here]() and run the following commands:
    * `$ sudo -i`
    * ``# curl -L https://github.com/docker/compose/releases/download/1.10.0/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose``
    * `chmod +x /usr/local/bin/docker-compose`