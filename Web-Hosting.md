***

See also: [[Docker Notes]]

***

#### Flask Deployment Options (5/18/17)
* [Handling multiple requests in Flask](http://stackoverflow.com/questions/14672753/handling-multiple-requests-in-flask)
  * "Is there any way that I can make my Flask application accept requests from multiple users?"
  * "Yes, deploy your application on a different WSGI server, see the Flask deployment options documentation."
* [Flask Deployment Options](http://flask.pocoo.org/docs/0.12/deploying/)
  * "While lightweight and easy to use, Flask's built-in server is not suitable for production as it doesn’t scale well and by default serves only one request at a time."
  * This said, when you go to the AWS EB link on that page, the example EB deployment still runs using "Flask's build-in server," just with `debug` set to False.
* [Deploying a Flask Application to AWS Elastic Beanstalk](http://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html)
  * These instructions don't use Docker; they go straight from Python to AWS.
  * [Using the AWS Elastic Beanstalk Python Platform](http://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-container.html)
  * `pip freeze > requirements.txt` to create this file from all the installed packages in a virtualenv
* [Dockerizing a Python web app (and deploying it to EB)](https://aws.amazon.com/blogs/devops/dockerizing-a-python-web-app/)
  * This approach uses a `Dockerrun.aws.json` file in place of `docker-compose.yml`
* [Configuring Docker Environments (for deployment to EB)](http://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create_deploy_docker.container.console.html)
  * "Specify images by name in `Dockerrun.aws.json`. Note these conventions:"
    * "Images in *official* repositories on Docker Hub use a single name (for example, ubuntu or mongo)."
    * "Images in *other* repositories on Docker Hub are qualified with an organization name (for example, amazon/amazon-ecs-agent)."
    * "Images in *other online* repositories are qualified further by a domain name (for example, quay.io/assemblyline/ubuntu or account-id.dkr.ecr.us-east-1.amazonaws.com/ubuntu:trusty)."
  * Includes instructions on "Using Images from an Amazon ECR Repository" on EB
* [Difference between Amazon ec2 and AWS Elastic Beanstalk](http://stackoverflow.com/questions/25956193/difference-between-amazon-ec2-and-aws-elastic-beanstalk)
  * "Elastic Beanstalk is one layer of abstraction away from the EC2 layer. Elastic Beanstalk will setup an 'environment' for you that can contain a number of EC2 instances, an optional database, as well as a few other AWS components such as a Elastic Load Balancer, Auto-Scaling Group, Security Group. Then Elastic Beanstalk will manage these items for you whenever you want to update your software running in AWS."
  * "EC2 Container Service is Amazon's managed replacement for running your own Mesos cluster. It's a good solution if you're running multiple applications, and simplifies management and billing. If you're running a single application, *unless you just like the Dockerized model*, Beanstalk is a better option."
* [CloudAcademy: Amazon EC2 Container Service and Elastic Beanstalk: Docker on AWS](http://cloudacademy.com/blog/amazon-ec2-container-service-docker-aws/)
  * "three ways to run Docker containers on AWS"
      1. "Deploying Docker containers directly to an Ec2 instance."
      2. "Using Docker containers on Elastic Beanstalk."
      3. "Docker cluster management using the AWS EC2 Container Service."
  * "Even a slight difference between your development, test, and production environments may completely break your application. **Traditional development models follow a change management process to solve these kind of the problems. But this process won’t fit in today’s rapid build and deploy cycles.**"

#### [Getting Started with AWS](https://www.youtube.com/watch?v=bFc5Fg9YSQg)
* Set up multi factor identification (MFA) on root account and all IAM accounts
* Common user access policies (distinct from security groups)
  * AdministratorAccess - almost everything root can do
  * PowerUserAccess - everything except IAM (i.e. except create new users & groups)
* Security groups are firewalls for EC2 instances
  * Choose who the ports are open to using CIDR notation (IP/#, e.g. IP/32 is a single IP)

#### SSL certificate signing request generation
* https://www.godaddy.com/help/generating-a-certificate-signing-request-csr-tomcat-4x5x6x-5276
* https://maxrohde.com/2013/09/07/setting-up-ssl-with-netty/
* Yes, the public and private keys were there [in the keystore file]!
  * I’ve added the response from CSR and  updated the keystore with godaddy certs chain, so that now the keystore is  signed. The stage server will still show red lock because the host is not hamstoo.com, but it’ll turn green when browser will be visiting hamstoo.com
  * Actually I thought I’d need to build the keystore from scratch, and having a keystore with keys already inside was half the job done. 
  * You can use Keystore Explorer to look inside the keystore and get familiar with its structure

#### AWS TODO
* [AWS Certificate Manager](https://aws.amazon.com/blogs/aws/new-aws-certificate-manager-deploy-ssltls-based-apps-on-aws/) has free certs
  * Supposedly they only work though with Elastic Load Balancing (and Amazon CloudFront)
  * We aren't using this because Play Framework is handling our encryption, not ELB.  We'll still probably need to use ELB though.  Here's the communication chain: user -> ELB -> ECS/EC2 -> Docker container -> Play
* [EC2 Guide: Hosting a website on Amazon EC2](http://www.paul-norman.co.uk/2011/02/hosting-a-website-on-amazon-ec2)

#### Amazon Web Services
* EC2 (Docker) Container Registry (ECR) - Alternative to using DockerHub as a container registry.
  * *Store container images securely* with Amazon ECR - Create and manage a new private image repository and use the Docker CLI to push and pull images. Access to the repository is managed through AWS Identity and Access Management. 
* EC2 Container Service (ECS) - **Before you can run tasks in Amazon ECS, you need to launch container instances into your cluster**. For more information about how to set up and launch container instances, see Setting Up with Amazon ECS and Getting Started with Amazon ECS.
  * [Launching an Amazon ECS Container Instance](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/launch_container_instance.html)
* [AWS Docker Basics](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html)
* The Amazon ECS [instance and service roles](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/get-set-up-for-amazon-ecs.html#create-an-iam-user) are automatically created for you in the console first run experience, so if you intend to use the Amazon ECS console, you can move ahead to Create a Key Pair. If you do not intend to use the Amazon ECS console, and instead plan to use the AWS CLI, complete the procedures in [Amazon ECS Container Instance IAM Role](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/instance_IAM_role.html) and [Amazon ECS Service Scheduler IAM Role](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/service_IAM_role.html) before launching container instances or using Elastic Load Balancing load balancers with services.
  * Note that if you plan to launch instances in multiple regions, you'll need to create a key pair in each region.
  * To connect to your Linux instance from a computer running Mac or Linux, specify the .pem (in ~/keys) file to your SSH client with the -i option and the path to your private key.
* [Amazon EC2 console](https://console.aws.amazon.com/ec2/)
* **The Amazon ECS CLI supports Docker Compose** [FWC - this is the `ecs-cli` command which is different from the `aws` command], a popular open-source tool for defining and running multi-container applications. For more information about installing and using the Amazon ECS CLI, see [Using the Amazon ECS Command Line Interface](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_CLI.html).
  * ecs-cli configuration: `$ ecs-cli configure --region us-east-1 --cluster hamstoo`
* [Docker on Amazon Lightsail](https://davekz.com/docker-on-lightsail/)
* [Storing](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/instance_IAM_role.html) configuration information in a private bucket in Amazon S3 and granting read-only access to your container instance IAM role is a secure and convenient way to allow container instance configuration at launch time.
* [ecs-cli Command Reference](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/cmd-ecs-cli.html)
* [Allowing inbound SSH traffic](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html#add-rule-authorize-access) via [here](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/TroubleshootingInstancesConnecting.html) ("If you try to connect to your instance and get an error message `Network error: Connection timed out or Error connecting to [instance], reason: -> Connection timed out: connect`"
* [Using Data Volumes in Tasks](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/using_data_volumes.html)

#### [Moving hosting from GoDaddy to AWS](http://serverfault.com/questions/611805/switching-hosting-from-godaddy-to-aws)
* "In order to move from GoDaddy to AWS, you can a) *just move your code to AWS* (if you have a static website, move it to S3 instead of EC2), *and point your GoDaddy DNS records at your new host* (e.g. your EC2 instance's IP address). In EC2, your instance's IP address will change when the instance reboots, etc. As such it is a dynamic IP address, not well suited for hosting a website. Instead, *you need to allocate a static IP address, once that can be assigned to an instance - AWS call this an* **'Elastic IP'. This is what you will use for your A record**. (*The same holds true whether you use GoDaddy's DNS or Route53* - you need an A record that points to the IP address of your server - but *there is no requirement to use Route53 just because you are using AWS to host* your site - there are some exceptions - e.g. using an elastic load balancer)."
* You don't get to choose your elastic IP. You allocate one, and then associate it with the instance you want to use it with (you can move it between instances if needed). It remains the same until you release the IP address (even it is not associated with an instance). *Just ensure you allocate an IP address in the same region (e.g. US East) and scope (i.e. EC2 or VPC) as the instance you want to associate it with*.