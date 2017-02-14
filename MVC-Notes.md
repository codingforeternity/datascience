#### [Relationship between Model (in MVC) and DAO](https://coderanch.com/t/467978/Relationship-Model-MVC-DAO)
* In the Model-View-Controller design pattern, the **Model represents business logic and business data. Data access objects (DAO) objects may be part of a Model, but they are not the only objects that make up a business object Model.**
  * If this web framework example has business logic coded in whatever the Controller is, then it is not accurate.
  * The purpose of the Data Access Object design pattern is to shield the Model from "data" access logic, e.g. physical implementation of a data storage system. Business logic should be loosely coupled with infrastructure logic and data (CRUD) logic. Design patterns such as Data Access Object and Service Locator provide a guide on how to realize this. 

#### [In MVC , DAO should be called from Controller or Model](http://softwareengineering.stackexchange.com/questions/175950/in-mvc-dao-should-be-called-from-controller-or-model)
* In my opinion, you have to distinguish between the MVC pattern and the 3-tier architecture. To sum up:
  * 3-tier architecture:
    * data: persisted data;
    * service: logical part of the application;
    * presentation: hmi, webservice...
  * The MVC pattern takes place in the presentation tier of the above architecture (for a webapp):
    * data: ...;
    * service: ...;
    * presentation:
      * controller: intercepts the HTTP request and returns the HTTP response;
      * model: stores data to be displayed/treated;
      * view: organises output/display.
  * Life cycle of a typical HTTP request:
    1. The user sends the HTTP request;
    1. The controller intercepts it;
    1. The controller calls the appropriate service;
    1. The service calls the appropriate dao, which returns some persisted data (for example);
    1. The service treats the data, and returns data to the controller;
    1. The controller stores the data in the appropriate model and calls the appropriate view;
    1. The view get instantiated with the model's data, and get returned as the HTTP response.
