# IT-Ticket-Assignmnet

To build an AI-based classifier model to assign the tickets to right functional groups by analyzing the given description

Understanding the Business
In any IT industry, Incident Management plays an important role in delivering quality support to customers. An incident ticket is created by various groups of people within the organization to resolve an issue as quickly as possible based on its severity. Whenever an incident is created, it reaches the Service desk team and then it gets assigned to the respective teams to work on the incident.

The Service Desk team (L1/L2) will perform basic analysis on the user's requirement, identify the issue based on given descriptions and assign it to the respective teams.

The manual assignment of these incidents might have below disadvantages: More resource usage and expenses. Human errors - Incidents get assigned to the wrong assignment groups Delay in assigning the tickets More resolution times If a particular ticket takes more time in analysis, other productive tasks get affected for the Service Desk If this ticket assignment is automated, it can be more cost-effective, less resolution time and the Service Desk team can focus on other productive tasks.

Objective
Our objective here is to build an AI-based classifier model to assign the tickets to right functional groups by analysing the given description with an accuracy of at least 85%.

Dataset - https://drive.google.com/open?id=143R3O0o3s3YyC2ZAoVDZOWM1APiNtGPA

Steps followed
Text in Description is pre-processed by removing unwanted characters and words. Some descriptions are given in other languages which are translated to english internally. Stop words are removed and all the words are lemmatized.

With this pre-processed description the words in the corpus are tokenized and vectors are created using Bow, TFIDF and response encoding. 

Different ML algorithms are tried out - Naive Bayes, KNN, Logistice regression, Random Forest, 
