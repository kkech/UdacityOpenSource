# private_ai_capstone_project

This project is part of the Udacity Private AI Challenge. A given scenario has been simulated in order to build a Credit Default Risk Shared Model:

It is common in Brazil that companies buy data from a bureau and then combine it with internal credit default indicators to create a risk model that drives concession. The credit default data is always siloed inside companies and since it is sensitive data, cannot be shared. But in the other hand, controlling default rate isn't usually the core business of these companies. So if there was a way for these companies to work together and achieve better default rates, while preserving its clients privacy... Enter Federated Learning and Encrypted Learning: this project aims to create a better credit default risk model by sharing a model between two or more companies, while still preserving customers' data.

There are 4 companies in the simulated scenario: "Shiny" and "High" are fictitional companies willing to rate their customers regarding their credit default risk and that have agreed to use their customer's data to train a shared model. "New Company" is another company, which haven't used it's data to train a model, but is also interested in using the shared model as Machine Learning as a Service. And finally, there's "Best View", which is us:  a fictitional bureau of data - a company that sells data and models, and which will be the the central point of the shared model.

With this scenario Pytorch and PySyft are being used to:
- Add Local Differential Privacy;
- Train a model using Federated Learning and
- Encrypt a model and forward it through encrypted data!