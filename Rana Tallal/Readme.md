# Healthcare Secure Classification
A platform to securely classify medical images using the latest models.

# How it Works:
- Companies provide encrypted models
- Users (Doctors / Patients) provide encrypted images for classification
- Users select the model to use and the results are shared in realtime

# Benefits:
- Gives patients / doctors a secure way of getting results with respect to their data
- With nothing being saved on the server

# Current Implementation:
- The current implementation, creates a server and a client
- The client sends data to the server and the server listens to for the data
- The server predicts the encrypted information provided by the client

# What needs to be done:
- A website where this is managed
- A client library to encrypt users data and upload to the server
- A web interface for companies/ researchers to uplaod their models on
