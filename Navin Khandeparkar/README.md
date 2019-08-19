QuickDocIt:
A health query related chatbot

Model Architecture:
LSTM model (to classify if the query is health-related or not)
Preprocessing (to get all the required keywords) 
DocProduct (BERT Trained on Medical Q&A)
Preprocessing (to select the meaningful response from the set of response)

Workflow:
The user speaks/enter the query:
	image
      2. Check if the query is health-related :
 	image
3. Do the preprocessing:
Check if the symptom is present, if not the ask.
Check if the duration is present, if not the ask.
Check if any medicine details are provided, if not the ask.
4. Form the query with all parameters and pass it to the DocProduct model for response
5. Do the preprocessing:
Select the most appropriate response from the list provided by the DocProduct.

NOTE: The preprocessing part is incomplete. I didn't get time to complete the project and I had to submit as it was the last hour. I am truly sorry.
