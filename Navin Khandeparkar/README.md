QuickDocIt:
A health query related chatbot

Model Architecture:
1. LSTM model (to classify if the query is health-related or not)
2. Preprocessing (to get all the required keywords) 
3. DocProduct (BERT Trained on Medical Q&A)
4. Preprocessing (to select the meaningful response from the set of response)

Workflow:
	1.The user speaks/enter the query:
	2. Check if the query is health-related :
	![Screenshot](Classify_if_healthRelated_query.png)
	![Screenshot](Not_a_healthRelated_query.png)
	3. Do the preprocessing:
	Check if the symptom is present, if not the ask.
	Check if the duration is present, if not the ask.
	Check if any medicine details are provided, if not the ask.
	4. Form the query with all parameters and pass it to the DocProduct model for response
	5. Do the preprocessing:
	Select the most appropriate response from the list provided by the DocProduct.
	![Screenshot](Final_results.png)

NOTE: The preprocessing part is incomplete. I didn't get time to complete the project and I had to submit as it was the last hour. I am truly sorry.
