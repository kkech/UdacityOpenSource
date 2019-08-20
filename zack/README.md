This notebook focuses on generating new, realistic-looking dog Images.

You can test this notebook thru uploading it directly to Kaggle Notebook.

If you choose to test this notebook other than Kaggle Notebook, you may update the following configuration
and consider downloading the necessary files:

For the dataset, kindly download it using this API command: kaggle competitions download -c generative-dog-images

Place this downloaded dataset to desire folder and update this configuration parameter:
	annotation_dir = desired directory folder
	dog_dataset = desired directory folder


For evaluation metrics file, kindly download it using this API command: kaggle datasets download -d wendykan/dog-face-generation-competition-kid-metric-input

Place this file to desire folder and update this configuration parameter:
	images_path = desired directory folder (same value as dog_dataset)
	public_path = desired directory folder

Areas for Improvement:
	You may update the hyperparameters and/or change the network architecture of Generator & Discriminator and try other variants of GANs Architecture.