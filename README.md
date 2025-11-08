# YZV448E Term Project

Our task is to create a GDPR cleaner, a tool that detects personal information in text and 
removes or anonymizes it to comply with GDPR regulations. The base task is to create a NER
system that can identify personal information such as names, addresses, phone numbers, email addresses,
and other sensitive data. The advanced task is to implement the anonymization or removal of the detected
personal information.

[Data Source](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/overview)

- We will use a BERT model
- We will fine-tune the BERT model on a dataset containing labeled personal information.
- We will implement a full pipeline that gets a directory of text files, processes each file to detect
  personal information, and outputs cleaned versions of the files with personal information removed or anonymized.
- We will evaluate the system using the F5-score, instead of F1.


