# Generative Financial Report Q&A Chatbot

## Introduction
Welcome to our GitHub repository for the Generative Financial Report Q&A Chatbot. This project aims to develop an innovative chatbot that leverages the power of natural language processing to provide factual answers to user queries, specifically focused on the financial reports of large listed public companies.

## Objective
Our primary objective is to create a generative chatbot that can accurately answer questions related to the financial reports of big public companies. This chatbot will utilize a custom-built knowledge base, generated from PDF files of financial reports, ensuring that all provided information is grounded in verified data.

## Problem Statement
The challenge lies in developing a chatbot capable of understanding and interpreting complex financial data encapsulated in the reports of large public companies. The bot is designed to:

- Reference a knowledge base created exclusively from the provided financial reports in PDF format.
- Minimize errors and hallucinations in generating responses.
- Deliver precise and accurate information to the user queries.

##  Installation 

```
pip install -r requirements.txt
```

```You will also need to add your OpenAI API key to the .env file.```

3. To use the application, run the ```app.py``` file with the streamlit CLI (after having installed streamlit):

```
streamlit run app.py
```

## Dataset

The pdfs was downloaded from the web and stored in the ./docs folder.

The automatic QA dataset generation code is QA_dataset_generation.ipynb, and the qa pairs are stored in the QA_dataset_v2.json (simple) and QA_dataset_v3.json (hard).

## Contact
xushihao6715@gmail.com