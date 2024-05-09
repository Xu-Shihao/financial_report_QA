# Finaatial Report QA

## Aboout the Project
This is a Python application that allows you to load a PDF and ask questions about it using natural language. The application uses a LLM to generate a response about your PDF. The LLM will not answer questions unrelated to the document. The application reads the PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response. Here is the Proof of Concept.

## Images of Proof of Concept

![logo](https://github.com/KalyanMurapaka45/DocGenius-Revolutionizing-PDFs-with-AI/blob/main/Outputs/Screenshot%202023-05-15%20212935.png)

![logo](https://github.com/KalyanMurapaka45/DocGenius-Revolutionizing-PDFs-with-AI/blob/main/Outputs/Screenshot%202023-05-15%20213027.png)

 
#  Installation 

```
pip install -r requirements.txt
```

```You will also need to add your OpenAI API key to the .env file.```

3. To use the application, run the ```app.py``` file with the streamlit CLI (after having installed streamlit):

```
streamlit run app.py
```
