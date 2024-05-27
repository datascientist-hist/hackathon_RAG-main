# AI Pharmacist Assistant

### Introduction and Motivation

<div style="text-align: center;">
    <img src="media/ai_pharm.png" alt="logo" style="width: 50%;">
</div>

Welcome to the documentation of our virtual pharmacy assistant. This chatbot was developed to improve the efficiency of stock management and to provide quick and precise support to pharmacists. The motivation behind this project stems from the need to optimise the processes of searching for and distributing medicines, reducing waiting times and increasing customer satisfaction.

One of the main strengths of our virtual assistant is its ability to recommend targeted medication for each patient. Based on the physical characteristics and specific symptoms of each customer, the chatbot avoids suggesting generic drugs, instead ensuring personalised and accurate advice. This approach ensures more effective treatment and greater attention to patients' individual needs.

In addition, interaction with the chatbot takes place through natural language, making it extremely intuitive and accessible. No computer skills or special training are required to use our virtual assistant. This makes the system easy to integrate into the daily workflow of any pharmacy, enhancing the experience for both pharmacists and customers.

Thanks to its advanced functionality for collecting and analysing sales statistics, the chatbot enables more informed and strategic stock management, further contributing to the operational efficiency of the pharmacy.

### How it works

The agent is designed to determine whether the question asked requires a technical answer or the execution of an SQL query. Depending on the nature of the question, the agent chooses the appropriate tool to use. 

If the question requires pharmaceutical advice, the agent uses the `pharmacist tool`, which is designed to recommend the most suitable drug for each patient, based on their physical condition and specific symptoms. 

If, on the other hand, the question requires data processing, the agent uses the `SQL tool`, which interprets the semantics of the question and executes an SQL query on an online server to provide the correct answer.

### How to use it

To run and use our chatbot, please ensure that all dependencies are correctly installed.

```{bash}
pip install code/requirements.txt
```
Start AI Pharmacist Assistent Chatbot by executing the following command:

```{bash}
cd code/
streamlit run chatbot.py
```

