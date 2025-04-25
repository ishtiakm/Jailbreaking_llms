# Jailbreaking_llms

There are two datasets folder here: **data** folder contains jailbreaking prompts, and **MMLU** contains mcq questions on regular topics like math, politics, logic, morality etc.
**base_JB.py**
it uses the jailbreaking prompts from data/forbidden_question/unique_filtered_questions.csv
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
This code does inference for deepseek-32b model. If you want to use it for 1.5b or 7b model, just change all '32B' to '7B' or anything else.
The output responses are saved to 'result' folder. Remember, this is only the generated output. 
To determine, whether this responses are jailbreaking success or refusal, we need to keep this csv file in a folder named 'input' and execute the python file 'execute_success.py', which will use JailBreakBench api to determine that.

**pm_JB.py**
it uses the jailbreaking prompts from data/forbidden_question/unique_filtered_questions.csv
The difference with 'base_JB.py' is how it formats the prompt with a different message.
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
This code does inference for deepseek-1.5B model. If you want to use it for 32b or 7b model, just change all '1.5B' to '7B' or anything else.
The output responses are saved to 'result' folder. Remember, this is only the generated output. 
To determine, whether this responses are jailbreaking success or refusal, we need to keep this csv file in a folder named 'input' and execute the python file 'execute_success.py', which will use JailBreakBench api to determine that.

**tts_32b_JB.py**
it uses the jailbreaking prompts from data/forbidden_question/unique_filtered_questions.csv
The difference with 'base_JB.py' is how it adds a reflective response to the initial response.
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
This code does inference for deepseek-32B model. If you want to use it for 1.5b or 7b model, just change all '32B' to '7B' or anything else.
The output responses are saved to 'result' folder. Remember, this is only the generated output. 
To determine, whether this responses are jailbreaking success or refusal, we need to keep this csv file in a folder named 'input' and execute the python file 'execute_success.py', which will use JailBreakBench api to determine that.

**SFT_7b_JB.py**
it uses the jailbreaking prompts from data/forbidden_question/unique_filtered_questions.csv
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
This code does inference for deepseek-7b SFT model, that we got after executing 'train7b_fast.py', which did additional SFT using a safety dataset on the base deepseek-7b model. 
If you want to use it for 1.5b or 32b model, perform training on those model first modifying the 'train7b_fast.py' file.
The output responses are saved to 'result' folder. Remember, this is only the generated output. 
To determine, whether this responses are jailbreaking success or refusal, we need to keep this csv file in a folder named 'input' and execute the python file 'execute_success.py', which will use JailBreakBench api to determine that.

**base_MMLU.py**
it uses the mcq questions on 5 regular topics from the 'MMLU' folder.
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
This code does inference for deepseek-7b model. If you want to use it for 1.5b or 32b model, just change all '7B' to '32B' or anything else.
The output responses are saved to 'result/regular_tasks' folder. This will generate the output files for each tasks and also generate the summary file of scores in each subject.

**pm_MMLU.py**
it uses the mcq questions on 5 regular topics from the 'MMLU' folder.
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
The difference of this file with 'base_MMLU.py' is how it formats the prompt with a different message.
This code does inference for deepseek-1.5b model. If you want to use it for 7b or 32b model, just change all '1.5B' to '32B' or anything else.
The output responses are saved to 'result/regular_tasks' folder. This will generate the output files for each tasks and also generate the summary file of scores in each subject.

**tts_32b_MMLU.py**
it uses the mcq questions on 5 regular topics from the 'MMLU' folder.
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
The difference of this file with 'base_MMLU.py' is how it adds a reflective response to the initial response.
This code does inference for deepseek-32b model. If you want to use it for 7b or 1.5b model, just change all '32B' to '7B' or anything else.
The output responses are saved to 'result/regular_tasks' folder. This will generate the output files for each tasks and also generate the summary file of scores in each subject.

**SFT_7b_MMLU.py**
it uses the mcq questions on 5 regular topics from the 'MMLU' folder.
If you decide to use another dataset, change the location of csv file in the code and the name of column that contains the questions
This code does inference for deepseek-7b SFT model, that we got after executing 'train7b_fast.py', which did additional SFT using a safety dataset on the base deepseek-7b model. 
If you want to use it for 1.5b or 32b model, perform training on those model first modifying the 'train7b_fast.py' file.
The output responses are saved to 'result/regular_tasks' folder. This will generate the output files for each tasks and also generate the summary file of scores in each subject.

**execute_success.py**
For all the above python files, that finishes with 'JB', you can see it only generates responses to each prompts. Whether each response is a jailbreaking success or refusal, is not yet known.
Doing this manually can be tiring for a large number of prompts. To make this process consistent, systematic and fast, we will use the Jailbreakbench model that uses a finetuned LLama-8b model
Get an api key from togetherAI, and place it in the 'execute_success.py' file (the variable is on the top). Then put all your output files (that you got by running '_____JB.py' files) in a folder named 'input'. 
Create another empty folder named 'success'. And then execute 'execute_success.py' file. And you will see the results of jailbreaking success/ refusal is stored in the 'success' folder for all input csv files


**train7b_fast.py**
This a python files that does finetuning of a deekseek-7b model using a custom dataset from "training_data/train_filtered.tsv". In this repository, we did not include that folder or file due to its size. 
You can download the tsv file from here: https://huggingface.co/datasets/allenai/wildjailbreak/viewer/train?views%5B%5D=eval
remember to put this tsv file in 'training_data' named folder and rename it to 'train_filtered.tsv'... or change the name and location of the file in the python code.
After executing this python file, it will save the model to a new folder. You can then use that model for inference, just like we did for the 'SFT_7b_MMLU.py' and 'SFT_7b_JB.py' python files.



**FOR any more queries or additional files... contact: ishtiakmahmud6@gmail.com**
