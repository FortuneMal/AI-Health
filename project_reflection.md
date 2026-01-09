1. Project Goal
The plan is to build an app that can predict if someone has heart disease based on their medical stats. This month, I'm going from raw data to a fully deployed AI. This first week was all about getting the data clean and ready for the models to learn from.

2. The Data
The Dataset: I chose the Heart Disease dataset (heart.csv).
What's inside: It has 14 different health markers like age, cholesterol, and blood pressure.
The "Health Check": I used Pandas to scan the data. It's a solid dataset with 1,025 rows and a very even split—about half the people have heart disease and half don't, which is perfect for training an AI without bias.

3. Getting the Data "Model-Ready"
I didn't just throw the raw numbers into the AI; I had to prep them first:
Cleaning: I double-checked for any empty spots or "N/A" values to make sure the model wouldn't crash.
Levelling the Playing Field: Features like "cholesterol" are in the hundreds, while "age" is much smaller. I used Standardization to put everything on the same scale so the model treats every feature fairly.
The Split: I broke the data into three groups: Training (70%), Validation (15%), and Test (15%). I used a "random seed" so that if I run the code again, the groups stay exactly the same.

4. Problems I Hit (and Fixed)
Missing Tools: At first, my code wouldn't graph anything because seaborn wasn't installed. A quick pip install fixed that.
File couldn't be found: My script couldn't find the heart.csv file because it was tucked away in a subfolder. I updated the code to look inside the /data folder specifically.
Folder Errors: The script tried to save my new files into a folder that didn't exist yet, causing an error. I added a line of code to automatically create the /data folder if it’s missing.
