# FRAUDGUARD AI

Repository Link: https://github.com/Lesmore2/AI-Based-Financial-Fraud-Detection-System.git
Group Name: Tech-Mate
Deadline: 02/11/2026  23h59

What FraudGuard AI does
FraudGuard AI is an AI-based financial fraud detection prototype. The project's objective is to rank transactions so human analysts review the riskiest first, with reasons.

The prototype delivers five things:
1.	Ranking transactions :  four different models each score a transaction. The model that performs best picks which transactions an analyst should look at first.
2.	A review menu for the analyst : each transaction is marked HIGH, MEDIUM or LOW priority, with context to explain why. The computer can read the priority out loud (using pyttsx3), and it still shows the text if there is no voice available.
3.	Hourly analysis and forecasting : the system adds up how many transactions and how many frauds happen each hour. It then predicts the next 24 hours and checks whether that prediction is better than the simple "same as yesterday" guess.
4.	Spotting scam SMS messages:  it learns typical scam wording from labelled text messages and uses that to flag spam.
5.	Ask FraudGuard : a simple question and answer helper. It answers from a fixed list of questions in fraudguard_faq.csv, and it can also answer questions about the project's own results.

Prototype notice
This is a prototype built on simulated data. It uses synthetic PaySim mobile money records and a public SMS Spam Collection dataset , not live bank data. Scores are uncalibrated, patterns may change, and the project is not ready for operational use. Any real deployment would require human review, access controls, and validation on representative local data.

Requirements
•	Python 3.10 or newer
•	pip 
•	pyttsx3 for voice output 

Datasets and where to download
PaySim (synthetic mobile-money transactions): Kaggle: ealaxi/paysim1
SMS Spam Collection: UCI: doi.org/10.24432/C5CC84

FAQ file placement
fraudguard_faq.csv must sit in the same directory as fraudguard_ai.py. Ask FraudGuard reads it at runtime for FAQ retrieval; if it is missing or in another folder, the assistant will not work.

Run command
python fraudguard_ai.py

The Project-board link:

Each member's responsibility
Lindokuhle Ntanzi / Part 1 Validate dataset discovery, feature inputs and chronological split; record the actual
counts.
Lesego Makgalemele/ Part 2 Train and evaluate the four models; interpret AP, queue size and errors.
Sibusiso Bangani / Part 3 Test transaction menu and speech; clarify review context and collect a demonstration.
Melusi Mbhele / Part 4 Check forecast selection/holdout protocol and record both MAE values.
Ernest Zwane / Part 5 Test SMS and FAQ behaviour; verify sources and capture supported/fallback cases.
Shared final tasks Complete report, slides, poster, signatures, Grammarly and final

To Do: The task exists but nobody has started it.
In progress: The assigned member has started and records progress.
In review: A reviewer is checking the concrete change or completed evidence.
Done: The acceptance checks pass and the required merge/evidence is complete

#1: Starting Part 1. Setting up PaySim and the SMS data, then running the profile and split.

#2: Starting Part 2. Running training for the four models and the evaluation.

#3: Starting Part 3. Testing the review menu and voice with random and fraud-labelled transactions.

#4: Starting Part 4. Running the hourly analysis and forecasting.

#5: Starting Part 5. Running the SMS evaluation and testing Ask FraudGuard.
