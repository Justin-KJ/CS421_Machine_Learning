This is the score after submitting on coda bench:
AUC:       0.8267
Precision: 0.6786
Recall:    0.3167
F1 Score:  0.4318

There are two additional information that was given to us
1) second_batch.npz includes four different anomaly classes generated from different heuristics

2) first_batch_with_labels.npz includes only two anomaly classes and these two anomaly classes are part of the four anomaly classes we have in this week's test data

Now you are tasked to improve the f1 score even further based on the information provided to you. 

Think step-by-step like a researcher:
1. Analyze the data from second_batch.npz
2. Analyze the previous iteration
3. Research online on latests and proven techniques that will maximize the F1 score
4. Design targeted fixes
5. Justify why each fix addresses a specific metrics

Your target: F1 score higher than 0.7 after submitting to coda bench
Do NOT rush into coding. Plan the architecture first, then implement.