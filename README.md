# DPHN
From a Perceptual Perspective: No-Reference Image Quality Assessment Using Dual Perception Hybrid Network
# DATASET
We used datasets:live\csiq\livec\livemd. You need to download the appropriate data set and place the data set in the /DATE directory.
To ensure fairness in method comparison, we adopted the same strategy as most NR-IQA studies(dataloader2 & dataset_folder2). 
# Advice
Trying batch sizes greater than 16(e.g.32) to see if better performance can be obtained.
We recommend considering pre-training the reconstruction network, as this approach may potentially enhance model performance. 
