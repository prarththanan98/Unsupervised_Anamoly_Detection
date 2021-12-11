# Unsupervised Anamoly Detection in Dataset.
Covid datset is used to build the pipeline for anamoly detection.
Preprocessed dataset is fed to One-Class SVM model and obtained the output and SVM score.
Then the datset with the SVM ouput is fed to Random Forest to obtain the Feature importance of the attributes.
Then the top eight attributes of high feature importance is obtained and and again fed to One Class SVM

 
