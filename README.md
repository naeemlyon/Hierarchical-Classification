# Hierarchical-Classification
Hierarchical classification
A look into the nature of train and test data
Data set link:
https://1drv.ms/u/s!Ajqi-jzfv9JugSmRehM37Gui8bj0


This is a hierarchical classification problem, trainind data contains three classes which are hierarchically
linked to each other. The adopted strategy is to classify each of the category starting from highest hierachy
and then onwards (top to  bottom). There are only three predictors (attributes) which are directly useable without 
requiring any data engineering 'Marque','Produit_Cdiscount','prix'. quality of prix was very poor, 
means poor predictor; but we better to keep it as it also contains some information (among acute shortage of 
predictors).

There are also two text columns (description, labele). certainly their textual shape never qualifies them 
to be used as predictor. So by means of NLP techniques, we have translated each of them (description, label)
into x number of numerical attributes. The repeted experiments shows that x be 5.
In this way, we data engineer into obtaining 3 + 5 + 5 = 13 predictors among of which 'marque' is categorical and 
all others are numerical.
 
Problems: While selecting the most suitable algorithm, I noticed that our three class attributes have 
distinct values as:  (categori1 = 51, categori2 < 600, categori1 > 1000). Large number of categorical classes
reduce the selection criteria of available classifier as most of them are unable to handle so large number of 
labelsin a class attribute. however, luckily, h2o implementation can handle upto one thousande categorical 
states of class variables. So I adopted the same. Second reason of choosing h2o is that it can be easily 
deployed in cluster of commodity machines yielding the real benefits of distributed computing for big data.  
In h2o, 5 classifiers are available. Comparing among linear regression (glm), random forest (rf), gradient descent
method (gbm), naive bayes (nb) and deep neural network (dnn), my preliminary results have shown that later (dnn) 
was most suitable for this problem.

First level of classification (categorie1): is straight forward which is to finds categorie1 given 13 predictor.

Second level of classification (categorie2): we split test data in 'n' number of categorie1 where n means 51 
(or less number of) of predicted classes of categorie1. For each categorie1, train data was subsetted accordingly.
A model is trained only on each of the subset and the prediction is sought across its relevat (not all) test dataset
at the end of this loop, all predictions are bind row wise. In situation of single instance in train, 
no classification was possible. So we take the intutive result upon such problems

Third level of classification (categorie3): The same concept was applied for prediction of third categorie 
in which train was subsetted on categorie1 and categorie2. In situation of single instance in train, 
no classification was possible. So we take the intutive result upon such problems
 
Another notable characteristic of the data was the problem of class imbalance. This phenomenon was evient across all 
three categories. An ideal solution is to issue a parameter in learning the model as "balance_classes = TRUE". 
However, it works only if the samples in cross validation are atleast 3, and we notice that in categorie2 
and categorie3, there were labels with 1 or 2 instances only. Secondly, setting the parameter as TRUE required
costly processing (both in space and time). The clear advantage is that it improoves recall and precision (eventually F1)

