################################################################################################################
# This is a hierarchical classification problem, trainind data contains three classes which are hierarchically
# linked to each other. The adopted strategy is to classify each of the category starting from highest hierachy
# and then onwards (top to  bottom). There are only three predictors (attributes) which are directly useable without 
# requiring any data engineering 'Marque','Produit_Cdiscount','prix'. quality of prix was very poor, 
# means poor predictor; but we better to keep it as it also contains some information (among acute shortage of 
# predictors).
#
# There are also two text columns (description, labele). certainly their textual shape never qualifies them 
# to be used as predictor. So by means of NLP techniques, we have translated each of them (description, label)
# into x number of numerical attributes. The repeted experiments shows that x be 5.
# In this way, we data engineer into obtaining 3 + 5 + 5 = 13 predictors among of which 'marque' is categorical and 
# all others are numerical.
# 
# Problems: While selecting the most suitable algorithm, I noticed that our three class attributes have 
# distinct values as:  (categori1 = 51, categori2 < 600, categori1 > 1000). Large number of categorical classes
# reduce the selection criteria of available classifier as most of them are unable to handle so large number of 
# labelsin a class attribute. however, luckily, h2o implementation can handle upto one thousande categorical 
# states of class variables. So I adopted the same. Second reason of choosing h2o is that it can be easily 
# deployed in cluster of commodity machines yielding the real benefits of distributed computing for big data.  
# In h2o, 5 classifiers are available. Comparing among linear regression (glm), random forest (rf), gradient descent
# method (gbm), naive bayes (nb) and deep neural network (dnn), my preliminary results have shown that later (dnn) 
# was most suitable for this problem.

# First level of classification (categorie1): is straight forward which is to finds categorie1 given 13 predictor.

# Second level of classification (categorie2): we split test data in 'n' number of categorie1 where n means 51 
# (or less number of) of predicted classes of categorie1. For each categorie1, train data was subsetted accordingly.
# A model is trained only on each of the subset and the prediction is sought across its relevat (not all) test dataset
# at the end of this loop, all predictions are bind row wise. In situation of single instance in train, 
# no classification was possible. So we take the intutive result upon such problems
# 
# Third level of classification (categorie3): The same concept was applied for prediction of third categorie 
# in which train was subsetted on categorie1 and categorie2. In situation of single instance in train, 
# no classification was possible. So we take the intutive result upon such problems
# 
# Another notable characteristic of the data was the problem of class imbalance. This phenomenon was evient across all 
# three categories. An ideal solution is to issue a parameter in learning the model as "balance_classes = TRUE". 
# However, it works only if the samples in cross validation are atleast 3, and we notice that in categorie2 
# and categorie3, there were labels with 1 or 2 instances only. Secondly, setting the parameter as TRUE required
# costly processing (both in space and time). The clear advantage is that it improoves recall and precision (eventually F1)
#
#################################################################################################################

#================================================================================================================
#        Step 1.1     
#    Loading required libraries etc 
#================================================================================================================

rm(list=ls()) # Remove every variable 
# Set working directory, mandatory step
setwd("/home/mnaeem/r.codes/internet.memory/")

# load libraries 
library(data.table) # data loading, writing result
library(text2vec) # text to numeric transformation
library(tm) # punctuation and stop word removal
library(irlba) # singular vector decomposition used in text to numeric transition
library(doParallel) # multi-core, text2vec is by default single core, but extendable for multicore  
library(dplyr) # sampling
library(h2o) # suite of machine learning classification

#================================================================================================================
#        Step 1.2     
#     Work with sampled data or full data 
#================================================================================================================

trainPath <- ''
work.with.sample.data <- TRUE

if (work.with.sample.data) {
  # work with sample data
  # linux terminal command (very fast sampling)
  system ('head -1 classification/train_data.csv > classification/train.sample.csv')
  system ("perl -ne 'print if (rand() < .07)' classification/train_data.csv >> classification/train.sample.csv")
  trainPath <- "classification/train.sample.csv" 
  
 # if above not works then pure R solution
 # tr <- fread ("classification/train_data.csv")
 # library(dplyr)
 #  tr <- sample_n(tr, 500000, replace = FALSE)
 #  save(tr, file="classification/train.1.csv")
  
} else
{
  # full data
  trainPath <- "classification/train_data.csv" 
}
remove(work.with.sample.data)
#-----------------------------------------------------------

tr <- fread(trainPath) 
tr$Identifiant_Produit <- NULL
save (tr, file='classification/tr.Rda')

  ##################################################################################
 
#================================================================================================================
#        Step 2.1     
#     Preprocessing 
#================================================================================================================


#------- Load Data train and test ------------------------------------------------
  #------- Turn value from text ------------------------------------------------------
  # this function will translate text into sz=5 number of numeric variables
  # it will save these on hard disk, later on be appended to train
  source('value.from.text.R')
  
  load(file = 'classification/tr.Rda')
  ts <- fread('classification/test_data_final.csv') 
  tr.row.count <- nrow(tr)
  tr <- data.frame(tr$Description)
  colnames(tr) <- 'Description'
  ts <- data.frame(ts$Description)
  colnames(ts) <- 'Description'
  
  tr <- rbind(tr,ts); remove(ts)
  
  tr <- data.frame(lapply(tr, as.character), stringsAsFactors=FALSE)
  
  value.from.text(sz=5, 1) # Description, second parameter is for file numbering

  #--------------------------------------------------
  load(file = 'classification/tr.Rda')
  ts <- fread('classification/test_data_final.csv') 
  tr.row.count <- nrow(tr)
  tr <- data.frame(tr$Libelle)
  colnames(tr) <- 'Libelle'
  ts <- data.frame(ts$Libelle)
  colnames(ts) <- 'Libelle'
  
  tr <- rbind(tr,ts); remove(ts)
  
  tr <- data.frame(lapply(tr, as.character), stringsAsFactors=FALSE)
  
  value.from.text(sz=5, 2) # Libelle, second parameter is for file numbering
  
  #================================================================================================================
  #        Step 2.2     
  #     Appending novel attributes 
  #================================================================================================================

load(file = 'classification/tr.Rda')
ts <- fread('classification/test_data_final.csv') 
tr.row.count <- nrow(tr) # splitter between union of tr and ts
  
tr$Description <- NULL
ts$Description <- NULL
tr$Libelle <- NULL
tr$Libelle <- NULL
ts$Identifiant_Produit <- NULL
  

# Append Latent Semantic Analysis attributes in the train generayed from "description"
load(file='classification/u.1.Rda')
tr <- cbind(tr, u[c(1: tr.row.count),])
ts <- cbind(ts, u[-c(1: tr.row.count),])
remove(u); gc()

# Append Latent Semantic Analysis attributes in the train generayed from label"
load(file='classification/u.2.Rda')
tr <- cbind(tr, u[c(1: tr.row.count),])
ts <- cbind(ts, u[-c(1: tr.row.count),])

remove(u, tr.row.count ); gc()

#================================================================================================================
#        Step 2.3     
#     Feature Engineering 
#================================================================================================================

tr$Categorie1 <- as.factor(tr$Categorie1)
tr$Categorie2 <- as.factor(tr$Categorie2)
tr$Categorie3 <- as.factor(tr$Categorie3)
tr$Marque <- as.factor(tr$Marque)

ts$Marque <- as.factor(ts$Marque) 

tr$prix <- as.numeric(tr$prix)
# na be considered as -1
tr$prix [is.na(tr$prix)] <- -1  
tr$Produit_Cdiscount <- as.numeric(tr$Produit_Cdiscount) 
# na be considered as -1
tr$Produit_Cdiscount [is.na(tr$Produit_Cdiscount)] <- -1

ts$prix <- as.numeric(ts$prix)
# na be considered as -1
ts$prix [is.na(ts$prix)] <- -1  
ts$Produit_Cdiscount <- as.numeric(ts$Produit_Cdiscount) 
# na be considered as -1
ts$Produit_Cdiscount [is.na(ts$Produit_Cdiscount)] <- -1


# We have saved our work at many stages so that we can resume our work 
# next time from a specific stage  
save(tr, file = 'classification/tr.Rda')
save(ts, file = 'classification/ts.Rda')
#---------------------------------------------------------------- 
# only required if we want to resume our work from this stage, saves time
load(file = 'classification/tr.Rda')
load(file = 'classification/ts.Rda')

#---------------------------------------------------------------- 

#- Examination of labels (categories 1,2,3) 
cat.1 <- as.data.frame (table (tr$Categorie1)) ; plot (cat.1 )
length (unique(tr$Categorie1)); length (unique(tr$Categorie2)); length (unique(tr$Categorie3))


#================================================================================================================
#        Step 2.4     
#     Subsetting 
#================================================================================================================

# subsetting, this is useful for reducing the stress on heavy processing
# subsetting across categorie 1
cat.1 <- data.frame(cat.1 = tr$Categorie1) ; cat.1 <- cat.1[!duplicated(cat.1), ] ; cat.1 <- droplevels(cat.1)

ul <- length(cat.1)
for (i in 1:ul ) {
  train <- subset(tr, Categorie1 == cat.1[i])
  train <- droplevels(train)
  save(train, file = paste('input/', cat.1[i], '.Rda', sep = '') )
  print( paste('saving ', i , '/', ul)   ); flush.console()
}

#==========================================================================
#==========================================================================
# subsetting across categorie 1 and 2
# if categorie 2 is unique then we don't need categorie 1 in subsetting
# but including both ensures to handle same categorie2 under two different categori 1
cat.1.2 <- data.frame(cat.1 = tr$Categorie1, cat.2 = tr$Categorie2) 
cat.1.2 <- cat.1.2[!duplicated(cat.1.2), ]; cat.1.2 <- droplevels(cat.1.2)


ul <- nrow(cat.1.2)
for (i in 1:ul ) {
  train <- subset(tr, (Categorie1 == cat.1.2[i,1] & Categorie2 == cat.1.2[i,2])   )
  train <- droplevels(train)
  save(train, file = paste('input/', cat.1.2[i,1],'.', cat.1.2[i,2], '.Rda', sep = '') )
  print( paste('saving ', i , '/', ul)   ); flush.console()
}
remove(ul, i, train)

#================================================================================================================
#        Step 3.1
#     Starting H2O, Loading (processed) traing and test data 
#================================================================================================================

#####################################################################################

# h2o demands jdk 1.6 to 1.8, if needed, then explicitly set java home for R session
# don't need to set java, if already set at Operating Sys level 
# Sys.setenv(JAVA_HOME = "/opt/java/jdk1.8.0_111")
library(h2o)
localH2O <- h2o.init(max_mem_size = '8g', nthreads = -1) 

#================================================================================================================
#        Step 3.2
#     train category 1
#================================================================================================================

# only required if we want to resume our work from this stage, saves time
load(file = 'classification/tr.Rda')
tr$Categorie2 <- NULL # not required at this staage
tr$Categorie3 <- NULL 

train.h2o <- as.h2o(tr)
y <- 1; # label number , in this case categorie1
index <- c(2: ncol(tr)) # index of predictors, all except categorie2 and categorie3
remove(tr); gc()

m <- h2o.randomForest(index, y, train.h2o, ntrees = 25, nfolds = 5, seed = 1234)

save(m, file='m.model')

print(m@model$cross_validation_metrics_summary )
# 0.6391% accuracy Model performance across 5 % sampling
# sampling is taken to consider every categorie 3 (shown earlier)
# print (m@model$run_time ) 

load(file = 'classification/ts.Rda')
test.h2o <- as.h2o(ts); 
pred <- as.data.frame(h2o.predict(m, newdata=test.h2o))
remove(cat.1, cat.1.2, m); gc()
cn <- c('Categorie1', colnames(ts))
ts <- cbind (pred$predict, ts)
colnames(ts) <- cn
save(ts, file='classification/ts.1.Rda')


#================================================================================================================
#        Step 3.3
#     train category 2
#================================================================================================================

load(file='classification/ts.1.Rda')

ts.cat.1 <- data.frame(ts.cat.1 = ts$Categorie1)
ts.cat.1 <- ts.cat.1[!duplicated(ts.cat.1), ] ; ts.cat.1 <- droplevels(ts.cat.1)
y <- 1;   

cur.result <- data.frame()

#load(file =  'cur.result.4')
#i <- 5 # testing individual one by one 
ul <- length(ts.cat.1) # classify 48 categorie 1

for (i in 1:ul ) {
  ts.current <- subset(ts, Categorie1 == ts.cat.1[i])
  ts.current <- droplevels(ts.current)
  
  print( paste('Training 2nd Category given category 1 (', ts.cat.1[i] , ')' )   ); flush.console()
  
  load(file = paste('input/', ts.cat.1[i], '.Rda', sep = '') )
  
  train$Categorie1 <- NULL # constant column, unnecessary load
  train$Categorie3 <- NULL
  index <- c(2:ncol(train)) # refresh the indexing of predictor variable
  
  Family = 'multinomial' # by default, we assume there are atleast three labels 
  sz <- length(unique(train$Categorie2))
  
  if (sz == 1 ) {
    Family = '' # no classification required, there is single label
    
    if (i == 1) { # it is first chunk of prediction
      
      cur.result <- data.frame(rep(train$Categorie2[1],nrow(ts.current)) )
      colnames(cur.result) <- 'Categorie2'
      row.Nums <- which((ts$Categorie1 == ts.cat.1[i] )) 
      
    } else { # 2nd or succesive chunk of prediction
      
      tmp <- data.frame(rep(train$Categorie2[1],nrow(ts.current)) )
      colnames(tmp) <- 'Categorie2'
      tmp$Categorie2 <- as.factor(as.character(tmp$Categorie2))
      cur.result <- rbind(cur.result, tmp)
      row.Nums <- c(row.Nums, which(ts$Categorie1 == ts.cat.1[i] ) )
      remove(tmp)
    
    }
    
  } else if (sz==2) {
    Family = 'bernoulli' # found only two labels
  } 
  
  
  if (Family != '') {
  
  train.h2o <- as.h2o(train)
  nrow.train = nrow(train)
  remove(train); gc()
  
  print( paste('Training: ', i, '/', ul)  ); flush.console()
  m <- ''
  if (nrow.train >= 20) {
    m <- h2o.gbm(index, y, train.h2o,  distribution = Family, model_id ="gbm.1", seed = 1234, nfolds = 5, ntrees = 100 )
  } else {
    m <- h2o.deeplearning(index, y, train.h2o, model_id="dl", 
                          distribution = Family, 
                          hidden=c(32,32,32),                  ## small network, runs faster
                          epochs=1000000,                      ## hopefully converges earlier...
                          score_validation_samples=10000,      ## sample the validation dataset (faster)
                          stopping_rounds=2,
                          stopping_metric="misclassification", ## could be "MSE","logloss","r2"
                          stopping_tolerance=0.01
    )
  } 
  
  print(m); flush.console()
  
  test.h2o <- as.h2o(ts.current)
  print( paste('Prediction: ', i, '/', ul)  ); flush.console()
  pred <- as.data.frame(h2o.predict(m, newdata=test.h2o))
  remove(test.h2o, train.h2o); gc()
  
  if (i == 1) {
    cur.result <- data.frame(pred$predict) 
    colnames(cur.result) <- 'Categorie2'
    row.Nums <- which(ts$Categorie1 == ts.cat.1[i]) 
  } else {
    tmp <- data.frame(pred$predict)
    colnames(tmp) <- 'Categorie2'
    cur.result <- rbind(cur.result, tmp)
    row.Nums <- c(row.Nums, which(ts$Categorie1 == ts.cat.1[i]) )
    remove(tmp)
  }
  remove(pred,m); gc()

  } #  if (Family != '') {
  save(cur.result, file = paste('cur.result', i, sep = '.') )

} # loop for category 2

#=======================================================
cur.result$row.Nums <- row.Nums
# results need to be ordered to align with the original test data 
cur.result <- cur.result[order(cur.result$row.Nums),]

ts <- cbind (ts, Categorie2 = cur.result$Categorie2)

save(ts, file='classification/ts.2.Rda')

remove(row.Nums, ts.cat.1, cur.result,i,ul,index,y,ts.current, Family,sz); gc()


#================================================================================================================
#        Step 3.4
#     train category 3
#================================================================================================================

load(file='classification/ts.2.Rda')

ts.cat.1.2 <- data.frame(cat.1 = ts$Categorie1, cat.2 = ts$Categorie2) 
ts.cat.1.2 <- ts.cat.1.2[!duplicated(ts.cat.1.2), ]; ts.cat.1.2 <- droplevels(ts.cat.1.2)

y <- 1; 

cur.result <- data.frame()
ul <- nrow(ts.cat.1.2)

#load(file = 'cur.result.244' )
#load(file = 'row.Nums' )
#i <- 245 #individual testing

for (i in 1:ul) {
  ts.current <- subset(ts, (Categorie1 == ts.cat.1.2[i,1] & Categorie2 == ts.cat.1.2[i,2])  )
  ts.current <- droplevels(ts.current)
  print( paste('Training 3rd Category given category 1 (', ts.cat.1.2[i,1], ') & 2 (',ts.cat.1.2[i,2], ')'  )   ); flush.console()
 
  load(file = paste('input/', ts.cat.1.2[i,1], '.', ts.cat.1.2[i,2], '.Rda', sep = '') )
  train$Categorie1 <- NULL # constant column
  train$Categorie2 <- NULL # unnecesary load
  index <- c(2:ncol(train)) # refresh the indexing of predictor variable
  
  Family = 'multinomial'
  sz <- length(unique(train$Categorie3))

  if (sz == 1 ) {
    Family = '' # no classification performed
    print( paste('Estimation: ', i, '/', ul)  ); flush.console()
    
    if (i == 1) {
      cur.result <- data.frame(rep(train$Categorie3[1],nrow(ts.current)) )
      colnames(cur.result) <- 'Categorie3'
      row.Nums <- which((ts$Categorie1 == ts.cat.1.2[i,1] & ts$Categorie2 == ts.cat.1.2[i,2])) 
    } else {
      tmp <- data.frame(rep(train$Categorie3[1],nrow(ts.current)) )
      colnames(tmp) <- 'Categorie3'
      tmp$Categorie3 <- as.factor(as.character(tmp$Categorie3))
      cur.result <- rbind(cur.result, tmp)
      row.Nums <- c(row.Nums, which((ts$Categorie1 == ts.cat.1.2[i,1] & ts$Categorie2 == ts.cat.1.2[i,2])) )
      remove(tmp)
    }
    
  } else if (sz==2) {
    Family = 'bernoulli'
  } 

  if (Family != '') {
  
  train.h2o <- as.h2o(train)
  nrow.train = nrow(train)
  remove(train); gc()
  
  print( paste('Training: ', i, '/', ul)  ); flush.console()  
  m <- ''
  if (nrow.train >= 27) {
    m <- h2o.gbm(index, y, train.h2o, distribution = Family, model_id ="gbm.3", seed = 1234, nfolds = 5, ntrees = 50 )
  } else {
    m <- h2o.deeplearning(index, y, train.h2o, model_id="dl", distribution = Family,   
                          hidden=c(32,32,32),                  ## small network, runs faster
                          epochs=1000000,                      ## hopefully converges earlier...
                          score_validation_samples=10000,      ## sample the validation dataset (faster)
                          stopping_rounds=2,
                          stopping_metric="misclassification", ## could be "MSE","logloss","r2"
                          stopping_tolerance=0.01
    )
  } 
  
  print(m)
  
  print( paste('Prediction: ', i, '/', ul)  ); flush.console()
  test.h2o <- as.h2o(ts.current); 
  pred <- as.data.frame(h2o.predict(m, newdata=test.h2o))
  remove(m, test.h2o, train.h2o); gc()
  
  if (i == 1) {
    cur.result <- data.frame(pred$predict)
    colnames(cur.result) <- 'Categorie3'
    row.Nums <- which((ts$Categorie1 == ts.cat.1.2[i,1] & ts$Categorie2 == ts.cat.1.2[i,2])) 
  } else {
    tmp <- data.frame(pred$predict)
    colnames(tmp) <- 'Categorie3'
    cur.result <- rbind(cur.result, tmp)
    row.Nums <- c(row.Nums, which((ts$Categorie1 == ts.cat.1.2[i,1] & ts$Categorie2 == ts.cat.1.2[i,2])) )
    remove(tmp)
  }
  remove(pred)
  } #   if (Family != '') { 

  save(cur.result, file = paste('cur.result', i, sep = '.') )
  save(row.Nums, file = 'row.Nums')
} # loop for 3rd category



#================================================================================================================
#        Step 3.5
#    Dumping the classification result of last phase 
# Finished learning, training, testing phase  
#================================================================================================================
# shut down h2o node 
h2o.shutdown(prompt = FALSE); gc()
 
cur.result$row.Nums <- row.Nums
# results need to be ordered to align with the original test data 
cur.result <- cur.result[order(cur.result$row.Nums),]

ts <- cbind (ts, Categorie3 = cur.result$Categorie3)

save(ts, file='classification/ts.3.Rda')

# clear up memory
remove(localH2O, ts.cat.1, ts.current, Family,i,cat.1,sz,index,y, row.Nums, ts.cat.1.2, cat.1.2, cur.result)
gc()


#================================================================================================================
#        Step 3.6
#     Formulate the result file 
#================================================================================================================

load(file='classification/ts.3.Rda')

result <- fread("classification/test_data_final.csv", select = c('Identifiant_Produit')) 
result <- data.frame(result$Identifiant_Produit, ts$Categorie1, ts$Categorie2, ts$Categorie3)
colnames(result)  <- c('Identifiant_Produit', 'Categorie1', 'Categorie2', 'Categorie3')
write.csv2(result, "result.cat123.csv", row.names=FALSE, quote=FALSE)

result$Categorie1 <- NULL
result$Categorie2 <- NULL
# desired result sheet
result$Categorie3 <- paste("<", result$Categorie3, ">", sep="")
write.csv2(result, "result.cat3.csv", row.names=FALSE, quote=FALSE)

#==================== END =========================================================
