###############################################################################
# Latent Semantic analysis 
# fn: feature/attribute number
# sz: number of right/lrft svd, controls the number of features to be returned
# an appropriate value of sz is optimized for gbm, dnn and other classifiers
# cnt: file counter to be used for file to be saved
##############################################################################

value.from.text <- function(sz, cnt) {
 fn <- 1
 
  print('preprocessing ...(cleaning punctuation)'); flush.console()  
  # Lowercase all words for convenience
  tr[[fn]]  <- tolower(tr[[fn]] )
  
  print("preprocessing ...(Remove all #hashtags and @mentions)")
  tr[[fn]] <- gsub("(?:#|@)[a-zA-Z0-9_]+ ?", "", tr[[fn]])
  
  print("preprocessing ...(Remove words with more than 3 numbers)")
  tr[[fn]] <- gsub("[a-zA-Z]*([0-9]{3,})[a-zA-Z0-9]* ?", "", tr[[fn]])
  
  print("preprocessing ...(Remove all punctuation)")
  tr[[fn]] <- gsub("[[:punct:]]", "", tr[[fn]])
  
  print("preprocessing ...(Remove all newline characters)")
  tr[[fn]] <- gsub("[\r\n]", "", tr[[fn]])
  
  print("preprocessing ...(Replace double+ whitespace)")
  tr[[fn]] <- gsub(" {2,}", " ", tr[[fn]])
  
  print("preprocessing ... Regex pattern for removing stop words")
  stop_pattern <- paste0("\\b(", paste0(stopwords('fr'), collapse="|"), ")\\b")
  tr[[fn]] <- gsub(stop_pattern, "", tr[[fn]])

  #  save (tr, file='tr.tmp')
  #  load (file='tr.tmp')
  
  #---------------------------------------------------
         
  # utilize all cores less than one for multiprocessing
  workers <- detectCores() - 3  
  if (is.na(detectCores())) {
    print('It seems your machine does not support multicore processing... allocating single core now')
    flush.console()
    workers <- 1
  }
  
  registerDoParallel(cores=workers)  
  cl <- makeCluster(workers, type="FORK")  
  #------------------------------------------------------
  print("prepare splits, 'jobs' acts as a list of itoken iterators!"); flush.console() 
  
  chunks = split_into(tr[[fn]], workers)
  
  jobs = Map(function(doc) {
    itoken(iterable = doc,  preprocess_function = tolower, 
           tokenizer = word_tokenizer, chunks_number = 1, progessbar = TRUE) 
  }, chunks)
  
  remove(chunks)
  
  # Now all below function calls will benefit from multicore machines
  # Each job will be evaluated in separate process
  
  #-------------------------------------------------------
  print("Feature hashing (google text2vec approach)"); flush.console() 
  h_vectorizer = hash_vectorizer(hash_size = 2 ^ 10, ngram = c(2L, 3L)) 
  
  print('Text to numeric.... (using multicore benefits) '); flush.console()  
  dtm_train = create_dtm(jobs, h_vectorizer)
  #vocab = create_vocabulary(it_train)
  save(dtm_train, file='dtm.Rda')
  
  # stop multi core processing, release the handler  
  stopCluster(cl)  
  gc()
  
  #----------------------------------------------------------
  #sort( sapply(ls(),function(x){object.size(get(x))})) 
  load(file='dtm.Rda')
  #sz <- 5
  print('singular vector decomposition from sparse matrix'); flush.console()  
  svd.tr <- irlba( dtm_train, nu=sz, nv=sz)
  
  d <- as.data.frame(svd.tr$d)
  total <- sum(d$`svd.tr$d`)
  d$`svd.tr$d` <- d$`svd.tr$d`/total
  d$cumsum <- cumsum(d$`svd.tr$d`)
  head(d, sz)
  
  threshold <- 1.0 # we took all svd
  d <- as.data.frame(d)
  d <-   d$cumsum[d$cumsum <= threshold]
  
  u <- as.data.frame(svd.tr$u)[1:length(d)]
  v <- as.data.frame(svd.tr$v)
  
  #plot(1:length(svd.tr$d), svd.tr$d)
  #plot(svd.tr$u[, 1], svd.tr$u[, 2], main = "SVD", xlab = "U1", ylab = "U2")
  
  remove(v,d,svd.tr,total,threshold, dtm_train,jobs,h_vectorizer,cl, workers)
  gc(); # signals O/S to take back surrender memory
  
  print( paste('finished processing feature no. ', fn)  ); flush.console()  
  
  cn <- names(tr)[fn]
  cn <- paste(cn,1:sz, sep = '.')
  colnames(u) <- cn 
  
  #cnt <- 2
  save(u, file =  paste('classification/u', cnt, 'Rda' , sep = '.' ))
  
 # return (u)
  
}

#-----------------------------------------------------

