# Version 29-JAN-2019 MJL
# Ubuntu 18.04.1 Lts
# Rstudio Version 1.1.456
# R version 3.4.4

# Clear workspace
rm(list = ls()); gc()

# File description comment, including purpose of program, inputs, outputs

# Program inputs
path.raw <- "/home/michael/Desktop/MAL_URL/web_data.csv"  # Path to raw dataset

# Program outputs
path.train <- "/home/michael/Desktop/MAL_URL/Objects/train_tsvd300.RDS"  # Training set destination
path.test <- "/home/michael/Desktop/MAL_URL/Objects/test_tsvd300.RDS"  # Test set destination

#--ToDo--#
#2. finish author comment
# 3. file description
# 4. fix working directory

# pivot normalization vs cosine normalization for document length  **what is the function computing?
# http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/

# library() statements
required.packages <- c("caret", "quanteda", "stopwords", "Matrix", "irlba")
lapply(required.packages, require, character.only = TRUE)

# Function definitions

# soft cosine measure, levenshtein distance

row_normalize <- function(m, verbose = FALSE){
  # https://stats.stackexchange.com/questions/61085/cosine-similarity-on-sparse-matrix
  # Function which applies normalization to rows such that all features per document of document feature matrix sum to 1
  #
  # Args:
  #   m: A document-feature matrix of class DgCMatrix to apply row normalization
  #   verbose: If TRUE, prints transformed document feature matrix. Default is FALSE
  #
  # Returns:
  #   A document-feature matrix in which normalization has been applied to observations  
  d <- Diagonal(x = 1 / sqrt(rowSums(m^2)))  # Create diagonal matrix of document length normalization weight scheme 
  return(t(crossprod(m, d)))  # Transpose cross product of document feature matrix and diagonal matrix d
}

# Dataset:  https://www.kaggle.com/antonyj453/urldataset # * need to verify dataset or produce new one

# LSA


#-----------------------------------------Raw Data Processing---------------------------------------#
data.raw <- read.csv(path.raw,  # Import raw data to created object data.raw
                     header = TRUE,  # Include header
                     stringsAsFactors = FALSE,  # Don't include strings as factors
                     fileEncoding = "UTF-8")  # Use UTF-8 as file encoding

str(data.raw)  # Observe structure of raw data set
names(data.raw) <- c("Text", "Label")  # Rename column names
data.raw$Label <- factor(data.raw$Label)  # Assign variable Label as factor
levels(data.raw$Label) <- c("malicious", "benign")  # Rename variable 'Label' factor levels
 
# Data Exploration
length(which(!complete.cases(data.raw)))  # Check for incomplete cases
prop.table(table(data.raw$Label))  # Investigate distribution of class labels

# Stratified 90/10 split of dataset
set.seed(7893)  # Set seed for reproducibility

index <- createDataPartition(data.raw$Label, times = 1,  # Create training/test 'Label' stratified data partition
                               p = 0.9, list = FALSE)

train <- data.raw[index, ]  # Training Partition
test <- data.raw[-index, ]  # Test Partition

# Verify proportions
prop.table(table(train$Label))  # Prop table for training data
prop.table(table(test$Label))  # Prop table for testing data

#----------------------------------------Trainingset Processing-------------------------------------#
# Process Url strings for training set
train.alphanum <- gsub("[^[:alnum:]]+", " ", train$Text)  # Split text by non-alphabets (excluding numerics), replace with white space
length(which(!complete.cases(train.alphanum)))  # Check for incomplete cases

# Tokenize training data, construct uni-grams and bi-grams as features
set.seed(7263)  # Set seed for reproducibility

train.tokens <- tokens(train.alphanum, what = "word",  # Tokenize urls as words
                       remove_numbers = TRUE, remove_punct = FALSE,  # Reomve numbers, preserve punctuation
                       remove_symbols = FALSE, remove_hyphens = FALSE,  # Preserve symbols and hyphens
                       remove_twitter = FALSE)  # Preserve twitter links if applicable

train.tokens <- tokens_tolower(train.tokens)  # Convert all characters to lowercase

# Process tokens
train.tokens <- tokens_select(train.tokens, stopwords(),  # Remove stopwords; * create custom list for url domain
                              selection = "remove")

train.tokens <- tokens_wordstem(train.tokens, language = "english")  # Stem Tokens; * Is it possible to stem in multiple languages simultaneously?

# N-grams
train.tokens <- tokens_ngrams(train.tokens, n = 1:2, concatenator = "_")  # Generate uni-grams and bi-grams from proccessed tokens

# Construct Document Feature Matrix from training data
train.dfm <- dfm(train.tokens, tolower = FALSE)  # Build document feature matrix from tokens
dim(train.dfm)  # Check dimensions of dfm
topfeatures(train.dfm, 50)  # Explore top features

# Process Document Feature Matrix
train.dfm <- dfm_trim(train.dfm, min_termfreq = 2, min_docfreq = 2)  # Remove sparse terms which lack sufficiant statistical information /
                                                                     # keep words occuring >= 2 times, and in >= 2 documents

dim(train.dfm)  # Check dimensions of dfm

train.dfm.dgc <- as(train.dfm, "dgCMatrix")  # Coerce dfm to dgCMatrix class
dim(train.dfm.dgc)  # Verify dimensions

# Tf - Contruct term-frequency row normalization
train.normalized.tf <- row_normalize(train.dfm.dgc)  # Standardize each vector to length |A| = 1 

# Calculate Inverse Doucument Frequency (IDF) vector
train.idf <- log(nrow(train.dfm.dgc)/colSums(train.dfm.dgc))  # note: this vector is important for translating future data /
                                                              # i.e. test data, to model vector space

# Build Term frequency-Inverse Document Frequency (TF-IDF) matrix
train.tfidf <- t(train.normalized.tf) * train.idf  # Term document format of normalized tf-idf
dim(train.tfidf)  # Verify dimensions

# Transpose tf-idf matrix
train.tfidf <-t(train.tfidf)  # Document term format of normalized tf-idf
dim(train.tfidf)  # Verify dimensions

path.tfidf <- "/home/michael/Desktop/MAL_URL/Objects/train_tfidf.RDS"
saveRDS(train.tfidf, file = path.tfidf)  # Save tf-idf as R object
gc()  # Clean up unused objects in memory

#---------------------------------Partial Singular Value Decomposition------------------------------#
# Reduce dimensionality for latent semantic analysis
# Implicitly Restarted Lanczos Bi-Diagonalization algorithm
# 300 right singular vectors will be used as a starting point for calculating partial svd

start.time <- Sys.time()  # Create start time for svd operation

set.seed(8739)  # Set seed for reproducibility

train.irlba <- irlba(t(train.tfidf), nv = 300, maxit = 600) # Compute 300 largest singular values

# Total processing time (SVD) 
total.time <- Sys.time() - start.time  # Calculate elapsed time of svd call
total.time  # Time difference of 3.750604 mins

saveRDS(train.irlba, file = "/home/michael/Desktop/MAL_URL/Objects/psvd_train.RDS")  # Save tsvd as R object

# Calculations for creating SVD semantic space for new data introduced to model
sigma.inverse <- 1 / train.irlba$d # train.irlba$d = approximate singular values
u.transpose <- t(train.irlba$u) # train.irlba$u = approximate left singular vectors
document <- train.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Inspect the first 10 components of projected document and the corresponding row in document semantic space, values should be the same
document.hat[1:10]
train.irlba$v[1, 1:10]

# Create df using semantic space generated by SVD   *** explain train.irlba$v component
train.svd <- data.frame(Label = train$Label, train.irlba$v)

#------------------------------------------Testset Processing---------------------------------------#
# Process Url strings for test set
test.alphanum <- gsub("[^[:alnum:]]+", " ", test$Text)  # Split text by non-alphabets (excluding numerics), replace with white space
length(which(!complete.cases(train.alphanum)))  # Check for incomplete cases

# Tokenize testset, construct uni-grams and bi-grams as features
set.seed(7263)  # Set seed for reproducibility

test.tokens <- tokens(test.alphanum, what = "word",  # Tokenize urls as words 
                      remove_numbers = TRUE, remove_punct = FALSE,  # Remove numbers, preserve punctuation
                      remove_symbols = FALSE, remove_hyphens = FALSE,   # Preserve symbols and hyphens
                      remove_twitter = FALSE)  # Preserve twitter links if applicable

# Process tokens
test.tokens <- tokens_tolower(test.tokens) # Convert all characters to lowercase

test.tokens <- tokens_select(test.tokens, stopwords(),  # Remove stopwords; * create custom list for url domain
                             selection = "remove")
test.tokens <- tokens_wordstem(test.tokens, language = "english")  # Stem Tokens; * Is it possible to stem in multiple languages simultaneously?

# N-grams
test.tokens <- tokens_ngrams(test.tokens, n = 1:2, concatenator = "_")  # Generate uni-grams and bi-grams from proccessed tokens

test.dfm <- dfm(test.tokens, tolower = FALSE)  # Convert n-grams to quanteda document-term frequency matrix.

# Process dfm
test.dfm <- dfm_select(test.dfm, pattern = train.dfm,  # Strip n-grams that did not appear in training data
                       selection = "keep")

test.dfm.dgc <- as(test.dfm, "dgCMatrix")  # Coerce dfm to dgCMatrix class
dim(test.dfm.dgc) # Verify dimensions

test.normalized.tf <- row_normalize(test.dfm.dgc)  # Apply row normalization to test dfm.  See function definitions for more information

# Build Term frequency-Inverse Document Frequency (TF-IDF) matrix for test set
test.tfidf <- t(test.normalized.tf) * train.idf # Use train.idf to transform test data to tf-idf model vector space of training data
dim(test.tfidf) # Verify dimensions

# Transpose tf-idf matrix
test.tfidf <-t(test.tfidf) # Document term format of normalized tf-idf
dim(test.tfidf) # Verify dimensions

# Check/Fix incomplete cases
summary(test.tfidf[1,])
test.tfidf[is.na(test.tfidf)] <- 0.0
summary(test.tfidf[1,])

saveRDS(test.tfidf, file = "/home/michael/Desktop/MAL_URL/Objects/test_tfidf.RDS")  # Save tf-idf as R object

gc()  # Clean up unused objects in memory

test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tfidf))  # Complete transformation of test data to model vector space 

test.svd <- data.frame(Label = test$Label, as.matrix(test.svd.raw))  # Construct test dataframe

saveRDS(train.svd, file = path.train)  # Save training set as R object
saveRDS(test.svd, file = path.test)  # Save test set as R object