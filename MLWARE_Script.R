setwd("H:/Analytics Vidhya/MLWARE_2")
library(readr)
train <- read_csv("H:/Analytics Vidhya/MLWARE_2/train_MLWARE2.csv")
View(train)
library(readr)
test <- read_csv("H:/Analytics Vidhya/MLWARE_2/test_MLWARE2.csv")
View(test)
#GBM
library(gbm)
library(caret)
train$ID <- NULL
test$ID <- NULL
train_index <- sample(1:nrow(train), nrow(train)*0.65)
train_new <- train[train_index,]
test_new <- train[-train_index,]
fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE)
set.seed(825)
gbmFit <- train(rating ~ ., data=train_new,
                method="gbm",
                trControl=fitControl,
                verbose=FALSE)
pred_gbm <- predict(gbmFit,test)
importance = summary(gbmFit, plotit=TRUE)
mysolution_new = data.frame( ID = test$ID, rating = pred_gbm, stringsAsFactors = FALSE)
submission_gbm = mysolution_new
write.csv(submission_gbm, file = "submission_gbm.csv", row.names = FALSE)
##XGB.CV###
cv <- xgb.cv(data = dtrain, nrounds = 500, nfold = 5, metrics = list("rmse"),
             max_depth = 8, eta = 0.05,subsample=0.6, colsample_bytree=0.85, objective = "reg:linear")
print(cv)
print(cv, verbose=TRUE)
model_xgb <- xgboost(data = dtrain, label=as.matrix(target),nrounds = 1000, metrics = list("rmse"),
                     max_depth = 8, eta = 0.1,subsample=0.7, colsample_bytree=0.7, objective = "reg:linear")
pred <- predict(model_xgb, as.matrix(test1))
mysolution_new = data.frame( ID = test$ID, rating = pred, stringsAsFactors = FALSE)
submission_xgb = mysolution_new
write.csv(submission_xgb, file = "submission_xgb.csv", row.names = FALSE)
##XGBOOST
library(xgboost)
train1 <- train
test1 <- test
train1$ID <- NULL
test1$ID <- NULL
train1$userId <- as.numeric(train1$userId)
train1$itemId <- as.numeric(train1$itemId)
test1$userId <- as.numeric(test1$userId)
test1$itemId <- as.numeric(test1$itemId)
train1$ratio <- (train1$userId)/(train1$itemId)
train1$log_ratio <- log(train1$ratio)
test1$ratio <- (test1$userId)/(test1$itemId)
test1$log_ratio <- log(test1$ratio)
target <- train1$rating
train1$rating <- NULL

dtrain<-xgb.DMatrix(data=data.matrix(train1),label=data.matrix(target),missing=NA)
dtest<-xgb.DMatrix(data=data.matrix(test1),missing=NA)

# creating second matrix (data and label) using sample_test
#dval<-xgb.DMatrix(data=data.matrix(sample_test),label=data.matrix(y),missing=NA)
#watchlist<-list(val=dtest,train=dtrain)
t1 <- Sys.time()
fin_pred={}
for (eta in c(0.9) )#(0.9,0.8)
{
  t <- Sys.time()
  for (colsample_bytree in c(0.6))
  {
    for(subsample in c(0.8))
    {
      param <- list(  objective           = "reg:linear", 
                      booster             = "gbtree",
                      eta                 = eta,
                      max_depth           = 10, #7 initially it was 8 
                      subsample           = subsample,
                      colsample_bytree    = colsample_bytree
      )
      gc()
      set.seed(1429)
      # creating the model 
      clf <- xgb.train(   params              = param, 
                          data                = dtrain, 
                          nrounds             = 1500, 
                          verbose             = 1,
                          #early.stop.round    = 150,
                          #watchlist           = watchlist,
                          maximize            = TRUE,
                          eval_metric       = "rmse"
      )
      
      pred_exp=predict(clf,data.matrix(test1),missing=NA)
      print(head(fin_pred))
      fin_pred<-cbind(fin_pred,pred_exp)
    }
  }
  print(Sys.time() - t)
}

print(Sys.time() - t1) 
final_pred=rowMeans(fin_pred)
mysolution_new = data.frame( ID = test$ID, rating = final_pred, stringsAsFactors = FALSE)
submission_xgb = mysolution_new
write.csv(submission_xgb, file = "submission_xgb.csv", row.names = FALSE)

######Grid Search######
library(xgboost)

xgbGrid <- expand.grid(
  nrounds = c(250, 500, 1000),
  max_depth = c(6, 8, 10),
  eta = c(0.75, 0.8, 0.9),
  gamma = c(0, 1, 2),
  colsample_bytree =  c(1, 0.6, 0.4),
  subsample       =   c(0.6,0.8,1),
  min_child_weight = c(1, 2)
  
)
library(caret)
xgbTrControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  verboseIter = FALSE,
  returnData = FALSE,
  allowParallel = TRUE
)

xgbTrain <- train(
  x = dtrain, 
  y = target,
  objective = "reg:linear",
  trControl = xgbTrControl,
  tuneGrid = xgbGrid,
  method = "xgbTree"
)

# get the top model and its results
head(xgbTrain$results[with(xgbTrain$results, 
                           order(RMSE)), ], 1)


########################### getting the output of 20 diff xgb models ######################################
#eta <- c(0.1,0.01,0.05,0.015,0.05,0.01,0.1,0.025,0.01,0.01,0.01,0.025,0.01,0.05,0.015,0.025,0.01,0.025,0.05,0.015)
eta <- c(0.01,0.1,0.2)#0.75,0.85,0.95
#max_depth <- c(8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8)
max_depth <- c(8,8,8)

#subsample <- c(0.8,0.9,0.9,0.95,0.8,0.8,0.85,0.85,0.85,0.95,0.85,0.8,0.9,0.85,0.85,0.9,0.8,0.95,0.8,0.85)
subsample <- c(0.8,0.8,0.9)
#colsample_bytree <- c(0.7,0.7,0.8,0.7,0.7,0.9,0.7,0.7,0.7,0.7,0.8,0.7,0.8,0.7,0.7,0.7,0.8,0.7,0.9,0.7)
colsample_bytree <- c(0.7,0.7,0.8)

predicted_df = {}
               
for (i in 1:3)
  {
  param <- list(  objective           = "reg:linear", 
                  booster             = "gbtree",
                  eta                 = eta[i],
                  max_depth           = max_depth[i] ,
                  subsample           = subsample[i],
                  colsample_bytree    = colsample_bytree[i]
  )
  
  set.seed(600)
  # creating the model 
  xgb_model <- xgboost(   params              = param, 
                          data                = dtrain,
                          label               = data.matrix(target), 
                          nrounds             = 500,
                          verbose             = 1,
                          maximize            = FALSE,
                          eval_metric       = "rmse",
                          missing=NA
  )
  
  prediction <- predict(xgb_model,data.matrix(test1),missing=NA)
  predicted_df<-cbind(predicted_df,prediction)
 }

# getting the average of prediction
final_pred=rowMeans(predicted_df)
mysolution_new = data.frame( ID = test$ID, rating =final_pred, stringsAsFactors = FALSE)
submission_xgb = mysolution_new
write.csv(submission_xgb, file = "submission_xgb.csv", row.names = FALSE)
