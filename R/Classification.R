# install.packages('randomForest')
# install.packages('ggplot2')
# install.packages('corrplot')
# install.packages('dplyr')
# install.packages('dummies')
# install.packages('caret')
# install.packages('e1071')
# install.packages('rpart')
# install.packages('rpart.plot')
# install.packages("xgboost")
# install.packages('naivebayes')
# install.packages('RDS')

library(dplyr)
#--------------------------------------------------------------------------------------------------------------#
# Change directory to data

getwd()
setwd('../Desktop/Python/data/heart-disease-uci/')
getwd()

#--------------------------------------------------------------------------------------------------------------#
# Read data

df <- read.csv('heart.csv', fileEncoding="UTF-8-BOM")

#--------------------------------------------------------------------------------------------------------------#
# Table of Contents
# 1. Data Exploration & Visualization
# 2. Preprocessing Data
# 3. Training Models
# 4. Optimizing Models
#--------------------------------------------------------------------------------------------------------------#

df[1,1]
head(df)
tail(df)

#--------------------------------------------------------------------------------------------------------------#
# 1. Data Exploration & Visualization

summary(df)

dimensions <- dim(df)

nrow(df)
ncol(df)

# List of columns
cols <- colnames(df)
for (i in 1:ncol(df)){
  print(cols[i])
}

#--------------------------------------------------------------------------------------------------------------#
# Correlation heatmap
library(corrplot)
corrs <- cor(df)
corrplot(corrs, method = "number")

# List Counts of categorical variables, however has to be strings
# head(subset(mtcars, select = 'gear'))

# alternatively given binary (0,1) can simply sum the target column to get the number of class "1"
sums <- colSums(df)
sprintf("Percent Heart Disease %.2f", (sums[14] / nrow(df) * 100))

# averages for each column
colMeans(df)

#--------------------------------------------------------------------------------------------------------------#
# AGE

# most common ages
head(sort(table(df$age),decreasing=TRUE))
# least common ages
head(sort(table(df$age)))

# Boxplot of age according to Heart Disease
boxplot(age~sex,
        data=df,
        main="Age by Gender",
        xlab="Gender",
        ylab="Age",
        col="orange",
        border="brown"
)

#--------------------------------------------------------------------------------------------------------------#
# GENDER

sprintf("Percent of male participants %.2f", (table(df$sex)[2]/nrow(df) * 100))

#--------------------------------------------------------------------------------------------------------------#
# CHOLESTEROL

# with high cholesterol defined as > 200 mg/dl
sprintf("Percent of people with high cholesterol %.2f", (length(df[df$chol > 200,'chol'])/nrow(df)*100))


#--------------------------------------------------------------------------------------------------------------#
# HEART DISEASE VS NO HEART DISEASE

# Slicing the data according to whether the person has heart disease or not
heart_disease <- df[df$target == 1,]
no_heart_disease <- df[df$target == 0,]

# printing the distributions of each column according to whether a person has heart disease or not
for (i in 1:(ncol(df)-1)){
  attach(df)
  par(mfrow=c(1,2))
  hist(heart_disease[,i],
       main="Heart Disease", 
       xlab=cols[i], 
       border="blue", 
       col="green",
       las=1, 
       breaks=length(table(df[,i])))
  
  hist(no_heart_disease[,i],
       main="No Heart Disease", 
       xlab=cols[i], 
       border="blue", 
       col="green",
       las=1, 
       breaks=length(table(df[,i])))
}

#--------------------------------------------------------------------------------------------------------------#
# 2. Preprocessing Data

#--------------------------------------------------------------------------------------------------------------#
# shuffling the rows
df2 <- df[sample(nrow(df)),]

head(df)
head(df2)

#--------------------------------------------------------------------------------------------------------------#
# dropping cp, slope, and thal due to inconsistent/ambiguous encoding schemes
df <- select(df2,-c(cp,slope,thal))
head(df)

# detecing outliers
# outlier_values <- boxplot.stats(df$age)$out
# https://www.rdocumentation.org/packages/tsoutliers/versions/0.3/topics/remove.outliers

#--------------------------------------------------------------------------------------------------------------#
# One Hot Encoding
# restecg

library(dummies)
df2 <- dummy.data.frame(df, names=c('restecg'), sep='_')
head(df)
head(df2)

# remove one of the dummy variables to prevent correlation
df <- select(df2,-c(restecg_2))
head(df)

#--------------------------------------------------------------------------------------------------------------#
# Scaling
# Scaling isn't going to be done here due to the presence of categorical variables.
# but heres how it could be done anyways.
# df <- scale(df)

#--------------------------------------------------------------------------------------------------------------#
# Train, Validation & Test Split

train_test_split <- function(data,train_split,valid_split){
  train_bound <- round(nrow(data) * train_split)
  valid_bound <- train_bound + round(nrow(data)*valid_split)
  
  train <- data[1:(train_bound - 1),]
  valid <- data[train_bound:(valid_bound - 1),]
  test <- data[valid_bound:nrow(data),]
  
  split_data <- list('train'=train,'validation'=valid,'test'=test)
  return(split_data)
}

df2 <- train_test_split(df,.6,.2)

nrow(df2$train)
nrow(df2$validation)
nrow(df2$test)

#--------------------------------------------------------------------------------------------------------------#
# 3. Training Models
library(caret)

#--------------------------------------------------------------------------------------------------------------#
# Random Forest
library(randomForest)

rf <- randomForest(as.factor(target) ~ ., data = df2$train, importance = TRUE)
valid_x <- select(df2$validation,-c(target))
predictions <- predict(rf, valid_x)
valid_y <- as.factor(df2$validation$target)

length(predictions)
length(valid_y)

summary(rf)

temp <- confusionMatrix(predictions,valid_y)
temp

accuracies['RF'] <- temp$overall[1]

#--------------------------------------------------------------------------------------------------------------#
# Decision Tree
library(rpart)
library(rpart.plot)

dt <- rpart(target~., data = df2$train, method = 'class')
predictions <- predict(dt, valid_x, type = 'class')
head(predictions)

summary(dt)

temp <- confusionMatrix(predictions,valid_y)
temp

accuracies['DT'] <- temp$overall[1]

#--------------------------------------------------------------------------------------------------------------#
# Logistic Regression

lr <- glm(target ~., data = df2$train, family = binomial)

summary(lr)

anova(lr, test="Chisq")

predictions <- predict(lr,valid_x, type = "response")
predictions <- ifelse(predictions > 0.5,1,0)

head(predictions)

temp <- confusionMatrix(as.factor(predictions),valid_y)
temp

accuracies['Log Reg'] <- temp$overall[1]

#--------------------------------------------------------------------------------------------------------------#
# XGBoost
library(xgboost)

train <- as.matrix(select(df2$train, -c(target)))
lab <- as.matrix(select(df2$train,c(target)))
train <- xgb.DMatrix(train, label = lab)

xgbust <- xgboost(data = train, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

valid_xg <- as.matrix(valid_x)
predictions <- predict(xgbust, valid_xg)
predictions <- ifelse(predictions > 0.5,1,0)
head(predictions)

temp <- confusionMatrix(as.factor(predictions),valid_y)
temp

accuracies['XGBoost'] <- temp$overall[1]

#--------------------------------------------------------------------------------------------------------------#
# Naive Bayes
library(naivebayes)

nb <- naive_bayes(as.factor(target) ~ ., data = df2$train)
predictions <- predict(nb,valid_x)

temp <- confusionMatrix(as.factor(predictions),valid_y)
temp

accuracies['NB'] <- temp$overall[1]

#--------------------------------------------------------------------------------------------------------------#
# Support Vector Machine
library(e1071)

svmodel <- svm(target ~ ., data = df2$train, kernel = "linear")
predictions <- predict(svmodel,valid_x)
predictions <- ifelse(predictions > 0.5,1,0)
head(predictions)

temp <- confusionMatrix(as.factor(predictions),valid_y)
temp
accuracies['SVM'] <- temp$overall[1]

#--------------------------------------------------------------------------------------------------------------#
# K Neighbors
library(class)

train <- select(df2$train, -c(target))
valid <- select(df2$validation, -c(target))
train_labels <- df2$train$target

kneigh <- knn(train, valid, train_labels, k = 3, prob=TRUE)

head(kneigh)

temp <- confusionMatrix(kneigh,valid_y)
temp

accuracies['KNN'] <- temp$overall[1]

#--------------------------------------------------------------------------------------------------------------#
# Model Comparison

accuracies <- as.data.frame(accuracies)
sort(accuracies,decreasing = TRUE)

#--------------------------------------------------------------------------------------------------------------#
# 4. Optimizing Models

model <- train(as.factor(target)~., data=df2$train, method="naive_bayes", tuneLength=5)
print(model)

nb <- naive_bayes(as.factor(target) ~ ., data = df2$train, laplace = 0, usekernel = FALSE, adjust = 1)
predictions <- predict(nb,valid_x)

confusionMatrix(as.factor(predictions),valid_y)

# Saving & Loading Best Model

saveRDS(nb, "NB.rds")
nb <- readRDS("NB.rds")