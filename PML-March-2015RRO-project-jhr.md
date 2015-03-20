# Practical Machine Learning Course Project
WhitefishDontJump  
March 2015  

The source of data for this project is from

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **Qualitative Activity Recognition of Weight Lifting Exercises.** Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3MCvfqKcP

### Synopsis

#### Exploration, Cleaning and Model Development

1.  Response is categorical (factor with 5 levels, A to E).
2.  The predictors, after cleaning, are entirely integer or numeric, except user_name.
3.  I will retain user_name as a factor variable.  It is possible that individual users uniquely condition the range of responses in other predictors.
4.  I plan to use random forest (via caret's train function), and will use a three fold cross validation of rf results.
5. There are 53 predictor variables after cleaning the raw dataset.  Given the size of the dataset, and selecting 50% of the training set, a 3 fold validation will yield more than 6000 observations per fold, on a base of more than 9800 observations in one half of the training dataset.

#### Results Summary

1. Initial model: Random Forest yielded a highly accurate predictive model (modela) with an estimated out of sample error rate of 1 %. It required 15 minutes to complete execution on my PC and correctly predicted the 20 submission test cases.

2. Two further Random Forest models: Modelb (removed user_name) and Modelc (18 predictors based varImp). Modelc ran ~ 10 x faster than Modela and Modelb, with an estimated out of sample error of 1.4%. Both Modelb and Modelc predicted the 20 submission test cases correctly.

3. I conclude that Modelc, with 18 predictors, provides a better balance of accuracy vs speed that either modela or modelb.


--------------------------------------------------------------------------



```r
training <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))

testing <- read.csv("pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))

## View(training)

### lots of missing data and/or NAs ##

# I will be removing any columns with NAs as predictors in the training set.

cols2get <- colSums(is.na(training)) == 0
cleantraining <- training[, cols2get]

### leaves 60 columns same cleaning for testing data (optional choice on my
### part)

cols2get2 <- colSums(is.na(testing)) == 0
cleantesting <- testing[, cols2get2]

### remove row 'X', as well as time date window columns which are not related
### to prediction of 'classe'

cleantraining <- cleantraining[, c(-1, -3, -4, -5, -6, -7)]
cleantesting <- cleantesting[, c(-1, -3, -4, -5, -6, -7)]
```


```r
require(parallel)
require(caret)

## create new partition in 'training set' and create training1, testing1 set.
## I will use training1 to create the model and testing1 to estimate out of
## sample error.

set.seed(150310)  ## for repeatability

trindex <- as.vector(createDataPartition(cleantraining$classe, p = 0.5, list = FALSE))

training1 <- cleantraining[trindex, ]
testing1 <- cleantraining[-trindex, ]

## Initial model with 1/2 of training set and 3 fold cross validation, using
## the random forest method and caret's train() function.  Afterwards, test
## the model with other half of training set to estimate out of sample
## accuracy.

controla <- trainControl(method = "cv", number = 3, allowParallel = TRUE)

modela <- train(classe ~ ., data = training1, method = "rf", trControl = controla)
```




```r
## determine in sample error, which vars are relevant and relative importance

modela
```

```
## Random Forest 
## 
## 9812 samples
##   53 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 6540, 6542, 6542 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9804318  0.9752386  0.002391992  0.003021179
##   29    0.9835912  0.9792383  0.002743784  0.003467594
##   57    0.9770680  0.9709877  0.006640541  0.008391455
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```

```r
confusionMatrix(modela)  ## in sample error
```

```
## Cross-Validated (3 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.3  0.0  0.0  0.0
##          B  0.0 18.9  0.3  0.0  0.1
##          C  0.0  0.2 17.0  0.4  0.0
##          D  0.0  0.0  0.2 15.9  0.1
##          E  0.0  0.0  0.0  0.0 18.2
```

```r
varImp(modela)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 57)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          60.15
## yaw_belt               51.81
## roll_forearm           45.34
## magnet_dumbbell_y      44.04
## magnet_dumbbell_z      43.98
## pitch_belt             42.83
## accel_dumbbell_y       24.69
## accel_forearm_x        19.06
## roll_dumbbell          18.91
## magnet_dumbbell_x      17.18
## accel_belt_z           17.02
## magnet_forearm_z       16.84
## magnet_belt_z          15.77
## accel_dumbbell_z       15.68
## magnet_belt_y          14.50
## total_accel_dumbbell   14.25
## gyros_belt_z           13.20
## magnet_belt_x          11.63
## yaw_arm                11.30
```

```r
## All factor levels of user_name, except Eurico, had relative importance of
## less than 1. Eurico's activity differed from those of the other users: his
## user_name factor level had relative importance of 2.35 on the scaled
## varImp results, compared to 0.54 for Charles, and smaller values for other
## users. I conclude that user_name had minimal importance in the final
## model.


## Out of Sample Error estimate

## I will estimate the out of sample error by applying the model to the
## second part of the training set (named testing1).

predicta <- predict(modela, testing1)

confusionMatrix(predicta, testing1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2784   15    0    0    0
##          B    5 1870   16    5    2
##          C    1   13 1692   21    8
##          D    0    0    3 1579    7
##          E    0    0    0    3 1786
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9899          
##                  95% CI : (0.9877, 0.9918)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9872          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9852   0.9889   0.9820   0.9906
## Specificity            0.9979   0.9965   0.9947   0.9988   0.9996
## Pos Pred Value         0.9946   0.9852   0.9752   0.9937   0.9983
## Neg Pred Value         0.9991   0.9965   0.9976   0.9965   0.9979
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1906   0.1725   0.1610   0.1821
## Detection Prevalence   0.2853   0.1935   0.1769   0.1620   0.1824
## Balanced Accuracy      0.9979   0.9909   0.9918   0.9904   0.9951
```

```r
## fit is very good, accuracy 0.99, and does not seem to be overfitting on
## the testing1 set. I am satisfied with the model result and also believe
## that using the entire training set can only increase the chance of
## overfitting.

## Based on this favorable result, I use this model, 'modela', on the class
## testing set:

answers <- predict(modela, cleantesting)

answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
## submitting 'answers' (via the function provided), scored 20 of 20 correct.
```

#### Initial Model comments:

The random forest method, properly tuned for 3 fold cross validation and a controlled training set size, created a prediction model with 99% accuracy in an out of sample test.


----------------------------------------------------------------------------


#### What about individual user conditioning the predictors?

Here is a scatter plot of total_accel_arm and roll_belt, colored by user_name to examine the variation across users for one of the most important features in the model, roll_belt (see varImp output, above). Note: The selection of total_accel_arm for the plot was somewhat arbitrary; it isn't very important in the model but it is highly variable for each user, so the resulting plot displays the roll_belt values for better visualization of user differences. 



```r
    qplot(total_accel_arm, roll_belt, data=training1, color = user_name)
```

![](PML-March-2015RRO-project-jhr_files/figure-html/Plots -1.png) 

Comments on the plot: Can this model be generalized to predict classe for other users not in the data set? While the plot confirms differences among users, it is not very important in the final model, as reported by varImp(). 


----------------------------------------------------------------------------


### Modelb: Building a model without user_name as predictor

Data prep:  remove user_ name column from previous split data:


```r
training2 <- training1[, -1]  ## remove user_name column
testing2 <- testing1[, -1]

## no need to change controla (train control parameters).  will call new
## model 'modelb'

modelb <- train(classe ~ ., data = training2, method = "rf", trControl = controla)

modelb
```

```
## Random Forest 
## 
## 9812 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 6542, 6542, 6540 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9816553  0.9767861  0.001857278  0.002352296
##   27    0.9836935  0.9793678  0.002147470  0.002718092
##   52    0.9763564  0.9700859  0.004426436  0.005598552
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
confusionMatrix(modelb)  ## in sample error
```

```
## Cross-Validated (3 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.3  0.0  0.0  0.0
##          B  0.0 18.8  0.2  0.0  0.1
##          C  0.0  0.2 17.0  0.3  0.0
##          D  0.0  0.0  0.2 16.0  0.1
##          E  0.0  0.0  0.0  0.0 18.1
```

```r
## modelb is comparable to modela

predictb <- predict(modelb, testing2)

confusionMatrix(predictb, testing2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2784   16    0    0    0
##          B    5 1870   16    5    2
##          C    1   12 1694   20    8
##          D    0    0    1 1582    7
##          E    0    0    0    1 1786
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9904          
##                  95% CI : (0.9883, 0.9922)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9879          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9852   0.9901   0.9838   0.9906
## Specificity            0.9977   0.9965   0.9949   0.9990   0.9999
## Pos Pred Value         0.9943   0.9852   0.9764   0.9950   0.9994
## Neg Pred Value         0.9991   0.9965   0.9979   0.9968   0.9979
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1906   0.1727   0.1613   0.1821
## Detection Prevalence   0.2854   0.1935   0.1769   0.1621   0.1822
## Balanced Accuracy      0.9978   0.9909   0.9925   0.9914   0.9952
```

```r
answersb <- predict(modelb, cleantesting)

answersb == answers  ## test compare predictions
```

```
##  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
## [15] TRUE TRUE TRUE TRUE TRUE TRUE
```

Removing user_name as a predictor caused little change in overall accuracy with the training2 set, and it’s performance on testing2 and on the submitted test set was excellent. I conclude that removing user_name from predictors was an acceptable approach to building the predictive model.



----------------------------------------------------------------------------

### Modelc: Reducing predictors to those with higher importance.


```r
## Use VarImp to remove lowest importance variables (scaled imp less than
## 10.0), leaving 18 predictors.

importance <- varImp(modelb)$importance
importance$vars <- rownames(importance)
## rownames(importance) <- NULL
importance <- importance[order(importance$Overall, decreasing = TRUE), ]

impCols <- importance[(importance$Overall >= 10), 2]
impCols
```

```
##  [1] "roll_belt"            "pitch_forearm"        "yaw_belt"            
##  [4] "magnet_dumbbell_z"    "roll_forearm"         "magnet_dumbbell_y"   
##  [7] "pitch_belt"           "accel_dumbbell_y"     "roll_dumbbell"       
## [10] "accel_forearm_x"      "magnet_dumbbell_x"    "accel_belt_z"        
## [13] "magnet_forearm_z"     "magnet_belt_z"        "accel_dumbbell_z"    
## [16] "total_accel_dumbbell" "magnet_belt_y"        "gyros_belt_z"
```

```r
training3 <- training2[, c(impCols, "classe")]
testing3 <- testing2[, c(impCols, "classe")]

### modelc will use reduced dimensionality

modelc <- train(classe ~ ., data = training3, method = "rf", trControl = controla)

modelc
```

```
## Random Forest 
## 
## 9812 samples
##   18 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 6541, 6542, 6541 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9777820  0.9718901  0.002909416  0.003679518
##   10    0.9764572  0.9702233  0.002008272  0.002539815
##   18    0.9706477  0.9628780  0.004140984  0.005235697
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
confusionMatrix(modelc)  ## in sample error
```

```
## Cross-Validated (3 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.3  0.4  0.0  0.0  0.0
##          B  0.1 18.6  0.3  0.0  0.1
##          C  0.1  0.3 16.9  0.4  0.1
##          D  0.0  0.1  0.2 15.9  0.1
##          E  0.0  0.0  0.0  0.0 18.1
```

```r
## modelc is comparable to modela & modelb

predictc <- predict(modelc, testing3)

confusionMatrix(predictc, testing3$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2781   22    0    0    1
##          B    7 1846   17    1    8
##          C    0   28 1690   33   10
##          D    2    2    4 1572    5
##          E    0    0    0    2 1779
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9855         
##                  95% CI : (0.983, 0.9878)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9817         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9968   0.9726   0.9877   0.9776   0.9867
## Specificity            0.9967   0.9958   0.9912   0.9984   0.9998
## Pos Pred Value         0.9918   0.9824   0.9597   0.9918   0.9989
## Neg Pred Value         0.9987   0.9934   0.9974   0.9956   0.9970
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2835   0.1882   0.1723   0.1602   0.1813
## Detection Prevalence   0.2858   0.1915   0.1795   0.1616   0.1815
## Balanced Accuracy      0.9967   0.9842   0.9895   0.9880   0.9932
```

```r
answersc <- predict(modelc, cleantesting)

answersc == answers  ## test compare predictions
```

```
##  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
## [15] TRUE TRUE TRUE TRUE TRUE TRUE
```

modelc loses very little accuracy on test set (compared to modela) and achieves the same results on the course submission test set.  I conclude that the dimension reduction strategy was successful and reduced training time to about 10% of the time required for the original modela.

Thank you for reading and reviewing my work.

--------------------------------------------------------------------------

Here is sessionInfo(), for reference:


```r
print(sessionInfo(), locale = FALSE)
```

```
## R version 3.1.3 (2015-03-09)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 8 x64 (build 9200)
## 
## attached base packages:
## [1] parallel  stats     graphics  grDevices utils     datasets  methods  
## [8] base     
## 
## other attached packages:
## [1] randomForest_4.6-10 caret_6.0-41        ggplot2_1.0.1      
## [4] lattice_0.20-30    
## 
## loaded via a namespace (and not attached):
##  [1] BradleyTerry2_1.0-6 brglm_0.5-9         car_2.0-25         
##  [4] class_7.3-12        codetools_0.2-11    colorspace_1.2-6   
##  [7] compiler_3.1.3      digest_0.6.8        e1071_1.6-4        
## [10] evaluate_0.5.5      foreach_1.4.2       formatR_1.0        
## [13] grid_3.1.3          gtable_0.1.2        gtools_3.4.1       
## [16] htmltools_0.2.6     iterators_1.0.7     knitr_1.9          
## [19] labeling_0.3        lme4_1.1-7          MASS_7.3-39        
## [22] Matrix_1.1-5        mgcv_1.8-5          minqa_1.2.4        
## [25] munsell_0.4.2       nlme_3.1-120        nloptr_1.0.4       
## [28] nnet_7.3-9          pbkrtest_0.4-2      plyr_1.8.1         
## [31] proto_0.3-10        quantreg_5.11       Rcpp_0.11.5        
## [34] reshape2_1.4.1      rmarkdown_0.5.1     scales_0.2.4       
## [37] SparseM_1.6         splines_3.1.3       stringr_0.6.2      
## [40] tools_3.1.3         yaml_2.1.13
```
