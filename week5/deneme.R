# %%
install.packages("tidymodels")
# Description
# A simulated data set containing information on ten
# thousand customers.The aim here is to predict
# which customers will default on their credit card debt.

# A data frame with 10000 observations on the following 4 variables.
# * default A factor with levels No and Yes indicating whether the 
#       customer defaulted on their debt
# * student A factor with levels No and Yes indicating whether 
#       the customer is a student
# * balance The average balance that the customer has remaining 
#       on their credit card after making their monthly payment
# * income Income of customer

library(tidymodels)
library(ISLR)
library(dplyr)

data = ISLR::Default
# data
#write.csv(data,"data.csv")


{
  # A few Observations and data length
  head(data)
  cat('Rows : ' , nrow(data))
  cat('Columns : ',length(data))
}

{
  # levels in factor
  table(data$default)

  table(data$student)

}

{
  # Summary statistics
  summary(data$balance)
  summary(data$income)
}



{
  # Splitting Data
  set.seed(42)
  data_split = initial_split(data,prop = 0.80,strata = default)
  
  data_train <- data_split %>% training()
  data_test <- data_split %>% testing()
  
  # Train and Test sets 
  table = data.frame(c(nrow(data_train),nrow(data_test)),
                     c(sum(data_train$default == 'No'),sum(data_test$default == 'No')),
                     c(sum(data_train$default == 'Yes'),sum(data_test$default == 'Yes')))
  colnames(table) = c('Rows','No','Yes')
  rownames(table) = c('Train','Test')
  table  
  
}




{
  logistic_model <- logistic_reg() %>% 
    set_engine('glm') %>% set_mode('classification')
  # Fitting regression model
  logistic_fit <- logistic_model %>% 
    fit(default~. , data = data_train)
  # Obtaining the estimated parameters
  tidy(logistic_fit)
  
}




{
  # Estimated labels for train
  data_labels_train <- logistic_fit %>%
    predict(new_data = data_train , type = 'class')
  head(data_labels_train,5)
  
  # Estimated probabilities for train 
  data_preds_train <- logistic_fit %>%
    predict(new_data = data_train,type = 'prob')
  head(data_preds_train,5) 
  
}
{

  # Estimated labels for test
  data_labels_test <- logistic_fit %>%
    predict(new_data = data_test , type = 'class')
  head(data_labels_test,25)
  
  # Estimated probabilities for test
  data_preds_test <- logistic_fit %>%
    predict(new_data = data_test,type = 'prob')
  head(data_preds_test,5)
}

{
  # Show a few predicted variables on train set
  df_labels = t(data.frame(head(data_labels_train,10),head(data_train$default,10)))
  rownames(df_labels) = c('Predicted_Train','Actual')
  df_labels
}

{
  # Evaluating Model Performance Train set
  data_results_train <- data_train %>%
    select(default) %>%
    bind_cols(data_preds_train,data_labels_train)
  
  # Confusion matrix Train set
  conf_mat(data_results_train,
           truth = default,
           estimate = .pred_class)
  # Heat map of the confusion matrix
  conf_mat(data_results_train, truth = default, estimate = .pred_class) %>%
    autoplot(type='heatmap')
}

{
  data_results_test <- data_test %>%
    select(default) %>%
    bind_cols(data_preds_test,data_labels_test)

  # Confusion matrix test set
  conf_mat(data_results_test,
           truth = default,
           estimate = .pred_class)
  # Heat map of the confusion matrix
  conf_mat(data_results_test, truth = default, estimate = .pred_class) %>%
    autoplot(type='heatmap')
}

{ 
   # For Train Accuracy , Sensitivity and Specificity
  acc_train = accuracy(data_results_train,truth = default, estimate = .pred_class)
  sens_train = sens(data_results_train, truth = default, estimate = .pred_class)  
  spec_train = spec(data_results_train, truth = default, estimate = .pred_class)  
  
  # For Test ; Accuracy , Sensitivity and Specificity
  acc_test = accuracy(data_results_test,truth = default, estimate = .pred_class)
  sens_test = sens(data_results_test, truth = default, estimate = .pred_class) 
  spec_test = spec(data_results_test, truth = default, estimate = .pred_class) 
}
{
  # Print calculated Accuracy
  cat("Accuracy for Train : ", acc_train$.estimate)
  cat("\nAccuracy for Test : ",acc_test$.estimate)

  # Print calculated Sensitivity
  cat("Sensitivity for Train : ", sens_train$.estimate)
  cat("\nSensitivity for Test : ",sens_test$.estimate)

  cat("Specificity for Train : ", spec_train$.estimate)
  cat("\nSpecificity for Test : ",spec_test$.estimate)
}


{
  # Visualizing performance across thresholds
  data_results_test %>%
    roc_curve(truth = default,estimate = .pred_No) %>%
    autoplot()
}

# ------------------------------------------------------------
#                 Feature Engineering Part
# --------------------------------------------------------------
{ # Creating Model & All in Recipe
  # correlate and Normalzie together
  data_recipe_all <- recipe(default~. , data = data_train) %>%
    # Remove correlated variables
    step_corr(all_numeric(),threshold = 0.8) %>%
    # Normalize
    step_normalize(all_numeric()) %>%
    # Create dummy variables
    step_dummy (all_nominal(),-all_outcomes())
  
    # Train the recipe 
    data_recipe_prep <- data_recipe_all %>%
      prep(training = data_train)
    
    # Apply to training data 
    data_train_prep <- data_recipe_prep %>%
      bake(new_data = data_train)
    # Apply to test data
    data_test_prep <- data_recipe_prep %>%
      bake(new_data = data_test)
    
}

{ #For the train set
    logistic_fit_rec <- logistic_model %>%
    fit(default~., data= data_train_prep)

  # Obtain class predictions
  class_preds_train_Rec <- predict(logistic_fit_rec,new_data = data_train_prep,
                                 type = 'class')
  # Obtain estimated probabilities
  prob_preds_train_Rec <- predict(logistic_fit_rec,new_data = data_train_prep,
                                type = 'prob')
  
  # Combine test set results
  data_results_recipe_train <- data_train_prep %>%
    select(default) %>%
    bind_cols (class_preds_train_Rec,prob_preds_train_Rec)

}

{ #For the test set
  logistic_fit_rec2 <- logistic_model %>%
    fit(default~., data= data_test_prep)
  
  # Obtain class predictions
  class_preds_test_Rec <- predict(logistic_fit_rec2,new_data = data_test_prep,
                                   type = 'class')
  # Obtain estimated probabilities
  prob_preds_test_Rec <- predict(logistic_fit_rec2,new_data = data_test_prep,
                                  type = 'prob')
  
  # Combine test set results
  data_results_recipe_test <- data_test_prep %>%
    select(default) %>%
    bind_cols (class_preds_test_Rec,prob_preds_test_Rec)
  
  accuracy (data_results_recipe_test,truth = default,estimate = .pred_class)


}

{
   # Print calculated Accuracy
  cat("Accuracy for Train : ", acc_train_recipe$.estimate)
  cat("\nAccuracy for Test : ",acc_test_recipe$.estimate)

  # Print calculated Sensitivity
  cat("Sensitivity for Train : ", sens_train_recipe$.estimate)
  cat("\nSensitivity for Test : ",sens_test_recipe$.estimate)

  cat("Specificity for Train : ", spec_train_recipe$.estimate)
  cat("\nSpecificity for Test : ",spec_test_recipe$.estimate)


}

{
  { # performance
  acc_train_recipe = accuracy(data_results_recipe_train,truth = default,estimate =.pred_class)
  sens_train_recipe = sens(data_results_recipe_train, truth = default, estimate = .pred_class)
  spec_train_recipe = spec(data_results_recipe_train, truth = default, estimate = .pred_class)
  
  acc_test_recipe = accuracy(data_results_recipe_test,truth = default,estimate =.pred_class)
  sens_test_recipe = sens(data_results_recipe_test, truth = default, estimate = .pred_class)
  spec_test_recipe = spec(data_results_recipe_test, truth = default, estimate = .pred_class)
  
  conf_mat(data_results_recipe_train, truth = default, estimate = .pred_class) %>%
    autoplot(type='heatmap')
  conf_mat(data_results_recipe_test, truth = default, estimate = .pred_class) %>%
    autoplot(type='heatmap')
}

{
  df_comparison = t(data.frame(c(acc_train$.estimate,sens_train$.estimate,spec_train$.estimate),
              c(acc_test$.estimate,sens_test$.estimate,spec_test$.estimate),
              c(acc_train_recipe$.estimate,sens_train_recipe$.estimate,spec_train_recipe$.estimate),
              c(acc_test_recipe$.estimate,sens_test_recipe$.estimate,spec_test_recipe$.estimate)
              ))
  rownames(df_comparison) = c('Train without F.','Test without F.',
                        'Train w/ Feature E.','Test w/ Feature E.')              
  colnames(df_comparison) = c('Accuracy','Sensitivity','Specificity')
  df_comparison
}











