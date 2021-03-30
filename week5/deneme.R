# %%
install.packages("tidymodels")


library(tidymodels)
library(ISLR)
library(dplyr)

data = ISLR::Default
data

str(data)

set.seed(42)
data_split = initial_split(data,prop = 0.80,strata = default)

{
  data_train <- data_split %>% training()
  data_test <- data_split %>% testing()
  
  dim(data_train)
  dim(data_test)
}

{
  logistic_model <- logistic_reg() %>% 
    set_engine('glm') %>% set_mode('classification')
  # Fitting a linear regression model
  logistic_fit <- logistic_model %>% 
    fit(default~. , data = data_train)
  # Obtaining the estimated parameters
  tidy(logistic_fit)
  
}

{
  # Estimated labels
  data_labels <- logistic_fit %>%
    predict(new_data = data_test , type = 'class')
  head(data_labels,25)
  
  # Estimated probabilities
  data_preds <- logistic_fit %>%
    predict(new_data = data_test,type = 'prob')
  head(data_preds,5)
}

{ # Evaluating Model Performance
  data_results <- data_test %>%
    select(default) %>%
    bind_cols(data_preds,data_labels)

  # Confusion matrix
  conf_mat(data_results,
           truth = default,
           estimate = .pred_class)
}

{ # Evaluation Model Performance
  # Accuary
  accuracy(data_results,truth = default, estimate = .pred_class)
  # Sensivity 
  sens(data_results, truth = default, estimate = .pred_class)  
  # Specificity
  spec(data_results, truth = default, estimate = .pred_class)  
}

write.csv(data,"data.csv")














