### the R code for the ML regression model used in J. Lewis webinar
#OpenGeoHub Summer School
library("tidymodels")
library("tidyverse")

## load the data
data(ames, package = "modeldata")

## looking at the data
head(colnames(ames), n=82)

glimpse(ames)

par(mfrow=c(1,2))
ggplot(ames, aes(x = Gr_Liv_Area)) +
  geom_histogram() +
  scale_x_log10()

ggplot(ames, aes(x = Gr_Liv_Area, Sale_Price)) +
  geom_point(alpha = .3) +
  scale_x_log10() +
  scale_y_log10() +
  facet_wrap(~Bldg_Type) +
  geom_smooth(method = lm)

par(mfrow=c(1,2))
ggplot(ames, aes(x = Longitude)) +
  geom_histogram()

ggplot(ames, aes(x = Latitude)) +
  geom_histogram()

par(mfrow=c(1,2))
ggplot(ames, aes(x = Sale_Price)) +
  geom_histogram()

ggplot(ames, aes(x = Sale_Price)) +
  geom_histogram() +
  scale_x_log10()

mini_ames <- ames %>%
  dplyr::select(Latitude, Longitude, Sale_Price)
head(mini_ames, n=5)

## developing the training and test data set
set.seed(7014) #for reproducability
ames_split <- initial_split(ames, prop = .70)
#prop defines the amount of split
ames_split
ames_train <- training(ames_split)
#
ames_cv <- vfold_cv(ames_train)# for later to do cross-validation

## the recipe
mod_rec <-
  recipe(Sale_Price ~ Longitude + Latitude,
                  data = ames_train) %>%
  step_log(Sale_Price, base = 10)

## Specify model information
knn_mod <-
  #specify the model as nearest neighbor
  nearest_neighbor() %>%
  #set tuning parameters-we chose 2 from the knn model - no. of neighbors
  #and distance exponent
  set_args(neighbors = tune(),dist_power = tune()) %>%
  # set R package that is associated with the model
  set_engine("kknn")%>%
  set_mode("regression")

## construct a workflow process
ml_wflow <-
  workflow() %>%
  #add the recipe
  add_recipe(mod_rec) %>%
  #add the model
  add_model(knn_mod)

## tune the model with specified grid
ml_wflow_tune <-
  ml_wflow %>%
  tune_grid(resamples = ames_cv, # cv object
                  grid = 10, # grid values
                  metrics = metric_set(rmse)) # performance metric of interest

# show performance metric
res  <- ml_wflow_tune %>%
  collect_metrics()
res

## plot the hyperparameter values
autoplot(ml_wflow_tune, metric = "rmse")

## Select best parameters
best_params <-
  ml_wflow_tune %>%
  select_best(metric = "rmse")

## Finalize workflow
# Now add this parameter (best_params) to the workflow using the
# finalize_workflow() function.
ames_reg_res <-
  ml_wflow %>%
  finalize_workflow(best_params)

## Fit final model using the entire training/test data using ames_split
ames_wfl_fit <- ames_reg_res %>%
  last_fit(ames_split)

## Collect test performance metrics
test_performance <- ames_wfl_fit %>%
  collect_metrics()
test_performance #print the results

## Collect test set predictions themselves
test_predictions <- ames_wfl_fit %>%
  collect_predictions()
test_predictions #print the results

## Plot predicted vs observed test data
ggplot(test_predictions,
  aes(x=Sale_Price, y=.pred)) +
    geom_point(alpha = .4) +
    geom_abline(col = "red")+
    coord_obs_pred() +
    ylab("Predicted") +
  ggtitle("Sales Price vs Predicted Price (log scale)")


