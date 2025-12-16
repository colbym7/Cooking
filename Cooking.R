library(tidyverse)
library(tidymodels)
library(jsonlite)
library(tidytext)
library(textrecipes)
library(lightgbm)
library(bonsai)
library(finetune)
library(ranger)
library(stacks)
library(kknn)

train <- read_file('C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\train.json') %>%
  fromJSON()
test <- read_file('C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\test.json') %>%
  fromJSON()

trainlong <- train %>%
  unnest(ingredients)
testlong <- test %>%
  unnest(ingredients)
trainlong <- trainlong %>%
  mutate(across(where(is.character), tolower)) %>%
  mutate(
    cuisine_in_ingredient = str_detect(ingredients, cuisine)
  )

print(unique(trainlong$ingredients))

unique_ingredients <- trainlong %>%
  count(cuisine, ingredients, name = "count_in_cuisine") %>%
  add_count(ingredients, name = "n_cuis") %>%
  filter(n_cuis == 1) %>%
  select(-n_cuis) %>% 
  arrange(desc(count_in_cuisine))
unique10 <- trainlong %>%
  count(cuisine, ingredients, name = "count_in_cuisine") %>%
  add_count(ingredients, name = "n_cuis") %>%          # how many cuisines each ingredient appears in
  filter(n_cuis == 1, count_in_cuisine >= 5) %>%      # unique to one cuisine + appears ≥10 times
  select(cuisine, ingredients)
trainlong <- trainlong %>%
  left_join(unique10 %>% mutate(is_special = TRUE),
            by = c("cuisine", "ingredients")) %>%
  mutate(
    is_special = if_else(is.na(is_special), FALSE, TRUE),
    special = as.integer(is_special),
    cuisine_in_ingredient = as.integer(str_detect(ingredients, cuisine))
  ) %>%
  select(-is_special)


trained <- trainlong %>%
  group_by(id, cuisine) %>%
  summarize(
    # 1 if any ingredient contains the cuisine name
    cuisine_in_ingredient = as.integer(any(str_detect(ingredients, cuisine))),
    
    # 1 if any ingredient is marked special (unique to this cuisine)
    special = as.integer(any(special == 1)),
    num_ingredients = n(),
    .groups = "drop"
  )
trained$cuisine <- as.factor(trained$cuisine)

# 'trainlong' is your long-format training dataset (one row per ingredient per recipe)

unique5 <- trainlong %>%
  count(cuisine, ingredients, name = "count_in_cuisine") %>%
  add_count(ingredients, name = "n_cuis") %>%
  filter(n_cuis == 1, count_in_cuisine >= 15) %>%   
  pull(ingredients)
cuisines <- unique(trainlong$cuisine)
prepare_recipe_features <- function(df_long, special_ingredients, cuisine_list) {
  
  # 1. Flag special ingredients
  df_long <- df_long %>%
    mutate(special = as.integer(ingredients %in% special_ingredients))
  
  # 2. Summarize total ingredients and number of special ingredients per recipe
  summary_df <- df_long %>%
    group_by(id) %>%
    summarize(
      num_ingredients = n(),
      num_special_ingredients = sum(special),
      .groups = "drop"
    )
  
  # 3. Wide indicator columns for special ingredients
  special_wide <- df_long %>%
    filter(special == 1) %>%
    group_by(id, ingredients) %>%          # aggregate duplicates
    summarize(indicator = 1, .groups = "drop") %>%
    mutate(ingredient_clean = str_replace_all(ingredients, "[^a-z0-9]", "_")) %>%
    select(id, ingredient_clean, indicator) %>%
    pivot_wider(
      names_from = ingredient_clean,
      values_from = indicator,
      values_fill = 0
    )
  
  # 4. Wide indicator columns for cuisine substring matches
  for(cui in cuisine_list) {
    flag <- df_long %>%
      group_by(id) %>%
      summarize(
        !!paste0("ingredient_matches_", cui) := as.integer(any(str_detect(ingredients, regex(cui, ignore_case = TRUE)))),
        .groups = "drop"
      )
    summary_df <- summary_df %>% left_join(flag, by = "id")
  }
  
  # 5. Combine with special ingredient indicators
  summary_df %>%
    left_join(special_wide, by = "id")
}
train_features <- prepare_recipe_features(trainlong, unique5, cuisines) %>%
  left_join(trainlong %>% select(id, cuisine) %>% distinct(), by = 'id')
test_features <- prepare_recipe_features(testlong, unique5, cuisines)
test_cols <- setdiff(names(test_features), "id")

# Keep only these columns in training
train_features <- train_features %>%
  select(id, all_of(test_cols), cuisine)  # keep the response variable






recipe1 <- recipe(cuisine ~ ., data = train_features) #%>%

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(recipe1) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,20)),
                            min_n(),
                            levels = 2)

folds <- vfold_cv(train_features, v = 4, repeats = 1)
CV_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(accuracy))
bestTune <- CV_results %>%
  select_best(metric = 'accuracy')

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_features)

rf_preds <- final_wf %>%
  predict(new_data=test_features, type='class')

kaggle_submission1 <- data.frame(
  id = test$id,
  cuisine = rf_preds$.pred_class
)
vroom_write(x=kaggle_submission1, 
            file="C://Users//cjmsp//Desktop//Stat348//WhatsCooking//Preds//rf_preds.csv", 
            delim=",")

recipe2 <- recipe(cuisine ~ ingredients, data = train) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=500) %>%
  step_tfidf(ingredients)

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(recipe2) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,20)),
                            min_n(),
                            levels = 2)

folds <- vfold_cv(train, v = 4, repeats = 1)
CV_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(accuracy))
bestTune <- CV_results %>%
  select_best(metric = 'accuracy')

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

tfidf_preds <- final_wf %>%
  predict(new_data=test, type='class')

kaggle_submission2 <- data.frame(
  id = test$id,
  cuisine = tfidf_preds$.pred_class
)
vroom_write(x=kaggle_submission2, 
            file="C://Users//cjmsp//Desktop//Stat348//WhatsCooking//Preds//tfidf_preds.csv", 
            delim=",")






recipe3 <- recipe(cuisine ~ ., data = train) %>%
  step_mutate(
    is_dairy = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("milk|cheese|cream|butter", ignore_case = TRUE)))
    ),
    has_peppers = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("pepper|chili|chilli|capsicum", ignore_case = TRUE)))
    )
  ) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=500) %>%
  step_tfidf(ingredients)

prepped_recipe <- prep(recipe3)
baked_data <- bake(prepped_recipe, new_data = NULL)


rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(recipe3) %>%
  add_model(rf_mod)


tuning_grid <- grid_regular(mtry(range=c(1,20)),
                            min_n(),
                            levels = 4)

folds <- vfold_cv(train, v = 4, repeats = 1)
CV_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(accuracy))
bestTune <- CV_results %>%
  select_best(metric = 'accuracy')

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

tfidf_preds <- final_wf %>%
  predict(new_data=test, type='class')

kaggle_submission3 <- data.frame(
  id = test$id,
  cuisine = tfidf_preds$.pred_class
)
vroom_write(x=kaggle_submission3, 
            file="C://Users//cjmsp//Desktop//Stat348//WhatsCooking//Preds//tfidf2_preds.csv", 
            delim=",")




recipe4 <- recipe(cuisine ~ ., data = train) %>%
  step_mutate(
    n_ingredients = map_int(ingredients, length),
    is_dairy = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("milk|cheese|cream|butter", ignore_case = TRUE)))
    ),
    has_peppers = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("pepper|chili|chilli|capsicum", ignore_case = TRUE)))
    )
  ) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=1000) %>%
  step_tfidf(ingredients)


knn_spec <- nearest_neighbor(neighbors=tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')


xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  mtry = tune(),
  sample_size = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

glm_spec <- multinom_reg(
  penalty = tune(),   # lambda
  mixture = tune()    # 0 = ridge, 1 = lasso, in between = elastic net
) %>%
  set_engine("glmnet")

knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')

knn_wf <- workflow() %>% add_model(knn_spec) %>% add_recipe(recipe4)
xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(recipe4)
glm_wf <- workflow() %>% add_model(glm_spec) %>% add_recipe(recipe4)


folds <- vfold_cv(train, v = 5, repeats = 1)
ctrl <- control_stack_resamples()
knn_res  <- tune_grid(knn_wf,  resamples = folds, grid = 5, control = ctrl)
xgb_res <- tune_grid(xgb_wf, resamples = folds, grid = 5, control = ctrl)
glm_res <- tune_grid(glm_wf, resamples = folds, grid = 5, control = ctrl)

model_stack <- stacks() %>%
  add_candidates(knn_res) %>%
  add_candidates(xgb_res) %>%
  add_candidates(glm_res)
stack_blended <- blend_predictions(model_stack)
stack_fit <- fit_members(stack_blended)
stack_preds <- predict(stack_fit, new_data = test, type = "class")

kaggle_submission4 <- data.frame(
  id = test$Id,
  Cover_Type = stack_preds$.pred_class
)
vroom_write(x=kaggle_submission4, 
            file="C://Users//cjmsp//Desktop//Stat348//WhatsCooking//Preds//stack_preds.csv", 
            delim=",")


### Light Gradient Boosted Model
lgbm_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune(),
  loss_reduction = tune()
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

lgbm_wf <- workflow() %>%
  add_model(lgbm_spec) %>%
  add_recipe(recipe4)

folds <- vfold_cv(train, v=5)
lgbm_grid <- grid_space_filling(
  trees(),
  tree_depth(),
  learn_rate(range = c(-4, -1)),   # log10 scale
  min_n(),
  loss_reduction(),
  size = 20
)
tuned_results <- lgbm_wf %>%
  tune_race_anova(
    resamples = folds,
    grid = lgbm_grid,
    metrics = metric_set(accuracy)
  )
best_tune <- tuned_results %>%
  select_best(metric = 'accuracy')

final_lgbm_wf <- lgbm_wf %>%
  finalize_workflow(best_tune)

final_lgbm_fit <- final_lgbm_wf %>%
  fit(data = train)
lgbm_preds <- predict(final_lgbm_fit, test) %>%
  bind_cols(test)

kaggle_submission5 <- data.frame(
  id = test$Id,
  Cover_Type = lgbm_preds$.pred_class
)
vroom_write(x=kaggle_submission5, 
            file="C://Users//cjmsp//Desktop//Stat348//WhatsCooking//Preds//lgbm_preds.csv", 
            delim=",")



recipe4 <- recipe(cuisine ~ ., data = train) %>%
  step_mutate(
    n_ingredients = map_int(ingredients, length),
    is_dairy = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("milk|cheese|cream|butter", ignore_case = TRUE)))
    ),
    has_peppers = map_lgl(
      ingredients,
      ~ any(str_detect(.x, regex("pepper|chili|chilli|capsicum", ignore_case = TRUE)))
    )
  ) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=1000) %>%
  step_tfidf(ingredients)
prepped_recipe <- prep(recipe4)
baked_data <- bake(prepped_recipe, new_data = NULL)
baked_2 <- bake(prepped_recipe, new_data = test)
write.csv(baked_data, file = 'C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\datarobot.csv')
write.csv(baked_2, file = 'C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\datarobottest.csv')

robot_preddf <- read.csv('C:\\Users\\cjmsp\\Desktop\\Stat348\\WhatsCooking\\robotresult.csv')
kaggle_submission6 <- data.frame(
  id = test$id,
  cuisine = robot_preddf$cuisine_PREDICTION
)
vroom_write(x=kaggle_submission6, 
            file="C://Users//cjmsp//Desktop//Stat348//WhatsCooking//Preds//robotnn1_preds.csv", 
            delim=",")
