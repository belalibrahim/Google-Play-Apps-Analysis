import utils


predictors, response, data = utils.get_data2()

# Best subset selection
# utils.subset_selection(predictors, response)

# Apply regularization
# utils.apply_reg(predictors, response)

# Apply overfit
# utils.apply_over(predictors, response)

# Models comparison
utils.models_compare(predictors, response)
