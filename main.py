import utils


predictors, response, data = utils.get_data2()

# Best subset selection
# utils.subset_selection(predictors, response)

# Apply regularization
utils.apply_reg(predictors, response)
