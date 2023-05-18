"""Contains constants to be used in the `models` module. Should just create constants and not import any other module from this project.
"""

# Logging constants
'''Keys used to log losses: defined here as they may need to be available elsewhere, like in callbacks or other scripts. Every time you want to log a new loss, you should append the following strings to it.
Example:
do
    `pl_module.log(f"Cool_{LOSS_TRAIN_KEY}")`
instead of
    `pl_module.log("Cool_loss/train")`
or
    `pl_module.log("Cool_loss_train")`
'''
LOSS_TRAIN_KEY = 'loss/train'
LOSS_VAL_KEY = 'loss/val'
LOSS_TEST_KEY = 'loss/test'