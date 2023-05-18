import argparse



def define_options():
    parser = argparse.ArgumentParser()
    # Naming
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help="Experiment's name."
    )
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Additional tag for experiment. If used, Tensorboard logs and checkpoints location will be chaged to `name-tag`.'
    )
    # Logs
    parser.add_argument(
        '--logdir',
        type=str,
        default='/src/logs',
        help="STDOUT is redirected to a log file which will be available at this location. To edit the folder where experiment's results are logged, please edit the YAML configuration files, specifically the `default_root_dir` field for the trainer options."
    )
    # Train or test
    train_test_parser = parser.add_mutually_exclusive_group(required=False)
    train_test_parser.add_argument(
        '--train',
        dest='train',
        action='store_true',
        help='Train model.'
    )
    train_test_parser.add_argument(
        '--test',
        dest='train',
        action='store_false',
        help='Test model. You should provide a checkpoint to load when testing/validating a model.'
    )
    parser.set_defaults(train=True)
    # Checkpoints
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='Path to the checkpoint of a trained model. This is needed for validating/testing. This is ignored if flag `--tests` is present.'
    )
    parser.add_argument(
        '--tests',
        type=str,
        default=None,
        help='Name of the YAML config file containing a list of checkpoints for trained models. This is useful when you want to define a list of models to test, then run the validation/test loop for all of them.'
    )
    return parser