# from: https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically

import glob, os
import pandas as pd
from tqdm.auto import tqdm
from tensorboard.backend.event_processing import event_accumulator



def extract_from_tfevents(
        data_tags: list,
        *models: str,
        rename_data: list = None,
        **kwargs
    ):
    # listify if needed
    if not isinstance(data_tags, (list,tuple)):
        data_tags = [data_tags]
    if rename_data is None:
        rename_data = [None] * len(data_tags)
    # sanity check
    assert len(data_tags) == len(rename_data), f"When renaming the data, make sure that you provide as many new names as there are tags to extract. You were trying to extract {len(data_tags)} tags but provided {len(rename_data)} new names."
    # extract
    for i, data_tag in enumerate(tqdm(data_tags)):
        tmp = extract_tag_from_tfevents(
            data_tag, *models, rename=rename_data[i], **kwargs
            ).drop(['wall_time', 'step'], axis=1)
        if i == 0:
            X = tmp
        else:
            X = pd.concat([X, tmp], axis=1)
    return X


def extract_tag_from_tfevents(
        data_tag: str,
        *names: str,
        rename_models: list = None,
        **kwargs
    ):
    # listify if needed
    if not isinstance(rename_models, (list,tuple)):
        rename_models = [rename_models] * len(names)
    # call helper function for each name
    for i, name in enumerate(names):
        tmp = extract_from_tag(data_tag, name, tag=rename_models[i], **kwargs)
        if i == 0:
            X = tmp
        else:
            X = pd.concat([X, tmp], axis=0)
    return X


def extract_from_tag(
        data_tag: str,
        modelname: str,
        tb_dir: str = "/src/tensorboard",
        pattern: str = "events.out.tfevents*",
        tag: str = None,
        alternatives: list = None,
        rename: str = None,
        nb_points: int = 1, # 0 for all points
    ):
    # in case data_tag is not found
    if alternatives is None:
        i = data_tag.rfind('/')
        j = data_tag.rfind('_')
        alternatives = [
            data_tag[:i] + val for val in ['/val', '/test', '/train']
        ] + [
            data_tag[:j] + val for val in ['_val', '_test', '_train']
        ]
    if not isinstance(alternatives, (list,tuple)):
        alternatives = [alternatives]
    # find file
    loc = f"{tb_dir}/{modelname}"
    try:
        to_find = f"{loc}/{pattern}"
        event_file = glob.glob(to_find)[-1]
    except Exception as ex:
        raise Exception(f'Failed to find file {to_find}') from ex
    # create event accululator from tensorboard
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance = { # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 1,
            event_accumulator.IMAGES: 1,
            event_accumulator.AUDIO: 1,
            event_accumulator.SCALARS: int(nb_points), # 0 for all, 1 for last
            event_accumulator.HISTOGRAMS: 1,
        }
    )
    ea.Reload()
    # read data
    try:
        X = ea.Scalars(data_tag)
        local_ex = None
    except Exception as ex:
        # show all available tags
        local_ex = Exception(f"Please make sure you provided an existing tag for file {event_file}. You provided {data_tag} but available tags are:\n{ea.Tags()}")
        # tag was not found, try alternatives
        for alternative_tag in alternatives:
            try:
                print(f"Could not find {data_tag} for model {modelname}, trying with alternative tag: {alternative_tag}...")
                X = ea.Scalars(alternative_tag)
                local_ex = None
            except Exception as last_ex:
                continue # if error, try next alternative tag
            break # if successful, break the for-loop
    if local_ex is not None: raise local_ex
    # load into dataframe
    X = pd.DataFrame(X)
    index = [modelname if tag is None else tag for _ in range(X.shape[0])]
    X = X.reindex(index)
    X = X.rename(columns={'value': data_tag if rename is None else rename})
    return X