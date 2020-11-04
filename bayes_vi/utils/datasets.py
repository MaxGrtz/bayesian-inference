from typing import List

import pandas as pd
import tensorflow as tf


def make_dataset_from_df(df: pd.DataFrame,
                         y: List[str],
                         x: List[str] = None,
                         format_features_as: str = 'tensor') -> tf.data.Dataset:
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df has to be of Type {}'.format(pd.DataFrame))

    if format_features_as not in ['tensor', 'dict']:
        raise ValueError('format_features_as has to be in ["tensor", "dict"]')

    if not isinstance(y, list):
        raise TypeError('y has to be a list')

    dict_map = lambda x_, y_: ({k: v[..., tf.newaxis] for k, v in x_.items()}, y_)

    if isinstance(x, list) and x != []:
        features = df[x]
        targets = df[y]

        if format_features_as == 'tensor':
            return tf.data.Dataset.from_tensor_slices((features.values, targets.values))
        else:
            return tf.data.Dataset.from_tensor_slices((dict(features), targets.values)).map(dict_map)

    else:
        targets = df[y]
        return tf.data.Dataset.from_tensor_slices(targets.values)
