from typing import List

import pandas as pd
import tensorflow as tf


def make_dataset_from_df(df: pd.DataFrame,
                         target_names: List[str],
                         feature_names: List[str] = None,
                         format_features_as: str = 'tensor') -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset` from a `pd.DataFrame` and lists of feature and target names.

    Parameters
    ----------
    df: `pd.DataFrame`
        A `pd.DataFrame` of the dataset.
    target_names: `list` of `str`
        A list of target names to select form `df`.
    feature_names: `list` of `str`, optional
        A list of feature names to select from `df`.
    format_features_as: {'tensor', 'dict'}
        One of 'tensor' or 'dict' depending on whether the `tf.data.Dataset` constructed
        should return the features as a single `tf.Tensor` or as a `dict[str, tf.Tensor]`.
        Note: If 'dict' is chosen, the feature names can be used to
              index the `features` parameter of the likelihood function.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df has to be of Type {}'.format(pd.DataFrame))

    if format_features_as not in ['tensor', 'dict']:
        raise ValueError('format_features_as has to be in ["tensor", "dict"]')

    if not isinstance(y, list):
        raise TypeError('target_names has to be a List[str]')

    dict_map = lambda x, y: ({k: v[..., tf.newaxis] for k, v in x.items()}, y)

    if isinstance(feature_names, list) and feature_names != []:
        features = df[feature_names]
        targets = df[target_names]

        if format_features_as == 'tensor':
            return tf.data.Dataset.from_tensor_slices((features.values, targets.values))
        else:
            return tf.data.Dataset.from_tensor_slices((dict(features), targets.values)).map(dict_map)

    else:
        targets = df[target_names]
        return tf.data.Dataset.from_tensor_slices(targets.values)