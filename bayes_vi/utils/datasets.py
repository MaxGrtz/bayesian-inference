"""Module providing utility function to construct datasets from pandas DataFrames."""

import pandas as pd
import tensorflow as tf


def make_dataset_from_df(df,
                         target_names,
                         feature_names=None,
                         format_features_as='dict'):
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

    Returns
    -------
    `tf.data.Dataset`
        A `tf.data.Dataset` consisting of the specified `features` and `targets`.
        Note: if `feature_names` are not provided, the dataset will return a default
              `feature` tensor of the same shape as `targets`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df has to be of Type {}'.format(pd.DataFrame))

    if format_features_as not in ['tensor', 'dict']:
        raise ValueError('format_features_as has to be in ["tensor", "dict"]')

    if not isinstance(target_names, list) and target_names != []:
        raise TypeError('target_names has to be a non-empty List[str]')

    dict_map = lambda x, y: ({k: v for k, v in x.items()}, y)

    if isinstance(feature_names, list) and feature_names != []:
        targets = df[target_names[0]] if len(target_names) == 1 else df[target_names]

        if format_features_as == 'tensor':
            features = df[feature_names[0]] if len(feature_names) == 1 else df[feature_names]
            return tf.data.Dataset.from_tensor_slices((features.values, targets.values))
        else:
            features = df[feature_names]
            return tf.data.Dataset.from_tensor_slices((dict(features), targets.values)).map(dict_map)

    else:
        targets = df[target_names[0]] if len(target_names) == 1 else df[target_names]
        return tf.data.Dataset.from_tensor_slices((tf.zeros_like(targets.values), targets.values))
