"""six center dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np
from math import cos, sin, pi
from sklearn.datasets import make_moons

### inter_twinning_moon
_SIZE = 2
INTER_TWINNING_MOON_NUM_CLASSES = 2
INTER_TWINNING_MOON_NUM_POINTS = 10000

INTER_TWINNING_MOON_DESCRIPTION = """
"""
INTER_TWINNING_MOON_CITATION = """
"""

class inter_twinning_moon(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for inter_twinning_moon dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=INTER_TWINNING_MOON_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'position': tfds.features.Tensor(shape=(_SIZE,), dtype=tf.float64),
            'label': tfds.features.ClassLabel(num_classes=INTER_TWINNING_MOON_NUM_CLASSES),
        }),
        supervised_keys=('position', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=INTER_TWINNING_MOON_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(split='all')),
    ]

  def _generate_examples(self, split):
    """Yields examples."""
    r = 20
    positions, labels = make_moons(INTER_TWINNING_MOON_NUM_POINTS, noise=0.0, random_state=1)
    trans = -np.mean(positions, axis=0) 
    positions = r*(positions + trans)
    
    if split == 'upper':
        delete_ele = np.where(labels == 1)
        labels = np.delete(labels, delete_ele, 0)
        positions = np.delete(positions, delete_ele, 0)

    elif split == 'lower':
        delete_ele = np.where(labels == 0)
        labels = np.delete(labels, delete_ele, 0)
        positions = np.delete(positions, delete_ele, 0)

    data = list(zip(positions, labels))
    for index, (position, label) in enumerate(data):
      record = {"position": position, "label": label}
      yield index, record

class inter_twinning_moon_upper(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for inter_twinning_moon dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=INTER_TWINNING_MOON_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'position': tfds.features.Tensor(shape=(_SIZE,), dtype=tf.float64),
            'label': tfds.features.ClassLabel(num_classes=INTER_TWINNING_MOON_NUM_CLASSES),
        }),
        supervised_keys=('position', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=INTER_TWINNING_MOON_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(split='upper')),
    ]

  def _generate_examples(self, split):
    """Yields examples."""
    r = 20
    positions, labels = make_moons(INTER_TWINNING_MOON_NUM_POINTS, noise=0.0, random_state=1)
    trans = -np.mean(positions, axis=0) 
    positions = r*(positions + trans)
    
    if split == 'upper':
        delete_ele = np.where(labels == 1)
        labels = np.delete(labels, delete_ele, 0)
        positions = np.delete(positions, delete_ele, 0)

    elif split == 'lower':
        delete_ele = np.where(labels == 0)
        labels = np.delete(labels, delete_ele, 0)
        positions = np.delete(positions, delete_ele, 0)

    data = list(zip(positions, labels))
    for index, (position, label) in enumerate(data):
      record = {"position": position, "label": label}
      yield index, record

class inter_twinning_moon_lower(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for inter_twinning_moon dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=INTER_TWINNING_MOON_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'position': tfds.features.Tensor(shape=(_SIZE,), dtype=tf.float64),
            'label': tfds.features.ClassLabel(num_classes=INTER_TWINNING_MOON_NUM_CLASSES),
        }),
        supervised_keys=('position', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=INTER_TWINNING_MOON_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(split='lower')),
    ]

  def _generate_examples(self, split):
    """Yields examples."""
    r = 20
    positions, labels = make_moons(INTER_TWINNING_MOON_NUM_POINTS, noise=0.0, random_state=1)
    trans = -np.mean(positions, axis=0) 
    positions = r*(positions + trans)
    
    if split == 'upper':
        delete_ele = np.where(labels == 1)
        labels = np.delete(labels, delete_ele, 0)
        positions = np.delete(positions, delete_ele, 0)

    elif split == 'lower':
        delete_ele = np.where(labels == 0)
        labels = np.delete(labels, delete_ele, 0)
        positions = np.delete(positions, delete_ele, 0)

    data = list(zip(positions, labels))
    for index, (position, label) in enumerate(data):
      record = {"position": position, "label": label}
      yield index, record