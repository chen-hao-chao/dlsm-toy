import tensorflow as tf
import tensorflow_datasets as tfds
import inter_twinning_moon.inter_twinning_moon

def get_dataset(config):
  """Construct data loaders for training and evaluation.
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
  Returns:
    train_ds: (tf dataset iter) The dataset iterator.
  """
  # Compute the batch size.
  batch_size = config.training.batch_size

  # Set buffer size.
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None

  # Build datasets.
  if config.data.dataset in ['inter_twinning_moon', 'inter_twinning_moon_upper', 'inter_twinning_moon_lower']:
    dataset_builder = tfds.builder(config.data.dataset)
    train_split_name = 'train'
  else:
    raise NotImplementedError(f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  preprocess_fn = lambda x: x

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  return train_ds
