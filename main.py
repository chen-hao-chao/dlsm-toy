"""Entrance file"""

import run_lib_score, run_lib_classifier, run_lib_eval, run_lib_classifier_plot
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_bool("plot", False, "Set true to save training log and plot.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or visualization.")
flags.mark_flags_as_required(["workdir", "config", "mode"])

def main(argv):
  if FLAGS.mode == "train":
    tf.io.gfile.makedirs(FLAGS.workdir)
    if FLAGS.config.model.type == "classifier":
      if FLAGS.plot:
        run_lib_classifier_plot.train(FLAGS.config, FLAGS.workdir)
      else:
        run_lib_classifier.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.config.model.type == "score_model":
      run_lib_score.train(FLAGS.config, FLAGS.workdir)
    else:
      raise ValueError(f"Model {FLAGS.config.model} not recognized.")

  elif FLAGS.mode == "eval":
    run_lib_eval.evaluate(FLAGS.config, FLAGS.workdir)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")

if __name__ == "__main__":
  app.run(main)