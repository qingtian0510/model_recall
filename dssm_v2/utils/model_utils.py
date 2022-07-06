import os

from tensorflow.python.estimator.export import export_lib
from tensorflow.python.framework import errors
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.client import session as tf_session
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import (
    estimator_training as distribute_coordinator_training,
)
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.summary import summary
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util
from tensorflow.python.util import compat
from tensorflow.python.util import compat_internal
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import estimator_export
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys


def _create_global_step(graph):
    """Creates the global step tensor in graph.

    The global step tensor must be an integer type with name 'global_step' and
    be added to the collection `tf.GraphKeys.GLOBAL_STEP`.

    Args:
      graph: The graph in which to create the global step tensor.

    Returns:
      The global step `tf.Tensor`.
    """
    return training.create_global_step(graph)


def _create_and_assert_global_step(graph):
    """Creates and asserts properties of the global step.

    Args:
      graph: The graph in which to create the global step tensor.

    Returns:
      The global step `tf.Tensor`.
    """
    step = _create_global_step(graph)
    assert step is training.get_global_step()
    assert step.dtype.is_integer
    return step


def _call_model_fn(features, labels, mode, config, _model_fn, params):
    """Calls model function.

    Args:
      features: features dict.
      labels: labels dict.
      mode: `tf.estimator.ModeKeys`
      config: `tf.estimator.RunConfig`

    Returns:
      An `tf.estimator.EstimatorSpec` object.

    Raises:
      ValueError: if `model_fn` returns invalid objects.
    """
    model_fn_args = function_utils.fn_args(_model_fn)
    kwargs = {}
    if "labels" in model_fn_args:
        kwargs["labels"] = labels
    else:
        if labels is not None:
            raise ValueError(
                "model_fn does not take labels, but input_fn returns labels."
            )
    if "mode" in model_fn_args:
        kwargs["mode"] = mode
    if "params" in model_fn_args:
        kwargs["params"] = params
    if "config" in model_fn_args:
        kwargs["config"] = config

    logging.info("Calling model_fn.")
    model_fn_results = _model_fn(features=features, **kwargs)
    logging.info("Done calling model_fn.")
    if not isinstance(model_fn_results, model_fn_lib.EstimatorSpec):
        raise ValueError("model_fn should return an EstimatorSpec.")
    return model_fn_results


def _add_meta_graph_for_mode(
    builder,
    input_receiver_fn_map,
    checkpoint_path,
    save_variables=True,
    mode=ModeKeys.PREDICT,
    export_tags=None,
    check_variables=True,
    strip_default_attrs=True,
    tf_random_seed=1024,
    config=None,
    session_config=None,
    classifier=None,
    # inputs_keys=None,
    # outputs_keys=None
    export_dict=None,
):
    """Loads variables and adds them along with a `tf.MetaGraphDef` for saving.

    Args:
      builder: instance of `tf.saved_modle.builder.SavedModelBuilder` that will
        be used for saving.
      input_receiver_fn_map: dict of `tf.estimator.ModeKeys` to
        `input_receiver_fn` mappings, where the `input_receiver_fn` is a
        function that takes no argument and returns the appropriate subclass of
        `InputReceiver`.
      checkpoint_path: The checkpoint path to export.
      save_variables: bool, whether variables should be saved. If `False`, just
        the `tf.MetaGraphDef` will be saved. Note that `save_variables` should
        only be `True` for the first call to this function, and the
        `SavedModelBuilder` will raise an error if that is not the case.
      mode: `tf.estimator.ModeKeys` value indicating which mode will be
        exported.
      export_tags: The set of tags with which to save `tf.MetaGraphDef`. If
        `None`, a default set will be selected to matched the passed mode.
      check_variables: bool, whether to check the checkpoint has all variables.
      strip_default_attrs: bool, whether to strip default attributes. This
        may only be True when called from the deprecated V1
        Estimator.export_savedmodel.

    Raises:
      ValueError: if `save_variables` is `True` and `check_variable` is `False`.
    """
    if export_tags is None:
        export_tags = export_lib.EXPORT_TAG_MAP[mode]
    input_receiver_fn = input_receiver_fn_map[mode]

    with ops.Graph().as_default() as g:
        _create_and_assert_global_step(g)
        random_seed.set_random_seed(tf_random_seed)

        input_receiver = input_receiver_fn()

        # Call the model_fn and collect the export_outputs.
        print("_call_model_fn\n\n")
        estimator_spec = _call_model_fn(
            features=input_receiver.features,
            labels=getattr(input_receiver, "labels", None),
            mode=mode,
            config=config,
            _model_fn=classifier.model_fn,
            params=classifier.params,
        )
        origin_predictions = estimator_spec.predictions
        origin_input_tensors = input_receiver.receiver_tensors
        signature_def_map_copy = {}
        for ky, v in export_dict.items():
            outputs_keys = v["outputs"]
            clip_predictions = {x: origin_predictions[x] for x in outputs_keys}
            export_outputs = export_lib.export_outputs_for_mode(
                mode=estimator_spec.mode,
                # serving_export_outputs=estimator_spec.export_outputs,
                # predictions=estimator_spec.predictions,
                predictions=clip_predictions,
                loss=estimator_spec.loss,
                metrics=estimator_spec.eval_metric_ops,
            )

            inputs_keys = v["inputs"]
            input_tensors = {x: origin_input_tensors[x] for x in inputs_keys}
            signature_def_map = export_lib.build_all_signature_defs(
                # input_receiver.receiver_tensors,
                input_tensors,
                export_outputs,
                getattr(input_receiver, "receiver_tensors_alternatives", None),
                serving_only=(mode == ModeKeys.PREDICT),
            )
            signature_def_map_copy[ky] = signature_def_map["serving_default"]

        # if outputs_keys:
        #     clip_predictions = {
        #         x: origin_predictions[x] for x in outputs_keys
        #     }
        # else:
        #     clip_predictions = origin_predictions
        # export_outputs = export_lib.export_outputs_for_mode(
        #     mode=estimator_spec.mode,
        #     # serving_export_outputs=estimator_spec.export_outputs,
        #     # predictions=estimator_spec.predictions,
        #     predictions=clip_predictions,
        #     loss=estimator_spec.loss,
        #     metrics=estimator_spec.eval_metric_ops)
        # print(export_outputs)

        # origin_input_tensors = input_receiver.receiver_tensors
        #
        # if inputs_keys:
        #     input_tensors = {
        #         x: origin_input_tensors[x] for x in inputs_keys
        #     }
        # else:
        #     input_tensors = origin_input_tensors
        # signature_def_map = export_lib.build_all_signature_defs(
        #     # input_receiver.receiver_tensors,
        #     input_tensors,
        #     export_outputs,
        #     getattr(input_receiver, 'receiver_tensors_alternatives', None),
        #     serving_only=(mode == ModeKeys.PREDICT))
        # print(input_receiver.receiver_tensors, )
        # print(signature_def_map)

        # signature_def_map_copy = {
        #     'user_embedding': signature_def_map['serving_default'],
        #     'item_embedding': signature_def_map['serving_default'],
        # }
        # signature_def_map['']

        print(signature_def_map_copy)
        with tf_session.Session(config=session_config) as session:

            if estimator_spec.scaffold.local_init_op is not None:
                local_init_op = estimator_spec.scaffold.local_init_op
            else:
                local_init_op = monitored_session.Scaffold.default_local_init_op()

            # This saver will be used both for restoring variables now,
            # and in saving out the metagraph below. This ensures that any
            # Custom Savers stored with the Scaffold are passed through to the
            # SavedModel for restore later.

            graph_saver = estimator_spec.scaffold.saver or saver.Saver(sharded=True)

            if save_variables and not check_variables:
                raise ValueError(
                    "If `save_variables` is `True, `check_variables`"
                    "must not be `False`."
                )
            if check_variables:
                try:
                    print("checkpoint_path", checkpoint_path)
                    graph_saver.restore(session, checkpoint_path)
                except errors.NotFoundError as e:
                    msg = (
                        "Could not load all requested variables from checkpoint. "
                        "Please make sure your model_fn does not expect variables "
                        "that were not saved in the checkpoint.\n\n"
                        "Encountered error with mode `{}` while restoring "
                        "checkpoint from: `{}`. Full Traceback:\n\n{}"
                    ).format(mode, checkpoint_path, e)
                    raise ValueError(msg)

            # We add the train op explicitly for now, so that we don't have to
            # change the Builder public interface. Note that this is a no-op
            # for prediction, where train_op is None.
            builder._add_train_op(
                estimator_spec.train_op
            )  # pylint: disable=protected-access

            meta_graph_kwargs = dict(
                tags=export_tags,
                # signature_def_map=signature_def_map,
                signature_def_map=signature_def_map_copy,
                assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
                main_op=local_init_op,
                saver=graph_saver,
                strip_default_attrs=strip_default_attrs,
            )

            if save_variables:
                builder.add_meta_graph_and_variables(session, **meta_graph_kwargs)
            else:
                builder.add_meta_graph(**meta_graph_kwargs)


def export_all_saved_models(
    classifier,
    model_dir,
    export_dir_base,
    input_receiver_fn_map,
    assets_extra=None,
    as_text=False,
    checkpoint_path=None,
    strip_default_attrs=True,
    export_dict=None
    # inputs_keys=None,
    # outputs_keys=None
):
    """Exports multiple modes in the model function to a SavedModel."""
    # TODO(b/65561022): Consider allowing multiple input_receiver_fns per mode.
    with context.graph_mode():
        if not checkpoint_path:
            # Locate the latest checkpoint
            checkpoint_path = checkpoint_management.latest_checkpoint(model_dir)
        # if not checkpoint_path:
        #     if self._warm_start_settings:
        #         checkpoint_path = self._warm_start_settings.ckpt_to_initialize_from
        #         if gfile.IsDirectory(checkpoint_path):
        #             checkpoint_path = checkpoint_management.latest_checkpoint(
        #                 checkpoint_path)
        #     else:
        #         raise ValueError("Couldn't find trained model at {}.".format(
        #             self._model_dir))

        export_dir = export_lib.get_timestamped_export_dir(export_dir_base)
        temp_export_dir = export_lib.get_temp_export_dir(export_dir)

        builder = saved_model_builder.SavedModelBuilder(temp_export_dir)

        save_variables = True
        # Note that the order in which we run here matters, as the first
        # mode we pass through will be used to save the variables. We run TRAIN
        # first, as that is also the mode used for checkpoints, and therefore
        # we are not likely to have vars in PREDICT that are not in the checkpoint
        # created by TRAIN.
        # if input_receiver_fn_map.get(ModeKeys.TRAIN):
        #     _add_meta_graph_for_mode(
        #         builder, input_receiver_fn_map, checkpoint_path,
        #         save_variables, mode=ModeKeys.TRAIN,
        #         strip_default_attrs=strip_default_attrs,
        #         classifier=classifier)
        #     save_variables = False
        # if input_receiver_fn_map.get(ModeKeys.EVAL):
        #     _add_meta_graph_for_mode(
        #         builder, input_receiver_fn_map, checkpoint_path,
        #         save_variables, mode=ModeKeys.EVAL,
        #         strip_default_attrs=strip_default_attrs,
        #         classifier=classifier)
        #     save_variables = False
        if input_receiver_fn_map.get(ModeKeys.PREDICT):
            _add_meta_graph_for_mode(
                builder,
                input_receiver_fn_map,
                checkpoint_path,
                save_variables,
                mode=ModeKeys.PREDICT,
                strip_default_attrs=strip_default_attrs,
                classifier=classifier,
                export_dict=export_dict
                # inputs_keys=inputs_keys,
                # outputs_keys=outputs_keys
            )
            save_variables = False

        if save_variables:
            raise ValueError(
                "No valid modes for exporting found. Got {}.".format(
                    input_receiver_fn_map.keys()
                )
            )

        builder.save(as_text)

        # Add the extra assets
        if assets_extra:
            assets_extra_path = os.path.join(
                compat.as_bytes(temp_export_dir), compat.as_bytes("assets.extra")
            )
            for dest_relative, source in assets_extra.items():
                dest_absolute = os.path.join(
                    compat.as_bytes(assets_extra_path), compat.as_bytes(dest_relative)
                )
                dest_path = os.path.dirname(dest_absolute)
                gfile.MakeDirs(dest_path)
                gfile.Copy(source, dest_absolute)

        gfile.Rename(temp_export_dir, export_dir)
        return export_dir
