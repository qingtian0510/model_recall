import tensorflow as tf
import time

from dssm_v2.utils.train_utils import LOG


class QpsHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        self.start = time.time()
        self.batch = 0
        self.starttime = 0

    def before_run(self, run_context):
        self.starttime = time.time()
        return run_context.original_args

    def after_run(self, run_context, run_values):
        self.batch += 1
        dur = time.time() - self.start
        batch_dur = time.time() - self.starttime
        if self.batch == 1:
            self.start = time.time()

        gstep = run_values.results["gstep"]
        auc = run_values.results["auc"]
        if self.batch % 100 == 0:
            LOG(
                "step %d, gstep %d, gqps %.2f, lqps %.2f, loss %.4f, auc: %.6f",
                (
                    self.batch,
                    gstep,
                    gstep / dur,
                    self.batch / dur,
                    run_values.results["loss"],
                    auc,
                ),
            )
        # if self.batch % 1000 == 0:
        #  tf.logging.info("step %d, gstep %d, gqps %.2f, lqps %.2f, loss %.2f, batch_dur %.4fs" % (
        #    self.batch, gstep, gstep / dur,
        #    self.batch / dur, run_values.results["loss"], batch_dur))


class BaseSyncHook(tf.train.SessionRunHook):
    def get_sync_op(self, worker_count, worker_index, name):
        with tf.variable_scope(name):
            sync_data = tf.get_variable(
                "sync_data",
                [worker_count],
                initializer=tf.constant_initializer(0),
                trainable=False,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
            )
            signal = tf.scatter_add(sync_data, worker_index, 1)
            count = tf.reduce_sum(sync_data)
        return signal, count


class InitSyncHook(BaseSyncHook):
    def __init__(self, worker_count, worker_index):
        self.worker_count = worker_count
        self.ok_signal, self.ok_count = self.get_sync_op(
            worker_count, worker_index, "init_sync_hook"
        )

    def after_create_session(self, session, coord):
        session.run(self.ok_signal)
        while True:
            count = session.run(self.ok_count)
            if count == self.worker_count:
                break
            time.sleep(1.0)
        # print "ok, worker_count: %d, expect %d" % (count, self.worker_count)


class FinishSyncHook(BaseSyncHook):
    def __init__(self, worker_count, worker_index, done_ops):
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.done_ops = done_ops
        self.finish_signal, self.finish_count = self.get_sync_op(
            worker_count, worker_index, "finish_sync_hook"
        )

    def end(self, session):
        session.run(self.finish_signal)
        while True:
            count = session.run(self.finish_count)
            if count == self.worker_count:
                break
            time.sleep(0.3)
            tf.logging.info(
                "waiting, worker_finish: %d/%d" % (count, self.worker_count)
            )
        tf.logging.info("ok, all_worker_finish")
        if self.worker_index == 0:
            session.run(self.done_ops)
            tf.logging.info("sent exit signal to ps")
        time.sleep(20)
        tf.logging.info("waited additional time for graceful exit")
