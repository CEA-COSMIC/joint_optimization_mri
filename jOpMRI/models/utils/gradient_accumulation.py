import tensorflow as tf
import logging

log = logging.getLogger(__name__)

"""Customizing the gradient accumulation optimizer for Joint Model"""


@tf.keras.utils.register_keras_serializable()  # adding this avoids needing to use custom_objects when loading model
class GAModelWrapper(tf.keras.Model):
    def __init__(self, accum_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FIXME : This wont work for graph mode
        if isinstance(accum_steps, int):
            self.accum_steps_traj, self.accum_steps_recon = accum_steps, accum_steps
        else:
            self.accum_steps_traj, self.accum_steps_recon = accum_steps
        self.accum_step_counter_traj, self.accum_step_counter_recon = 0, 0
        self.gradient_accumulation_traj = []
        self.gradient_accumulation_recon = []

    def train_step(self, data):
        self.accum_step_counter_traj += 1
        self.accum_step_counter_recon += 1
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # NOTE that x and y are lists of inputs and outputs,
        # hence this wrapper supports multi-input-output models
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # forward pass

            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
            # We dont change loss as we want to scale the gradients rather!
            #loss = loss / tf.cast(self.accum_steps, tf.float32)  # MEAN reduction here IMPORTANT! Don't use SUM!
            
        # Calculate batch gradients -> these are scaled gradients if mixed precision is enabled
        gradients = tape.gradient(loss, [self.base_traj.trainable_variables, self.recon_net.trainable_variables])
        
        
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation_traj)):
            if gradients[0][i] is not None:
                self.gradient_accumulation_traj[i] += gradients[0][i] / tf.cast(self.accum_steps_traj, tf.float32)
        
        for i in range(len(self.gradient_accumulation_recon)):
            if gradients[1][i] is not None:
                self.gradient_accumulation_recon[i] += gradients[1][i] / tf.cast(self.accum_steps_recon, tf.float32)
                
        # If accum_step_counter reach the accum_steps then we apply accumulated gradients to update the variables
        # otherwise do nothing
        tf.cond(
            tf.equal(self.accum_step_counter_traj, self.accum_steps_traj),
            true_fn=self.apply_accu_gradients_traj,
            false_fn=lambda: None
        )
        tf.cond(
            tf.equal(self.accum_step_counter_recon, self.accum_steps_recon),
            true_fn=self.apply_accu_gradients_recon,
            false_fn=lambda: None
        )
        # update metrics
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients_traj(self):
        # apply accumulated gradients for traj
        log.info("Applying accumulated gradients for Traj, trainable: {}".format(self.base_traj.trainable))
        self.optimizer.apply_gradients(zip(self.gradient_accumulation_traj, self.base_traj.trainable_variables))

        # reset
        self.accum_step_counter_traj = 0
        for i in range(len(self.gradient_accumulation_traj)):
            self.gradient_accumulation_traj[i] = tf.zeros_like(self.gradient_accumulation_traj[i])
    
    def apply_accu_gradients_recon(self):
        # apply accumulated gradients for recon, a bit repeated codes, but mehh
        log.info("Applying accumulated gradients for Recon, trainable: {}".format(self.recon_net.trainable))
        self.optimizer.apply_gradients(zip(self.gradient_accumulation_recon, self.recon_net.trainable_variables))

        # reset
        self.accum_step_counter_recon = 0
        for i in range(len(self.gradient_accumulation_recon)):
            self.gradient_accumulation_recon[i] = tf.zeros_like(self.gradient_accumulation_recon[i])
        
    def reinit_grad_accum(self):
        log.info("Reinit accumulated gradients")
        del self.gradient_accumulation_recon
        del self.gradient_accumulation_traj
        self.gradient_accumulation_traj = [ 
            tf.zeros_like(var)
            for i, var in enumerate(self.base_traj.trainable_variables)
        ]
        self.gradient_accumulation_recon = [ 
            tf.zeros_like(var)
            for i, var in enumerate(self.recon_net.trainable_variables)
        ]
        self.accum_step_counter_traj = 0
        self.accum_step_counter_recon = 0
        