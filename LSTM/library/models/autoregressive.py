"""dumb model, predict no change ."""
import tensorflow as tf


class FeedBack(tf.keras.Model):
    def __init__(self, num_features, num_labels, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.lstm_cell = tf.keras.layers.LSTMCell(num_features)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_labels)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        internal_pred, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(internal_pred)
        return prediction, internal_pred, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # arnaud: tf graph requires to store all elements to apply grad
        # internal_preds = []
        # Initialize the LSTM state.
        prediction, internal_pred, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)
        # internal_preds.append(internal_pred)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input and execute one lstm step.
            internal_pred, state = self.lstm_cell(internal_pred, states=state,
                                                  training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(internal_pred)
            # Add the prediction to the output.
            # internal_preds.append(internal_pred)
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions
