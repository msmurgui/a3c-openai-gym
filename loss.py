import tensorflow as tf

def computeLoss(rewards, cumulativeReward, values, logProbabilities, entropies, params):
    policyLoss = 0
    valueLoss = 0
    gae = tf.zeros((1, 1))
    for i in reversed(range(len(rewards))):

        cumulativeReward = params.gamma * cumulativeReward + rewards[i]

        advantage = cumulativeReward - values[i]

        valueLoss = valueLoss + 0.5 * (advantage * advantage)

        TD = rewards[i] + params.gamma * values[i+1] - values[i]

        gae = gae * params.gamma * params.tau + TD

        policyLoss = policyLoss - logProbabilities[i] * gae - 0.01 * entropies[i]

    loss = 0.5 * valueLoss + policyLoss
    return tf.abs(loss[0][0])

def computeActorLoss(self, actions, logits, advantages):
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    entropy_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True)
    actions = tf.cast(actions, tf.int32)
    policy_loss = ce_loss(
        actions, logits, sample_weight=tf.stop_gradient(advantages))
    entropy = entropy_loss(logits, logits)
    return policy_loss - self.entropy_beta * entropy

#v_pred => critic output, td_targets => cumulativeReward (after computeLoss())
def computeCriticLoss(self, v_pred, td_targets):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(td_targets, v_pred)
