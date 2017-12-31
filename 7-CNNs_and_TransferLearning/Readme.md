Goal is to:

Train an MNIST CNN classifier on just the digits: 1, 4, 5 and 9

Architecture: <br>
"conv1": conv2d 3x3x4, stride=1, ReLU, padding = "SAME"   <br>
"conv2": conv2d 3x3x8, stride=2, ReLU, padding = "SAME" <br>
"pool": pool 2x2 <br>
"fc1": fc 16  <br>
"fc2": fc 10   <br>
"softmax": xentropy loss, fc-logits = 4 (we have 4 classes) <br>

Optimizer used: ADAM

5 epochs, 10 batch size

the trained model’s weights are used on the lower 4 layers to train a classifier for the rest of MNIST (excluding 1,4,5 and 9)

New layers are created for the top (5 and 6) <br>
Try to run as few epochs as possible to get a good classification (> 99% on test) <br>
Try a session with freezing the lower layers weights, and also a session of just fine-tuning the weights. <br>
Use (for speed) a constraint on the optimizer for freezing: <br>
For freezing <br>

`train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc2_023678|softmax_023678")`
`training_op = optimizer.minimize(loss, var_list=train_vars)`
 

Test loss curve on transferred MNIST-023678: <br>
       - with fine-tuning everything <br>
       - with frozen layers up to fc2 (and not including) 
