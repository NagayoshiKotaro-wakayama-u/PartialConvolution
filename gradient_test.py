import pdb
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

u_gt = 0.0
alpha = 10

sess = tf.Session()

x = tf.Variable([1.1,1.5,2.0,0.5,1.1])
x_ind = tf.Variable([0.0,1.0,2.0,3.0,4.0])
thre = 1.0

#x_sig = 1/(1+tf.exp(-alpha*(x-thre)))
x_ind_thre = x*x_ind
u_pred = tf.reduce_mean(x_ind_thre)

output = tf.square(u_gt - u_pred)

grad_x = tf.gradients(output, x)[0]
grad_x_ind = tf.gradients(output, x_ind)[0]

sess.run(tf.global_variables_initializer())
print(sess.run([x, x_ind_thre, u_pred, output, grad_x, grad_x_ind])) 