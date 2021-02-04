import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()

x = tf.Variable([0.0,0.5,1.5,1.1,2.0,0.3,0.1])
thre = 1.0

xthre = tf.where(x>thre, x, tf.zeros_like(x))
output_where = tf.square(xthre)
gradient_where = tf.gradients(output_where, x)[0]

x_mask = tf.greater(x,1.0)
output_mask = tf.square(tf.boolean_mask(x,x_mask))
gradient_mask = tf.gradients(output_mask,x)[0]


sess.run(tf.global_variables_initializer())
print(sess.run([x, output_where, gradient_where])) 
print(sess.run([x, output_mask, gradient_mask])) 