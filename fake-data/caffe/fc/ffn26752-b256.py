import mxnet as mx


data = mx.symbol.Variable(name = data)
H1 = mx.symbol.FullyConnected(data = data, num_hidden = 2048)
H1_A = mx.symbol.Activation(data = H1, act_type="sigmoid")
H2 = mx.symbol.FullyConnected(data = H1_A, num_hidden = 2048)
H2_A = mx.symbol.Activation(data = H2, act_type="sigmoid")
H3 = mx.symbol.FullyConnected(data = H2_A, num_hidden = 2048)
H3_A = mx.symbol.Activation(data = H3, act_type="sigmoid")
L = mx.symbol.FullyConnected(data = H3_A, num_hidden = 26752)
loss = mx.symbol.SoftmaxOutput(data = label, name = 'softmax')
