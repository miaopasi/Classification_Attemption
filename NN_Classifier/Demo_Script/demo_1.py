import pyfann.libfann as fann
import numpy
from matplotlib import pyplot, rc

# init pyplot
rc('text', usetex=True)
rc('font', family='serif')

# create an empty simple network
def create_net(layers, funcs):
	net = fann.neural_net()
	net.create_sparse_array(1, layers)
	net.set_activation_function_hidden(funcs[0])
	net.set_activation_function_output(funcs[1])

	return net

# convinience method for 1-dimensional arrays
# fann cannot handle those and leaves with a segfault :(
def check_matrix(matrix):
	if matrix.ndim == 1:
		new = numpy.empty((matrix.shape[0], 1))

		for i, x in enumerate(matrix):
			new[(i, 0)] = x

		return new

	return matrix

def load_data(filename, in_outs):
	a = numpy.loadtxt(filename)
	inp = numpy.compress(numpy.ones(in_outs[0]), a, axis=1)
	inp = check_matrix(inp)
	out = numpy.compress(numpy.concatenate([numpy.zeros(in_outs[0]), numpy.ones(in_outs[1])]), a, axis=1)
	out = check_matrix(out)

	data = fann.training_data()
	data.set_train_data(inp,out)

	return data

# load data and create training set
def load_data_prefix(prefix):
	inp = numpy.loadtxt(prefix + "_i.txt")
	inp = check_matrix(inp)
	out = numpy.loadtxt(prefix + "_o.txt")
	out = check_matrix(out)

	data = fann.training_data()
	data.set_train_data(inp,out)

	return data

# learning routine
def learn(data, layers, funcs):
	net = create_net((len(data.get_input()[0]), layers, len(data.get_output()[0])), funcs)
	net.train_on_data(data, 10000, 1000, 0.0001)

	return net

# run the net with some data
def run(net, data):
	out = numpy.zeros((data.length_train_data(), data.num_output_train_data()))
	for i, y in enumerate(data.get_input()):
		out[i] = net.run(y)

	return out

def save_plot(filename):
	pyplot.savefig(filename, format = "pdf")

def run_suite(filename, num_hidden, func, in_outs):
	data = load_data(filename, in_outs)
	net = learn(data, num_hidden, func)
	out = run(net, data)

	#pyplot.subplot(311)
	#pyplot.plot(data.get_output())
	#pyplot.xlabel("Output data")

	#pyplot.subplot(312)
	#pyplot.plot(out)
	#pyplot.xlabel("Network output on input data")

	error = out - data.get_output()
	pyplot.subplot(111)
	pyplot.plot(error)
	pyplot.xlabel("Error")
	pyplot.show()

