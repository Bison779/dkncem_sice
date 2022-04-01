# Functions for freezing a Keras graph to use with tensorflow
# sources
# https://stackoverflow.com/questions/60974077/how-to-save-keras-model-as-frozen-graph
# https://stackoverflow.com/questions/68656751/how-to-speed-up-the-keras-model-predict
# https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
#
# in: 
#     model - keras model
# 
# out:
#     xdot  - state change and derivatives
#
# Author = Brendan Michael (heavily based on above references)

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

# Load the frozen model from the default directory	 
def load_default_frozen_model(modelname):

	# Load frozen graph using TensorFlow 1.x functions
	with tf.io.gfile.GFile("../src/frozen_models/"+modelname+"/frozen_graph.pb", "rb") as f:
	    graph_def = tf.compat.v1.GraphDef()
	    loaded = graph_def.ParseFromString(f.read())

	# Wrap frozen graph to ConcreteFunctions
	frozen_func = wrap_frozen_graph(graph_def=graph_def,
		                        inputs=["x:0"],
		                        outputs=["Identity:0"],
		                        print_graph=True)
	return frozen_func

# Convert a keras model to a frozen graph (only needs to be run once per trained model)
def convert_keras_frozen(modelname):
	model = keras.models.load_model('../src/keras_models/'+modelname, compile=False)
	# Convert Keras model to ConcreteFunction
	full_model = tf.function(lambda x: model(x))
	full_model = full_model.get_concrete_function(
	    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

	# Get frozen ConcreteFunction
	frozen_func = convert_variables_to_constants_v2(full_model)
	frozen_func.graph.as_graph_def()

	# Save frozen graph from frozen ConcreteFunction to hard drive
	tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
			  logdir="../src/frozen_models/"+modelname,
			  name="frozen_graph.pb",
			  as_text=False)

# wrapper for converting
def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
	def _imports_graph_def():
		tf.compat.v1.import_graph_def(graph_def, name="")

	wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
	import_graph = wrapped_import.graph

	#print("-" * 50)
	#print("Frozen model layers: ")
	layers = [op.name for op in import_graph.get_operations()]
	#if print_graph == True:
#		for layer in layers:
#	    		print(layer)
#	print("-" * 50)

	return wrapped_import.prune(
		tf.nest.map_structure(import_graph.as_graph_element, inputs),
		tf.nest.map_structure(import_graph.as_graph_element, outputs))	  
