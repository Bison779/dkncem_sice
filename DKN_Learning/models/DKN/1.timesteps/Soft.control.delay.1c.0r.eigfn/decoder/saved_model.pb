??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8ƽ
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:P*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:P*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:PP*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:P*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:P*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
regularization_losses
		variables

trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
 
*
0
1
2
3
!4
"5
*
0
1
2
3
!4
"5
?
regularization_losses
		variables
+metrics

trainable_variables
,layer_regularization_losses
-layer_metrics

.layers
/non_trainable_variables
 
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
0metrics
trainable_variables
1layer_regularization_losses
2layer_metrics

3layers
4non_trainable_variables
 
 
 
?
regularization_losses
	variables
5metrics
trainable_variables
6layer_regularization_losses
7layer_metrics

8layers
9non_trainable_variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
:metrics
trainable_variables
;layer_regularization_losses
<layer_metrics

=layers
>non_trainable_variables
 
 
 
?
regularization_losses
	variables
?metrics
trainable_variables
@layer_regularization_losses
Alayer_metrics

Blayers
Cnon_trainable_variables
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
?
#regularization_losses
$	variables
Dmetrics
%trainable_variables
Elayer_regularization_losses
Flayer_metrics

Glayers
Hnon_trainable_variables
 
 
 
?
'regularization_losses
(	variables
Imetrics
)trainable_variables
Jlayer_regularization_losses
Klayer_metrics

Llayers
Mnon_trainable_variables
 
 
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_2Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_10081
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_10788
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_10816??
?
G
-__inference_dense_6_activity_regularizer_9428
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
G
-__inference_dense_5_activity_regularizer_9513
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_10508

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:?????????P2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
F__inference_dense_6_layer_call_and_return_all_conditional_losses_10583

inputs
dense_6_kernel
dense_6_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_kerneldense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_95842
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_6_activity_regularizer_96102
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????P2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*2
_input_shapes!
:?????????P::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
F__inference_dense_7_layer_call_and_return_all_conditional_losses_10663

inputs
dense_7_kernel
dense_7_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_kerneldense_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_96812
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_7_activity_regularizer_97072
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*2
_input_shapes!
:?????????P::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_9402
input_2;
7model_2_dense_5_tensordot_readvariableop_dense_5_kernel7
3model_2_dense_5_biasadd_readvariableop_dense_5_bias;
7model_2_dense_6_tensordot_readvariableop_dense_6_kernel7
3model_2_dense_6_biasadd_readvariableop_dense_6_bias;
7model_2_dense_7_tensordot_readvariableop_dense_7_kernel7
3model_2_dense_7_biasadd_readvariableop_dense_7_bias
identity??&model_2/dense_5/BiasAdd/ReadVariableOp?(model_2/dense_5/Tensordot/ReadVariableOp?&model_2/dense_6/BiasAdd/ReadVariableOp?(model_2/dense_6/Tensordot/ReadVariableOp?&model_2/dense_7/BiasAdd/ReadVariableOp?(model_2/dense_7/Tensordot/ReadVariableOp?
(model_2/dense_5/Tensordot/ReadVariableOpReadVariableOp7model_2_dense_5_tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype02*
(model_2/dense_5/Tensordot/ReadVariableOp?
model_2/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_2/dense_5/Tensordot/axes?
model_2/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_2/dense_5/Tensordot/freey
model_2/dense_5/Tensordot/ShapeShapeinput_2*
T0*
_output_shapes
:2!
model_2/dense_5/Tensordot/Shape?
'model_2/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_2/dense_5/Tensordot/GatherV2/axis?
"model_2/dense_5/Tensordot/GatherV2GatherV2(model_2/dense_5/Tensordot/Shape:output:0'model_2/dense_5/Tensordot/free:output:00model_2/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_2/dense_5/Tensordot/GatherV2?
)model_2/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_2/dense_5/Tensordot/GatherV2_1/axis?
$model_2/dense_5/Tensordot/GatherV2_1GatherV2(model_2/dense_5/Tensordot/Shape:output:0'model_2/dense_5/Tensordot/axes:output:02model_2/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_2/dense_5/Tensordot/GatherV2_1?
model_2/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_2/dense_5/Tensordot/Const?
model_2/dense_5/Tensordot/ProdProd+model_2/dense_5/Tensordot/GatherV2:output:0(model_2/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_2/dense_5/Tensordot/Prod?
!model_2/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_2/dense_5/Tensordot/Const_1?
 model_2/dense_5/Tensordot/Prod_1Prod-model_2/dense_5/Tensordot/GatherV2_1:output:0*model_2/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_2/dense_5/Tensordot/Prod_1?
%model_2/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_2/dense_5/Tensordot/concat/axis?
 model_2/dense_5/Tensordot/concatConcatV2'model_2/dense_5/Tensordot/free:output:0'model_2/dense_5/Tensordot/axes:output:0.model_2/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_2/dense_5/Tensordot/concat?
model_2/dense_5/Tensordot/stackPack'model_2/dense_5/Tensordot/Prod:output:0)model_2/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_2/dense_5/Tensordot/stack?
#model_2/dense_5/Tensordot/transpose	Transposeinput_2)model_2/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2%
#model_2/dense_5/Tensordot/transpose?
!model_2/dense_5/Tensordot/ReshapeReshape'model_2/dense_5/Tensordot/transpose:y:0(model_2/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_2/dense_5/Tensordot/Reshape?
 model_2/dense_5/Tensordot/MatMulMatMul*model_2/dense_5/Tensordot/Reshape:output:00model_2/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2"
 model_2/dense_5/Tensordot/MatMul?
!model_2/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2#
!model_2/dense_5/Tensordot/Const_2?
'model_2/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_2/dense_5/Tensordot/concat_1/axis?
"model_2/dense_5/Tensordot/concat_1ConcatV2+model_2/dense_5/Tensordot/GatherV2:output:0*model_2/dense_5/Tensordot/Const_2:output:00model_2/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_2/dense_5/Tensordot/concat_1?
model_2/dense_5/TensordotReshape*model_2/dense_5/Tensordot/MatMul:product:0+model_2/dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
model_2/dense_5/Tensordot?
&model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp3model_2_dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype02(
&model_2/dense_5/BiasAdd/ReadVariableOp?
model_2/dense_5/BiasAddBiasAdd"model_2/dense_5/Tensordot:output:0.model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
model_2/dense_5/BiasAdd?
*model_2/dense_5/ActivityRegularizer/SquareSquare model_2/dense_5/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2,
*model_2/dense_5/ActivityRegularizer/Square?
)model_2/dense_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_2/dense_5/ActivityRegularizer/Const?
'model_2/dense_5/ActivityRegularizer/SumSum.model_2/dense_5/ActivityRegularizer/Square:y:02model_2/dense_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_2/dense_5/ActivityRegularizer/Sum?
)model_2/dense_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_2/dense_5/ActivityRegularizer/mul/x?
'model_2/dense_5/ActivityRegularizer/mulMul2model_2/dense_5/ActivityRegularizer/mul/x:output:00model_2/dense_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_2/dense_5/ActivityRegularizer/mul?
)model_2/dense_5/ActivityRegularizer/ShapeShape model_2/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_2/dense_5/ActivityRegularizer/Shape?
7model_2/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_2/dense_5/ActivityRegularizer/strided_slice/stack?
9model_2/dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_2/dense_5/ActivityRegularizer/strided_slice/stack_1?
9model_2/dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_2/dense_5/ActivityRegularizer/strided_slice/stack_2?
1model_2/dense_5/ActivityRegularizer/strided_sliceStridedSlice2model_2/dense_5/ActivityRegularizer/Shape:output:0@model_2/dense_5/ActivityRegularizer/strided_slice/stack:output:0Bmodel_2/dense_5/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_2/dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_2/dense_5/ActivityRegularizer/strided_slice?
(model_2/dense_5/ActivityRegularizer/CastCast:model_2/dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_2/dense_5/ActivityRegularizer/Cast?
+model_2/dense_5/ActivityRegularizer/truedivRealDiv+model_2/dense_5/ActivityRegularizer/mul:z:0,model_2/dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_2/dense_5/ActivityRegularizer/truediv?
model_2/activation_3/ReluRelu model_2/dense_5/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
model_2/activation_3/Relu?
(model_2/dense_6/Tensordot/ReadVariableOpReadVariableOp7model_2_dense_6_tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype02*
(model_2/dense_6/Tensordot/ReadVariableOp?
model_2/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_2/dense_6/Tensordot/axes?
model_2/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_2/dense_6/Tensordot/free?
model_2/dense_6/Tensordot/ShapeShape'model_2/activation_3/Relu:activations:0*
T0*
_output_shapes
:2!
model_2/dense_6/Tensordot/Shape?
'model_2/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_2/dense_6/Tensordot/GatherV2/axis?
"model_2/dense_6/Tensordot/GatherV2GatherV2(model_2/dense_6/Tensordot/Shape:output:0'model_2/dense_6/Tensordot/free:output:00model_2/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_2/dense_6/Tensordot/GatherV2?
)model_2/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_2/dense_6/Tensordot/GatherV2_1/axis?
$model_2/dense_6/Tensordot/GatherV2_1GatherV2(model_2/dense_6/Tensordot/Shape:output:0'model_2/dense_6/Tensordot/axes:output:02model_2/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_2/dense_6/Tensordot/GatherV2_1?
model_2/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_2/dense_6/Tensordot/Const?
model_2/dense_6/Tensordot/ProdProd+model_2/dense_6/Tensordot/GatherV2:output:0(model_2/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_2/dense_6/Tensordot/Prod?
!model_2/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_2/dense_6/Tensordot/Const_1?
 model_2/dense_6/Tensordot/Prod_1Prod-model_2/dense_6/Tensordot/GatherV2_1:output:0*model_2/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_2/dense_6/Tensordot/Prod_1?
%model_2/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_2/dense_6/Tensordot/concat/axis?
 model_2/dense_6/Tensordot/concatConcatV2'model_2/dense_6/Tensordot/free:output:0'model_2/dense_6/Tensordot/axes:output:0.model_2/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_2/dense_6/Tensordot/concat?
model_2/dense_6/Tensordot/stackPack'model_2/dense_6/Tensordot/Prod:output:0)model_2/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_2/dense_6/Tensordot/stack?
#model_2/dense_6/Tensordot/transpose	Transpose'model_2/activation_3/Relu:activations:0)model_2/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2%
#model_2/dense_6/Tensordot/transpose?
!model_2/dense_6/Tensordot/ReshapeReshape'model_2/dense_6/Tensordot/transpose:y:0(model_2/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_2/dense_6/Tensordot/Reshape?
 model_2/dense_6/Tensordot/MatMulMatMul*model_2/dense_6/Tensordot/Reshape:output:00model_2/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2"
 model_2/dense_6/Tensordot/MatMul?
!model_2/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2#
!model_2/dense_6/Tensordot/Const_2?
'model_2/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_2/dense_6/Tensordot/concat_1/axis?
"model_2/dense_6/Tensordot/concat_1ConcatV2+model_2/dense_6/Tensordot/GatherV2:output:0*model_2/dense_6/Tensordot/Const_2:output:00model_2/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_2/dense_6/Tensordot/concat_1?
model_2/dense_6/TensordotReshape*model_2/dense_6/Tensordot/MatMul:product:0+model_2/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
model_2/dense_6/Tensordot?
&model_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp3model_2_dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype02(
&model_2/dense_6/BiasAdd/ReadVariableOp?
model_2/dense_6/BiasAddBiasAdd"model_2/dense_6/Tensordot:output:0.model_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
model_2/dense_6/BiasAdd?
*model_2/dense_6/ActivityRegularizer/SquareSquare model_2/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2,
*model_2/dense_6/ActivityRegularizer/Square?
)model_2/dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_2/dense_6/ActivityRegularizer/Const?
'model_2/dense_6/ActivityRegularizer/SumSum.model_2/dense_6/ActivityRegularizer/Square:y:02model_2/dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_2/dense_6/ActivityRegularizer/Sum?
)model_2/dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_2/dense_6/ActivityRegularizer/mul/x?
'model_2/dense_6/ActivityRegularizer/mulMul2model_2/dense_6/ActivityRegularizer/mul/x:output:00model_2/dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_2/dense_6/ActivityRegularizer/mul?
)model_2/dense_6/ActivityRegularizer/ShapeShape model_2/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_2/dense_6/ActivityRegularizer/Shape?
7model_2/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_2/dense_6/ActivityRegularizer/strided_slice/stack?
9model_2/dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_2/dense_6/ActivityRegularizer/strided_slice/stack_1?
9model_2/dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_2/dense_6/ActivityRegularizer/strided_slice/stack_2?
1model_2/dense_6/ActivityRegularizer/strided_sliceStridedSlice2model_2/dense_6/ActivityRegularizer/Shape:output:0@model_2/dense_6/ActivityRegularizer/strided_slice/stack:output:0Bmodel_2/dense_6/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_2/dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_2/dense_6/ActivityRegularizer/strided_slice?
(model_2/dense_6/ActivityRegularizer/CastCast:model_2/dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_2/dense_6/ActivityRegularizer/Cast?
+model_2/dense_6/ActivityRegularizer/truedivRealDiv+model_2/dense_6/ActivityRegularizer/mul:z:0,model_2/dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_2/dense_6/ActivityRegularizer/truediv?
model_2/activation_4/ReluRelu model_2/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
model_2/activation_4/Relu?
(model_2/dense_7/Tensordot/ReadVariableOpReadVariableOp7model_2_dense_7_tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype02*
(model_2/dense_7/Tensordot/ReadVariableOp?
model_2/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_2/dense_7/Tensordot/axes?
model_2/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_2/dense_7/Tensordot/free?
model_2/dense_7/Tensordot/ShapeShape'model_2/activation_4/Relu:activations:0*
T0*
_output_shapes
:2!
model_2/dense_7/Tensordot/Shape?
'model_2/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_2/dense_7/Tensordot/GatherV2/axis?
"model_2/dense_7/Tensordot/GatherV2GatherV2(model_2/dense_7/Tensordot/Shape:output:0'model_2/dense_7/Tensordot/free:output:00model_2/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_2/dense_7/Tensordot/GatherV2?
)model_2/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_2/dense_7/Tensordot/GatherV2_1/axis?
$model_2/dense_7/Tensordot/GatherV2_1GatherV2(model_2/dense_7/Tensordot/Shape:output:0'model_2/dense_7/Tensordot/axes:output:02model_2/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_2/dense_7/Tensordot/GatherV2_1?
model_2/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_2/dense_7/Tensordot/Const?
model_2/dense_7/Tensordot/ProdProd+model_2/dense_7/Tensordot/GatherV2:output:0(model_2/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_2/dense_7/Tensordot/Prod?
!model_2/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_2/dense_7/Tensordot/Const_1?
 model_2/dense_7/Tensordot/Prod_1Prod-model_2/dense_7/Tensordot/GatherV2_1:output:0*model_2/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_2/dense_7/Tensordot/Prod_1?
%model_2/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_2/dense_7/Tensordot/concat/axis?
 model_2/dense_7/Tensordot/concatConcatV2'model_2/dense_7/Tensordot/free:output:0'model_2/dense_7/Tensordot/axes:output:0.model_2/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_2/dense_7/Tensordot/concat?
model_2/dense_7/Tensordot/stackPack'model_2/dense_7/Tensordot/Prod:output:0)model_2/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_2/dense_7/Tensordot/stack?
#model_2/dense_7/Tensordot/transpose	Transpose'model_2/activation_4/Relu:activations:0)model_2/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2%
#model_2/dense_7/Tensordot/transpose?
!model_2/dense_7/Tensordot/ReshapeReshape'model_2/dense_7/Tensordot/transpose:y:0(model_2/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_2/dense_7/Tensordot/Reshape?
 model_2/dense_7/Tensordot/MatMulMatMul*model_2/dense_7/Tensordot/Reshape:output:00model_2/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 model_2/dense_7/Tensordot/MatMul?
!model_2/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model_2/dense_7/Tensordot/Const_2?
'model_2/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_2/dense_7/Tensordot/concat_1/axis?
"model_2/dense_7/Tensordot/concat_1ConcatV2+model_2/dense_7/Tensordot/GatherV2:output:0*model_2/dense_7/Tensordot/Const_2:output:00model_2/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_2/dense_7/Tensordot/concat_1?
model_2/dense_7/TensordotReshape*model_2/dense_7/Tensordot/MatMul:product:0+model_2/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
model_2/dense_7/Tensordot?
&model_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp3model_2_dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype02(
&model_2/dense_7/BiasAdd/ReadVariableOp?
model_2/dense_7/BiasAddBiasAdd"model_2/dense_7/Tensordot:output:0.model_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model_2/dense_7/BiasAdd?
*model_2/dense_7/ActivityRegularizer/SquareSquare model_2/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2,
*model_2/dense_7/ActivityRegularizer/Square?
)model_2/dense_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_2/dense_7/ActivityRegularizer/Const?
'model_2/dense_7/ActivityRegularizer/SumSum.model_2/dense_7/ActivityRegularizer/Square:y:02model_2/dense_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_2/dense_7/ActivityRegularizer/Sum?
)model_2/dense_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_2/dense_7/ActivityRegularizer/mul/x?
'model_2/dense_7/ActivityRegularizer/mulMul2model_2/dense_7/ActivityRegularizer/mul/x:output:00model_2/dense_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_2/dense_7/ActivityRegularizer/mul?
)model_2/dense_7/ActivityRegularizer/ShapeShape model_2/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_2/dense_7/ActivityRegularizer/Shape?
7model_2/dense_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_2/dense_7/ActivityRegularizer/strided_slice/stack?
9model_2/dense_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_2/dense_7/ActivityRegularizer/strided_slice/stack_1?
9model_2/dense_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_2/dense_7/ActivityRegularizer/strided_slice/stack_2?
1model_2/dense_7/ActivityRegularizer/strided_sliceStridedSlice2model_2/dense_7/ActivityRegularizer/Shape:output:0@model_2/dense_7/ActivityRegularizer/strided_slice/stack:output:0Bmodel_2/dense_7/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_2/dense_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_2/dense_7/ActivityRegularizer/strided_slice?
(model_2/dense_7/ActivityRegularizer/CastCast:model_2/dense_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_2/dense_7/ActivityRegularizer/Cast?
+model_2/dense_7/ActivityRegularizer/truedivRealDiv+model_2/dense_7/ActivityRegularizer/mul:z:0,model_2/dense_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_2/dense_7/ActivityRegularizer/truediv?
model_2/reshape_1/ShapeShape model_2/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:2
model_2/reshape_1/Shape?
%model_2/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_2/reshape_1/strided_slice/stack?
'model_2/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_2/reshape_1/strided_slice/stack_1?
'model_2/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_2/reshape_1/strided_slice/stack_2?
model_2/reshape_1/strided_sliceStridedSlice model_2/reshape_1/Shape:output:0.model_2/reshape_1/strided_slice/stack:output:00model_2/reshape_1/strided_slice/stack_1:output:00model_2/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_2/reshape_1/strided_slice?
!model_2/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/reshape_1/Reshape/shape/1?
!model_2/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/reshape_1/Reshape/shape/2?
model_2/reshape_1/Reshape/shapePack(model_2/reshape_1/strided_slice:output:0*model_2/reshape_1/Reshape/shape/1:output:0*model_2/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2!
model_2/reshape_1/Reshape/shape?
model_2/reshape_1/ReshapeReshape model_2/dense_7/BiasAdd:output:0(model_2/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
model_2/reshape_1/Reshape?
IdentityIdentity"model_2/reshape_1/Reshape:output:0'^model_2/dense_5/BiasAdd/ReadVariableOp)^model_2/dense_5/Tensordot/ReadVariableOp'^model_2/dense_6/BiasAdd/ReadVariableOp)^model_2/dense_6/Tensordot/ReadVariableOp'^model_2/dense_7/BiasAdd/ReadVariableOp)^model_2/dense_7/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2P
&model_2/dense_5/BiasAdd/ReadVariableOp&model_2/dense_5/BiasAdd/ReadVariableOp2T
(model_2/dense_5/Tensordot/ReadVariableOp(model_2/dense_5/Tensordot/ReadVariableOp2P
&model_2/dense_6/BiasAdd/ReadVariableOp&model_2/dense_6/BiasAdd/ReadVariableOp2T
(model_2/dense_6/Tensordot/ReadVariableOp(model_2/dense_6/Tensordot/ReadVariableOp2P
&model_2/dense_7/BiasAdd/ReadVariableOp&model_2/dense_7/BiasAdd/ReadVariableOp2T
(model_2/dense_7/Tensordot/ReadVariableOp(model_2/dense_7/Tensordot/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
!__inference__traced_restore_10816
file_prefix#
assignvariableop_dense_5_kernel#
assignvariableop_1_dense_5_bias%
!assignvariableop_2_dense_6_kernel#
assignvariableop_3_dense_6_bias%
!assignvariableop_4_dense_7_kernel#
assignvariableop_5_dense_7_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_10588

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:?????????P2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
E
)__inference_reshape_1_layer_call_fn_10681

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_97362
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_10714C
?dense_6_kernel_regularizer_square_readvariableop_dense_6_kernel
identity??0dense_6/kernel/Regularizer/Square/ReadVariableOp?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_6_kernel_regularizer_square_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity"dense_6/kernel/Regularizer/mul:z:01^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_4_10736C
?dense_7_kernel_regularizer_square_readvariableop_dense_7_kernel
identity??0dense_7/kernel/Regularizer/Square/ReadVariableOp?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_7_kernel_regularizer_square_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
IdentityIdentity"dense_7/kernel/Regularizer/mul:z:01^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp
?3
?
A__inference_dense_6_layer_call_and_return_conditional_losses_9584

inputs+
'tensordot_readvariableop_dense_6_kernel'
#biasadd_readvariableop_dense_6_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
'__inference_dense_5_layer_call_fn_10494

inputs
dense_5_kernel
dense_5_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_kerneldense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_94872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
-__inference_dense_7_activity_regularizer_9707
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
G
-__inference_dense_7_activity_regularizer_9441
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
?
'__inference_dense_7_layer_call_fn_10654

inputs
dense_7_kernel
dense_7_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_kerneldense_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_96812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
B__inference_model_2_layer_call_and_return_conditional_losses_10023

inputs
dense_5_dense_5_kernel
dense_5_dense_5_bias
dense_6_dense_6_kernel
dense_6_dense_6_bias
dense_7_dense_7_kernel
dense_7_dense_7_bias
identity??dense_5/StatefulPartitionedCall?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?dense_6/StatefulPartitionedCall?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?dense_7/StatefulPartitionedCall?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_94872!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_5_activity_regularizer_95132-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_95342
activation_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_95842!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_6_activity_regularizer_96102-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
activation_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_96312
activation_4/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_96812!
dense_7/StatefulPartitionedCall?
+dense_7/ActivityRegularizer/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_7_activity_regularizer_97072-
+dense_7/ActivityRegularizer/PartitionedCall?
!dense_7/ActivityRegularizer/ShapeShape(dense_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_7/ActivityRegularizer/Shape?
/dense_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_7/ActivityRegularizer/strided_slice/stack?
1dense_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_1?
1dense_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_2?
)dense_7/ActivityRegularizer/strided_sliceStridedSlice*dense_7/ActivityRegularizer/Shape:output:08dense_7/ActivityRegularizer/strided_slice/stack:output:0:dense_7/ActivityRegularizer/strided_slice/stack_1:output:0:dense_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_7/ActivityRegularizer/strided_slice?
 dense_7/ActivityRegularizer/CastCast2dense_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_7/ActivityRegularizer/Cast?
#dense_7/ActivityRegularizer/truedivRealDiv4dense_7/ActivityRegularizer/PartitionedCall:output:0$dense_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_7/ActivityRegularizer/truediv?
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_97362
reshape_1/PartitionedCall?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_5/StatefulPartitionedCall/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
-__inference_dense_6_activity_regularizer_9610
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
?
'__inference_model_2_layer_call_fn_10433

inputs
dense_5_kernel
dense_5_bias
dense_6_kernel
dense_6_bias
dense_7_kernel
dense_7_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_kerneldense_5_biasdense_6_kerneldense_6_biasdense_7_kerneldense_7_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_100232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_model_2_layer_call_fn_9945
input_2
dense_5_kernel
dense_5_bias
dense_6_kernel
dense_6_bias
dense_7_kernel
dense_7_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2dense_5_kerneldense_5_biasdense_6_kerneldense_6_biasdense_7_kerneldense_7_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_99362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
F__inference_dense_5_layer_call_and_return_all_conditional_losses_10503

inputs
dense_5_kernel
dense_5_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_kerneldense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_94872
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_5_activity_regularizer_95132
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????P2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_10788
file_prefix-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: :P:P:PP:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::

_output_shapes
: 
?3
?
B__inference_dense_5_layer_call_and_return_conditional_losses_10487

inputs+
'tensordot_readvariableop_dense_5_kernel'
#biasadd_readvariableop_dense_5_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_10747?
;dense_7_bias_regularizer_square_readvariableop_dense_7_bias
identity??.dense_7/bias/Regularizer/Square/ReadVariableOp?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_7_bias_regularizer_square_readvariableop_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentity dense_7/bias/Regularizer/mul:z:0/^dense_7/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp
?3
?
A__inference_dense_5_layer_call_and_return_conditional_losses_9487

inputs+
'tensordot_readvariableop_dense_5_kernel'
#biasadd_readvariableop_dense_5_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_model_2_layer_call_and_return_conditional_losses_9936

inputs
dense_5_dense_5_kernel
dense_5_dense_5_bias
dense_6_dense_6_kernel
dense_6_dense_6_bias
dense_7_dense_7_kernel
dense_7_dense_7_bias
identity??dense_5/StatefulPartitionedCall?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?dense_6/StatefulPartitionedCall?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?dense_7/StatefulPartitionedCall?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_94872!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_5_activity_regularizer_95132-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_95342
activation_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_95842!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_6_activity_regularizer_96102-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
activation_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_96312
activation_4/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_96812!
dense_7/StatefulPartitionedCall?
+dense_7/ActivityRegularizer/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_7_activity_regularizer_97072-
+dense_7/ActivityRegularizer/PartitionedCall?
!dense_7/ActivityRegularizer/ShapeShape(dense_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_7/ActivityRegularizer/Shape?
/dense_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_7/ActivityRegularizer/strided_slice/stack?
1dense_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_1?
1dense_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_2?
)dense_7/ActivityRegularizer/strided_sliceStridedSlice*dense_7/ActivityRegularizer/Shape:output:08dense_7/ActivityRegularizer/strided_slice/stack:output:0:dense_7/ActivityRegularizer/strided_slice/stack_1:output:0:dense_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_7/ActivityRegularizer/strided_slice?
 dense_7/ActivityRegularizer/CastCast2dense_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_7/ActivityRegularizer/Cast?
#dense_7/ActivityRegularizer/truedivRealDiv4dense_7/ActivityRegularizer/PartitionedCall:output:0$dense_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_7/ActivityRegularizer/truediv?
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_97362
reshape_1/PartitionedCall?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_5/StatefulPartitionedCall/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_10422

inputs
dense_5_kernel
dense_5_bias
dense_6_kernel
dense_6_bias
dense_7_kernel
dense_7_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_kerneldense_5_biasdense_6_kerneldense_6_biasdense_7_kerneldense_7_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_99362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
B__inference_dense_7_layer_call_and_return_conditional_losses_10647

inputs+
'tensordot_readvariableop_dense_7_kernel'
#biasadd_readvariableop_dense_7_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
'__inference_dense_6_layer_call_fn_10574

inputs
dense_6_kernel
dense_6_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_kerneldense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_95842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_10692C
?dense_5_kernel_regularizer_square_readvariableop_dense_5_kernel
identity??0dense_5/kernel/Regularizer/Square/ReadVariableOp?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_5_kernel_regularizer_square_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity"dense_5/kernel/Regularizer/mul:z:01^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp
?
H
,__inference_activation_4_layer_call_fn_10593

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_96312
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_10703?
;dense_5_bias_regularizer_square_readvariableop_dense_5_bias
identity??.dense_5/bias/Regularizer/Square/ReadVariableOp?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_5_bias_regularizer_square_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
IdentityIdentity dense_5/bias/Regularizer/mul:z:0/^dense_5/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp
?
?
'__inference_model_2_layer_call_fn_10032
input_2
dense_5_kernel
dense_5_bias
dense_6_kernel
dense_6_bias
dense_7_kernel
dense_7_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2dense_5_kerneldense_5_biasdense_6_kerneldense_6_biasdense_7_kerneldense_7_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_100232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_2
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_10411

inputs3
/dense_5_tensordot_readvariableop_dense_5_kernel/
+dense_5_biasadd_readvariableop_dense_5_bias3
/dense_6_tensordot_readvariableop_dense_6_kernel/
+dense_6_biasadd_readvariableop_dense_6_bias3
/dense_7_tensordot_readvariableop_dense_7_kernel/
+dense_7_biasadd_readvariableop_dense_7_bias
identity??dense_5/BiasAdd/ReadVariableOp? dense_5/Tensordot/ReadVariableOp?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp? dense_6/Tensordot/ReadVariableOp?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp? dense_7/Tensordot/ReadVariableOp?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
 dense_5/Tensordot/ReadVariableOpReadVariableOp/dense_5_tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes?
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/freeh
dense_5/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape?
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axis?
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2?
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis?
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const?
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod?
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1?
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1?
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axis?
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat?
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack?
dense_5/Tensordot/transpose	Transposeinputs!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_5/Tensordot/transpose?
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_5/Tensordot/Reshape?
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_5/Tensordot/MatMul?
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense_5/Tensordot/Const_2?
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axis?
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1?
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_5/Tensordot?
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_5/BiasAdd?
"dense_5/ActivityRegularizer/SquareSquaredense_5/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2$
"dense_5/ActivityRegularizer/Square?
!dense_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_5/ActivityRegularizer/Const?
dense_5/ActivityRegularizer/SumSum&dense_5/ActivityRegularizer/Square:y:0*dense_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/Sum?
!dense_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_5/ActivityRegularizer/mul/x?
dense_5/ActivityRegularizer/mulMul*dense_5/ActivityRegularizer/mul/x:output:0(dense_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/mul?
!dense_5/ActivityRegularizer/ShapeShapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv#dense_5/ActivityRegularizer/mul:z:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv~
activation_3/ReluReludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation_3/Relu?
 dense_6/Tensordot/ReadVariableOpReadVariableOp/dense_6_tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes?
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free?
dense_6/Tensordot/ShapeShapeactivation_3/Relu:activations:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape?
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis?
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2?
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis?
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const?
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod?
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1?
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1?
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis?
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat?
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack?
dense_6/Tensordot/transpose	Transposeactivation_3/Relu:activations:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_6/Tensordot/transpose?
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_6/Tensordot/Reshape?
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_6/Tensordot/MatMul?
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense_6/Tensordot/Const_2?
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis?
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1?
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_6/Tensordot?
dense_6/BiasAdd/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_6/BiasAdd?
"dense_6/ActivityRegularizer/SquareSquaredense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2$
"dense_6/ActivityRegularizer/Square?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum&dense_6/ActivityRegularizer/Square:y:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/ShapeShapedense_6/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv#dense_6/ActivityRegularizer/mul:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv~
activation_4/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation_4/Relu?
 dense_7/Tensordot/ReadVariableOpReadVariableOp/dense_7_tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes?
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free?
dense_7/Tensordot/ShapeShapeactivation_4/Relu:activations:0*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape?
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis?
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2?
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis?
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const?
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod?
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1?
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1?
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis?
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat?
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack?
dense_7/Tensordot/transpose	Transposeactivation_4/Relu:activations:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_7/Tensordot/transpose?
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_7/Tensordot/Reshape?
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/Tensordot/MatMul?
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2?
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis?
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1?
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_7/Tensordot?
dense_7/BiasAdd/ReadVariableOpReadVariableOp+dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_7/BiasAdd?
"dense_7/ActivityRegularizer/SquareSquaredense_7/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
"dense_7/ActivityRegularizer/Square?
!dense_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_7/ActivityRegularizer/Const?
dense_7/ActivityRegularizer/SumSum&dense_7/ActivityRegularizer/Square:y:0*dense_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_7/ActivityRegularizer/Sum?
!dense_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_7/ActivityRegularizer/mul/x?
dense_7/ActivityRegularizer/mulMul*dense_7/ActivityRegularizer/mul/x:output:0(dense_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_7/ActivityRegularizer/mul?
!dense_7/ActivityRegularizer/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_7/ActivityRegularizer/Shape?
/dense_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_7/ActivityRegularizer/strided_slice/stack?
1dense_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_1?
1dense_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_2?
)dense_7/ActivityRegularizer/strided_sliceStridedSlice*dense_7/ActivityRegularizer/Shape:output:08dense_7/ActivityRegularizer/strided_slice/stack:output:0:dense_7/ActivityRegularizer/strided_slice/stack_1:output:0:dense_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_7/ActivityRegularizer/strided_slice?
 dense_7/ActivityRegularizer/CastCast2dense_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_7/ActivityRegularizer/Cast?
#dense_7/ActivityRegularizer/truedivRealDiv#dense_7/ActivityRegularizer/mul:z:0$dense_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_7/ActivityRegularizer/truedivj
reshape_1/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_7/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_1/Reshape?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_5_tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_6_tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_7_tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentityreshape_1/Reshape:output:0^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
B__inference_dense_6_layer_call_and_return_conditional_losses_10567

inputs+
'tensordot_readvariableop_dense_6_kernel'
#biasadd_readvariableop_dense_6_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
H
,__inference_activation_3_layer_call_fn_10513

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_95342
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
A__inference_model_2_layer_call_and_return_conditional_losses_9857
input_2
dense_5_dense_5_kernel
dense_5_dense_5_bias
dense_6_dense_6_kernel
dense_6_dense_6_bias
dense_7_dense_7_kernel
dense_7_dense_7_bias
identity??dense_5/StatefulPartitionedCall?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?dense_6/StatefulPartitionedCall?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?dense_7/StatefulPartitionedCall?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_94872!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_5_activity_regularizer_95132-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_95342
activation_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_95842!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_6_activity_regularizer_96102-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
activation_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_96312
activation_4/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_96812!
dense_7/StatefulPartitionedCall?
+dense_7/ActivityRegularizer/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_7_activity_regularizer_97072-
+dense_7/ActivityRegularizer/PartitionedCall?
!dense_7/ActivityRegularizer/ShapeShape(dense_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_7/ActivityRegularizer/Shape?
/dense_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_7/ActivityRegularizer/strided_slice/stack?
1dense_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_1?
1dense_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_2?
)dense_7/ActivityRegularizer/strided_sliceStridedSlice*dense_7/ActivityRegularizer/Shape:output:08dense_7/ActivityRegularizer/strided_slice/stack:output:0:dense_7/ActivityRegularizer/strided_slice/stack_1:output:0:dense_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_7/ActivityRegularizer/strided_slice?
 dense_7/ActivityRegularizer/CastCast2dense_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_7/ActivityRegularizer/Cast?
#dense_7/ActivityRegularizer/truedivRealDiv4dense_7/ActivityRegularizer/PartitionedCall:output:0$dense_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_7/ActivityRegularizer/truediv?
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_97362
reshape_1/PartitionedCall?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_5/StatefulPartitionedCall/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?
b
F__inference_activation_4_layer_call_and_return_conditional_losses_9631

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:?????????P2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_9736

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_10246

inputs3
/dense_5_tensordot_readvariableop_dense_5_kernel/
+dense_5_biasadd_readvariableop_dense_5_bias3
/dense_6_tensordot_readvariableop_dense_6_kernel/
+dense_6_biasadd_readvariableop_dense_6_bias3
/dense_7_tensordot_readvariableop_dense_7_kernel/
+dense_7_biasadd_readvariableop_dense_7_bias
identity??dense_5/BiasAdd/ReadVariableOp? dense_5/Tensordot/ReadVariableOp?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp? dense_6/Tensordot/ReadVariableOp?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp? dense_7/Tensordot/ReadVariableOp?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
 dense_5/Tensordot/ReadVariableOpReadVariableOp/dense_5_tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes?
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/freeh
dense_5/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape?
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axis?
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2?
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis?
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const?
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod?
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1?
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1?
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axis?
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat?
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack?
dense_5/Tensordot/transpose	Transposeinputs!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_5/Tensordot/transpose?
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_5/Tensordot/Reshape?
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_5/Tensordot/MatMul?
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense_5/Tensordot/Const_2?
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axis?
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1?
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_5/Tensordot?
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_5/BiasAdd?
"dense_5/ActivityRegularizer/SquareSquaredense_5/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2$
"dense_5/ActivityRegularizer/Square?
!dense_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_5/ActivityRegularizer/Const?
dense_5/ActivityRegularizer/SumSum&dense_5/ActivityRegularizer/Square:y:0*dense_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/Sum?
!dense_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_5/ActivityRegularizer/mul/x?
dense_5/ActivityRegularizer/mulMul*dense_5/ActivityRegularizer/mul/x:output:0(dense_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/mul?
!dense_5/ActivityRegularizer/ShapeShapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv#dense_5/ActivityRegularizer/mul:z:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv~
activation_3/ReluReludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation_3/Relu?
 dense_6/Tensordot/ReadVariableOpReadVariableOp/dense_6_tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes?
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free?
dense_6/Tensordot/ShapeShapeactivation_3/Relu:activations:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape?
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis?
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2?
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis?
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const?
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod?
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1?
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1?
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis?
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat?
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack?
dense_6/Tensordot/transpose	Transposeactivation_3/Relu:activations:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_6/Tensordot/transpose?
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_6/Tensordot/Reshape?
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_6/Tensordot/MatMul?
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense_6/Tensordot/Const_2?
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis?
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1?
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_6/Tensordot?
dense_6/BiasAdd/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_6/BiasAdd?
"dense_6/ActivityRegularizer/SquareSquaredense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2$
"dense_6/ActivityRegularizer/Square?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum&dense_6/ActivityRegularizer/Square:y:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/ShapeShapedense_6/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv#dense_6/ActivityRegularizer/mul:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv~
activation_4/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation_4/Relu?
 dense_7/Tensordot/ReadVariableOpReadVariableOp/dense_7_tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes?
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free?
dense_7/Tensordot/ShapeShapeactivation_4/Relu:activations:0*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape?
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis?
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2?
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis?
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const?
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod?
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1?
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1?
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis?
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat?
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack?
dense_7/Tensordot/transpose	Transposeactivation_4/Relu:activations:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_7/Tensordot/transpose?
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_7/Tensordot/Reshape?
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/Tensordot/MatMul?
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2?
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis?
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1?
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_7/Tensordot?
dense_7/BiasAdd/ReadVariableOpReadVariableOp+dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_7/BiasAdd?
"dense_7/ActivityRegularizer/SquareSquaredense_7/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
"dense_7/ActivityRegularizer/Square?
!dense_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_7/ActivityRegularizer/Const?
dense_7/ActivityRegularizer/SumSum&dense_7/ActivityRegularizer/Square:y:0*dense_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_7/ActivityRegularizer/Sum?
!dense_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_7/ActivityRegularizer/mul/x?
dense_7/ActivityRegularizer/mulMul*dense_7/ActivityRegularizer/mul/x:output:0(dense_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_7/ActivityRegularizer/mul?
!dense_7/ActivityRegularizer/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_7/ActivityRegularizer/Shape?
/dense_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_7/ActivityRegularizer/strided_slice/stack?
1dense_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_1?
1dense_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_2?
)dense_7/ActivityRegularizer/strided_sliceStridedSlice*dense_7/ActivityRegularizer/Shape:output:08dense_7/ActivityRegularizer/strided_slice/stack:output:0:dense_7/ActivityRegularizer/strided_slice/stack_1:output:0:dense_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_7/ActivityRegularizer/strided_slice?
 dense_7/ActivityRegularizer/CastCast2dense_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_7/ActivityRegularizer/Cast?
#dense_7/ActivityRegularizer/truedivRealDiv#dense_7/ActivityRegularizer/mul:z:0$dense_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_7/ActivityRegularizer/truedivj
reshape_1/ShapeShapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_7/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_1/Reshape?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_5_tensordot_readvariableop_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_6_tensordot_readvariableop_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_7_tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_7_biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentityreshape_1/Reshape:output:0^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_activation_3_layer_call_and_return_conditional_losses_9534

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:?????????P2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_10081
input_2
dense_5_kernel
dense_5_bias
dense_6_kernel
dense_6_bias
dense_7_kernel
dense_7_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2dense_5_kerneldense_5_biasdense_6_kerneldense_6_biasdense_7_kerneldense_7_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_94022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
A__inference_model_2_layer_call_and_return_conditional_losses_9781
input_2
dense_5_dense_5_kernel
dense_5_dense_5_bias
dense_6_dense_6_kernel
dense_6_dense_6_bias
dense_7_dense_7_kernel
dense_7_dense_7_bias
identity??dense_5/StatefulPartitionedCall?.dense_5/bias/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?dense_6/StatefulPartitionedCall?.dense_6/bias/Regularizer/Square/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?dense_7/StatefulPartitionedCall?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_94872!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_5_activity_regularizer_95132-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_95342
activation_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_95842!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_6_activity_regularizer_96102-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
activation_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_96312
activation_4/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_7_dense_7_kerneldense_7_dense_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_96812!
dense_7/StatefulPartitionedCall?
+dense_7/ActivityRegularizer/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *6
f1R/
-__inference_dense_7_activity_regularizer_97072-
+dense_7/ActivityRegularizer/PartitionedCall?
!dense_7/ActivityRegularizer/ShapeShape(dense_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_7/ActivityRegularizer/Shape?
/dense_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_7/ActivityRegularizer/strided_slice/stack?
1dense_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_1?
1dense_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_7/ActivityRegularizer/strided_slice/stack_2?
)dense_7/ActivityRegularizer/strided_sliceStridedSlice*dense_7/ActivityRegularizer/Shape:output:08dense_7/ActivityRegularizer/strided_slice/stack:output:0:dense_7/ActivityRegularizer/strided_slice/stack_1:output:0:dense_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_7/ActivityRegularizer/strided_slice?
 dense_7/ActivityRegularizer/CastCast2dense_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_7/ActivityRegularizer/Cast?
#dense_7/ActivityRegularizer/truedivRealDiv4dense_7/ActivityRegularizer/PartitionedCall:output:0$dense_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_7/ActivityRegularizer/truediv?
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_97362
reshape_1/PartitionedCall?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_kernel*
_output_shapes

:P*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_dense_5_bias*
_output_shapes
:P*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOp?
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_5/bias/Regularizer/Square?
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/Const?
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum?
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_5/bias/Regularizer/mul/x?
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_kernel*
_output_shapes

:PP*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_7_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_5/StatefulPartitionedCall/^dense_5/bias/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall/^dense_6/bias/Regularizer/Square/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2`
.dense_5/bias/Regularizer/Square/ReadVariableOp.dense_5/bias/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?3
?
A__inference_dense_7_layer_call_and_return_conditional_losses_9681

inputs+
'tensordot_readvariableop_dense_7_kernel'
#biasadd_readvariableop_dense_7_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_7/bias/Regularizer/Square/ReadVariableOp?0dense_7/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_7_kernel*
_output_shapes

:P*
dtype022
0dense_7/kernel/Regularizer/Square/ReadVariableOp?
!dense_7/kernel/Regularizer/SquareSquare8dense_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_7/kernel/Regularizer/Square?
 dense_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_7/kernel/Regularizer/Const?
dense_7/kernel/Regularizer/SumSum%dense_7/kernel/Regularizer/Square:y:0)dense_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/Sum?
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_7/kernel/Regularizer/mul/x?
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0'dense_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_7/kernel/Regularizer/mul?
.dense_7/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_7_bias*
_output_shapes
:*
dtype020
.dense_7/bias/Regularizer/Square/ReadVariableOp?
dense_7/bias/Regularizer/SquareSquare6dense_7/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_7/bias/Regularizer/Square?
dense_7/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_7/bias/Regularizer/Const?
dense_7/bias/Regularizer/SumSum#dense_7/bias/Regularizer/Square:y:0'dense_7/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/Sum?
dense_7/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_7/bias/Regularizer/mul/x?
dense_7/bias/Regularizer/mulMul'dense_7/bias/Regularizer/mul/x:output:0%dense_7/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_7/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_7/bias/Regularizer/Square/ReadVariableOp1^dense_7/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_7/bias/Regularizer/Square/ReadVariableOp.dense_7/bias/Regularizer/Square/ReadVariableOp2d
0dense_7/kernel/Regularizer/Square/ReadVariableOp0dense_7/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_10725?
;dense_6_bias_regularizer_square_readvariableop_dense_6_bias
identity??.dense_6/bias/Regularizer/Square/ReadVariableOp?
.dense_6/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_6_bias_regularizer_square_readvariableop_dense_6_bias*
_output_shapes
:P*
dtype020
.dense_6/bias/Regularizer/Square/ReadVariableOp?
dense_6/bias/Regularizer/SquareSquare6dense_6/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_6/bias/Regularizer/Square?
dense_6/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_6/bias/Regularizer/Const?
dense_6/bias/Regularizer/SumSum#dense_6/bias/Regularizer/Square:y:0'dense_6/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/Sum?
dense_6/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_6/bias/Regularizer/mul/x?
dense_6/bias/Regularizer/mulMul'dense_6/bias/Regularizer/mul/x:output:0%dense_6/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_6/bias/Regularizer/mul?
IdentityIdentity dense_6/bias/Regularizer/mul:z:0/^dense_6/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense_6/bias/Regularizer/Square/ReadVariableOp.dense_6/bias/Regularizer/Square/ReadVariableOp
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_10676

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
-__inference_dense_5_activity_regularizer_9415
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_24
serving_default_input_2:0?????????A
	reshape_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?8
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
regularization_losses
		variables

trainable_variables
	keras_api

signatures
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses"?5
_tf_keras_network?5{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 4]}}, "name": "reshape_1", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["reshape_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 4]}}, "name": "reshape_1", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["reshape_1", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 2]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 80]}}
?
regularization_losses
	variables
trainable_variables
 	keras_api
W__call__
*X&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 80]}}
?
'regularization_losses
(	variables
)trainable_variables
*	keras_api
[__call__
*\&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 4]}}}
J
]0
^1
_2
`3
a4
b5"
trackable_list_wrapper
J
0
1
2
3
!4
"5"
trackable_list_wrapper
J
0
1
2
3
!4
"5"
trackable_list_wrapper
?
regularization_losses
		variables
+metrics

trainable_variables
,layer_regularization_losses
-layer_metrics

.layers
/non_trainable_variables
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
,
cserving_default"
signature_map
 :P2dense_5/kernel
:P2dense_5/bias
.
]0
^1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
0metrics
trainable_variables
1layer_regularization_losses
2layer_metrics

3layers
4non_trainable_variables
Q__call__
dactivity_regularizer_fn
*R&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables
5metrics
trainable_variables
6layer_regularization_losses
7layer_metrics

8layers
9non_trainable_variables
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 :PP2dense_6/kernel
:P2dense_6/bias
.
_0
`1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
:metrics
trainable_variables
;layer_regularization_losses
<layer_metrics

=layers
>non_trainable_variables
U__call__
factivity_regularizer_fn
*V&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables
?metrics
trainable_variables
@layer_regularization_losses
Alayer_metrics

Blayers
Cnon_trainable_variables
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 :P2dense_7/kernel
:2dense_7/bias
.
a0
b1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
#regularization_losses
$	variables
Dmetrics
%trainable_variables
Elayer_regularization_losses
Flayer_metrics

Glayers
Hnon_trainable_variables
Y__call__
hactivity_regularizer_fn
*Z&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'regularization_losses
(	variables
Imetrics
)trainable_variables
Jlayer_regularization_losses
Klayer_metrics

Llayers
Mnon_trainable_variables
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
'__inference_model_2_layer_call_fn_10032
'__inference_model_2_layer_call_fn_10422
&__inference_model_2_layer_call_fn_9945
'__inference_model_2_layer_call_fn_10433?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_9402?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
input_2?????????
?2?
B__inference_model_2_layer_call_and_return_conditional_losses_10411
A__inference_model_2_layer_call_and_return_conditional_losses_9781
B__inference_model_2_layer_call_and_return_conditional_losses_10246
A__inference_model_2_layer_call_and_return_conditional_losses_9857?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_5_layer_call_fn_10494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_5_layer_call_and_return_all_conditional_losses_10503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_3_layer_call_fn_10513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_3_layer_call_and_return_conditional_losses_10508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_6_layer_call_fn_10574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_6_layer_call_and_return_all_conditional_losses_10583?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_4_layer_call_fn_10593?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_4_layer_call_and_return_conditional_losses_10588?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_7_layer_call_fn_10654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_7_layer_call_and_return_all_conditional_losses_10663?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_1_layer_call_fn_10681?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_1_layer_call_and_return_conditional_losses_10676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_10692?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_10703?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_10714?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_10725?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_10736?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_10747?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
#__inference_signature_wrapper_10081input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_5_activity_regularizer_9415?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_10487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_6_activity_regularizer_9428?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
B__inference_dense_6_layer_call_and_return_conditional_losses_10567?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_7_activity_regularizer_9441?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
B__inference_dense_7_layer_call_and_return_conditional_losses_10647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_9402y!"4?1
*?'
%?"
input_2?????????
? "9?6
4
	reshape_1'?$
	reshape_1??????????
G__inference_activation_3_layer_call_and_return_conditional_losses_10508`3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????P
? ?
,__inference_activation_3_layer_call_fn_10513S3?0
)?&
$?!
inputs?????????P
? "??????????P?
G__inference_activation_4_layer_call_and_return_conditional_losses_10588`3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????P
? ?
,__inference_activation_4_layer_call_fn_10593S3?0
)?&
$?!
inputs?????????P
? "??????????PZ
-__inference_dense_5_activity_regularizer_9415)?
?
?
self
? "? ?
F__inference_dense_5_layer_call_and_return_all_conditional_losses_10503r3?0
)?&
$?!
inputs?????????
? "7?4
?
0?????????P
?
?	
1/0 ?
B__inference_dense_5_layer_call_and_return_conditional_losses_10487d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????P
? ?
'__inference_dense_5_layer_call_fn_10494W3?0
)?&
$?!
inputs?????????
? "??????????PZ
-__inference_dense_6_activity_regularizer_9428)?
?
?
self
? "? ?
F__inference_dense_6_layer_call_and_return_all_conditional_losses_10583r3?0
)?&
$?!
inputs?????????P
? "7?4
?
0?????????P
?
?	
1/0 ?
B__inference_dense_6_layer_call_and_return_conditional_losses_10567d3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????P
? ?
'__inference_dense_6_layer_call_fn_10574W3?0
)?&
$?!
inputs?????????P
? "??????????PZ
-__inference_dense_7_activity_regularizer_9441)?
?
?
self
? "? ?
F__inference_dense_7_layer_call_and_return_all_conditional_losses_10663r!"3?0
)?&
$?!
inputs?????????P
? "7?4
?
0?????????
?
?	
1/0 ?
B__inference_dense_7_layer_call_and_return_conditional_losses_10647d!"3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????
? ?
'__inference_dense_7_layer_call_fn_10654W!"3?0
)?&
$?!
inputs?????????P
? "??????????:
__inference_loss_fn_0_10692?

? 
? "? :
__inference_loss_fn_1_10703?

? 
? "? :
__inference_loss_fn_2_10714?

? 
? "? :
__inference_loss_fn_3_10725?

? 
? "? :
__inference_loss_fn_4_10736!?

? 
? "? :
__inference_loss_fn_5_10747"?

? 
? "? ?
B__inference_model_2_layer_call_and_return_conditional_losses_10246p!";?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_10411p!";?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
A__inference_model_2_layer_call_and_return_conditional_losses_9781q!"<?9
2?/
%?"
input_2?????????
p

 
? ")?&
?
0?????????
? ?
A__inference_model_2_layer_call_and_return_conditional_losses_9857q!"<?9
2?/
%?"
input_2?????????
p 

 
? ")?&
?
0?????????
? ?
'__inference_model_2_layer_call_fn_10032d!"<?9
2?/
%?"
input_2?????????
p 

 
? "???????????
'__inference_model_2_layer_call_fn_10422c!";?8
1?.
$?!
inputs?????????
p

 
? "???????????
'__inference_model_2_layer_call_fn_10433c!";?8
1?.
$?!
inputs?????????
p 

 
? "???????????
&__inference_model_2_layer_call_fn_9945d!"<?9
2?/
%?"
input_2?????????
p

 
? "???????????
D__inference_reshape_1_layer_call_and_return_conditional_losses_10676`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
)__inference_reshape_1_layer_call_fn_10681S3?0
)?&
$?!
inputs?????????
? "???????????
#__inference_signature_wrapper_10081?!"??<
? 
5?2
0
input_2%?"
input_2?????????"9?6
4
	reshape_1'?$
	reshape_1?????????