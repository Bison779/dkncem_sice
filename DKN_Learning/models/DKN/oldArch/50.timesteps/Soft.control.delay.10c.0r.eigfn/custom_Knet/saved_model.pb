ȗ
??
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
,
Cos
x"T
y"T"
Ttype:

2
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
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
0
Neg
x"T
y"T"
Ttype:
2
	
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
,
Sin
x"T
y"T"
Ttype:

2
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
trainable_variables
regularization_losses
	variables
		keras_api


signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
_
kernels
trainable_variables
regularization_losses
	variables
	keras_api

0
1
2
3
 

0
1
2
3
?
 layer_metrics
!metrics

"layers
#non_trainable_variables
trainable_variables
$layer_regularization_losses
regularization_losses
	variables
 
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
%layer_metrics
&metrics

'layers
(non_trainable_variables
trainable_variables
)layer_regularization_losses
regularization_losses
	variables
 
 
 
?
*layer_metrics
+metrics

,layers
-non_trainable_variables
trainable_variables
.layer_regularization_losses
regularization_losses
	variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
/layer_metrics
0metrics

1layers
2non_trainable_variables
trainable_variables
3layer_regularization_losses
regularization_losses
	variables
 
 
 
 
?
4layer_metrics
5metrics

6layers
7non_trainable_variables
trainable_variables
8layer_regularization_losses
regularization_losses
	variables
 
 
#
0
1
2
3
4
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
serving_default_input_3Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_35543
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpConst*
Tin

2*
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
__inference__traced_save_37020
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin	
2*
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
!__inference__traced_restore_37042ة
?Z
?
B__inference_model_4_layer_call_and_return_conditional_losses_35497

inputs
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_348112!
dense_3/StatefulPartitionedCall?
+dense_3/ActivityRegularizer/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_3_activity_regularizer_348372-
+dense_3/ActivityRegularizer/PartitionedCall?
!dense_3/ActivityRegularizer/ShapeShape(dense_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_3/ActivityRegularizer/Shape?
/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_3/ActivityRegularizer/strided_slice/stack?
1dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_1?
1dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_2?
)dense_3/ActivityRegularizer/strided_sliceStridedSlice*dense_3/ActivityRegularizer/Shape:output:08dense_3/ActivityRegularizer/strided_slice/stack:output:0:dense_3/ActivityRegularizer/strided_slice/stack_1:output:0:dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_3/ActivityRegularizer/strided_slice?
 dense_3/ActivityRegularizer/CastCast2dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_3/ActivityRegularizer/Cast?
#dense_3/ActivityRegularizer/truedivRealDiv4dense_3/ActivityRegularizer/PartitionedCall:output:0$dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_3/ActivityRegularizer/truediv?
activation_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_348582
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_349082!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_4_activity_regularizer_349342-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
linear_update/PartitionedCallPartitionedCallinputs(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_352892
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
-__inference_linear_update_layer_call_fn_36940
x_0
x_1
identity?
PartitionedCallPartitionedCallx_0x_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_352892
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????:?????????:P L
+
_output_shapes
:?????????

_user_specified_namex/0:PL
+
_output_shapes
:?????????

_user_specified_namex/1
?
?
__inference_loss_fn_1_36962?
;dense_3_bias_regularizer_square_readvariableop_dense_3_bias
identity??.dense_3/bias/Regularizer/Square/ReadVariableOp?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_3_bias_regularizer_square_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentity dense_3/bias/Regularizer/mul:z:0/^dense_3/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp
?3
?
B__inference_dense_4_layer_call_and_return_conditional_losses_34908

inputs+
'tensordot_readvariableop_dense_4_kernel'
#biasadd_readvariableop_dense_4_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
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
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
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
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?3
?
B__inference_dense_4_layer_call_and_return_conditional_losses_36579

inputs+
'tensordot_readvariableop_dense_4_kernel'
#biasadd_readvariableop_dense_4_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
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
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
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
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_36586

inputs
dense_4_kernel
dense_4_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_349082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
'__inference_model_4_layer_call_fn_35506
input_3
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3dense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_354972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
??
j
H__inference_linear_update_layer_call_and_return_conditional_losses_36934
x_0
x_1
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlicex_0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlicex_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2S
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
mul/yi
mulMulstrided_slice_2:output:0mul/y:output:0*
T0*#
_output_shapes
:?????????2
mulH
ExpExpmul:z:0*
T0*#
_output_shapes
:?????????2
Exp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3W
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_1/yo
mul_1Mulstrided_slice_3:output:0mul_1/y:output:0*
T0*#
_output_shapes
:?????????2
mul_1J
CosCos	mul_1:z:0*
T0*#
_output_shapes
:?????????2
Cos
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4W
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_2/yo
mul_2Mulstrided_slice_4:output:0mul_2/y:output:0*
T0*#
_output_shapes
:?????????2
mul_2J
SinSin	mul_2:z:0*
T0*#
_output_shapes
:?????????2
SinU
Mul_3MulExp:y:0Cos:y:0*
T0*#
_output_shapes
:?????????2
Mul_3U
Mul_4MulExp:y:0Sin:y:0*
T0*#
_output_shapes
:?????????2
Mul_4J
NegNeg	Mul_4:z:0*
T0*#
_output_shapes
:?????????2
Neg?
stackPack	Mul_3:z:0Neg:y:0	Mul_4:z:0	Mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
stacks
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shape{
ReshapeReshapestack:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshape
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_5?
einsum/EinsumEinsumstrided_slice_5:output:0Reshape:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum/Einsum
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_5/yo
mul_5Mulstrided_slice_6:output:0mul_5/y:output:0*
T0*#
_output_shapes
:?????????2
mul_5N
Exp_1Exp	mul_5:z:0*
T0*#
_output_shapes
:?????????2
Exp_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_6/yo
mul_6Mulstrided_slice_7:output:0mul_6/y:output:0*
T0*#
_output_shapes
:?????????2
mul_6N
Cos_1Cos	mul_6:z:0*
T0*#
_output_shapes
:?????????2
Cos_1
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8W
mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_7/yo
mul_7Mulstrided_slice_8:output:0mul_7/y:output:0*
T0*#
_output_shapes
:?????????2
mul_7N
Sin_1Sin	mul_7:z:0*
T0*#
_output_shapes
:?????????2
Sin_1Y
Mul_8Mul	Exp_1:y:0	Cos_1:y:0*
T0*#
_output_shapes
:?????????2
Mul_8Y
Mul_9Mul	Exp_1:y:0	Sin_1:y:0*
T0*#
_output_shapes
:?????????2
Mul_9N
Neg_1Neg	Mul_9:z:0*
T0*#
_output_shapes
:?????????2
Neg_1?
stack_1Pack	Mul_8:z:0	Neg_1:y:0	Mul_9:z:0	Mul_8:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_1w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapestack_1:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_9?
einsum_1/EinsumEinsumstrided_slice_9:output:0Reshape_1:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_1/Einsum?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10Y
mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_10/ys
mul_10Mulstrided_slice_10:output:0mul_10/y:output:0*
T0*#
_output_shapes
:?????????2
mul_10O
Exp_2Exp
mul_10:z:0*
T0*#
_output_shapes
:?????????2
Exp_2?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11Y
mul_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_11/ys
mul_11Mulstrided_slice_11:output:0mul_11/y:output:0*
T0*#
_output_shapes
:?????????2
mul_11O
Cos_2Cos
mul_11:z:0*
T0*#
_output_shapes
:?????????2
Cos_2?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12Y
mul_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_12/ys
mul_12Mulstrided_slice_12:output:0mul_12/y:output:0*
T0*#
_output_shapes
:?????????2
mul_12O
Sin_2Sin
mul_12:z:0*
T0*#
_output_shapes
:?????????2
Sin_2[
Mul_13Mul	Exp_2:y:0	Cos_2:y:0*
T0*#
_output_shapes
:?????????2
Mul_13[
Mul_14Mul	Exp_2:y:0	Sin_2:y:0*
T0*#
_output_shapes
:?????????2
Mul_14O
Neg_2Neg
Mul_14:z:0*
T0*#
_output_shapes
:?????????2
Neg_2?
stack_2Pack
Mul_13:z:0	Neg_2:y:0
Mul_14:z:0
Mul_13:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_2w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_2/shape?
	Reshape_2Reshapestack_2:output:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_2?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_13?
einsum_2/EinsumEinsumstrided_slice_13:output:0Reshape_2:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_2/Einsum?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14Y
mul_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_15/ys
mul_15Mulstrided_slice_14:output:0mul_15/y:output:0*
T0*#
_output_shapes
:?????????2
mul_15O
Exp_3Exp
mul_15:z:0*
T0*#
_output_shapes
:?????????2
Exp_3?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15Y
mul_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_16/ys
mul_16Mulstrided_slice_15:output:0mul_16/y:output:0*
T0*#
_output_shapes
:?????????2
mul_16O
Cos_3Cos
mul_16:z:0*
T0*#
_output_shapes
:?????????2
Cos_3?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16Y
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_17/ys
mul_17Mulstrided_slice_16:output:0mul_17/y:output:0*
T0*#
_output_shapes
:?????????2
mul_17O
Sin_3Sin
mul_17:z:0*
T0*#
_output_shapes
:?????????2
Sin_3[
Mul_18Mul	Exp_3:y:0	Cos_3:y:0*
T0*#
_output_shapes
:?????????2
Mul_18[
Mul_19Mul	Exp_3:y:0	Sin_3:y:0*
T0*#
_output_shapes
:?????????2
Mul_19O
Neg_3Neg
Mul_19:z:0*
T0*#
_output_shapes
:?????????2
Neg_3?
stack_3Pack
Mul_18:z:0	Neg_3:y:0
Mul_19:z:0
Mul_18:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_3w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_3/shape?
	Reshape_3Reshapestack_3:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_3?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_17?
einsum_3/EinsumEinsumstrided_slice_17:output:0Reshape_3:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_3/Einsum?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18Y
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_20/ys
mul_20Mulstrided_slice_18:output:0mul_20/y:output:0*
T0*#
_output_shapes
:?????????2
mul_20O
Exp_4Exp
mul_20:z:0*
T0*#
_output_shapes
:?????????2
Exp_4?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19Y
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_21/ys
mul_21Mulstrided_slice_19:output:0mul_21/y:output:0*
T0*#
_output_shapes
:?????????2
mul_21O
Cos_4Cos
mul_21:z:0*
T0*#
_output_shapes
:?????????2
Cos_4?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20Y
mul_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_22/ys
mul_22Mulstrided_slice_20:output:0mul_22/y:output:0*
T0*#
_output_shapes
:?????????2
mul_22O
Sin_4Sin
mul_22:z:0*
T0*#
_output_shapes
:?????????2
Sin_4[
Mul_23Mul	Exp_4:y:0	Cos_4:y:0*
T0*#
_output_shapes
:?????????2
Mul_23[
Mul_24Mul	Exp_4:y:0	Sin_4:y:0*
T0*#
_output_shapes
:?????????2
Mul_24O
Neg_4Neg
Mul_24:z:0*
T0*#
_output_shapes
:?????????2
Neg_4?
stack_4Pack
Mul_23:z:0	Neg_4:y:0
Mul_24:z:0
Mul_23:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_4w
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_4/shape?
	Reshape_4Reshapestack_4:output:0Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_4?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_21?
einsum_4/EinsumEinsumstrided_slice_21:output:0Reshape_4:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_4/Einsum?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22Y
mul_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_25/ys
mul_25Mulstrided_slice_22:output:0mul_25/y:output:0*
T0*#
_output_shapes
:?????????2
mul_25O
Exp_5Exp
mul_25:z:0*
T0*#
_output_shapes
:?????????2
Exp_5?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23Y
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_26/ys
mul_26Mulstrided_slice_23:output:0mul_26/y:output:0*
T0*#
_output_shapes
:?????????2
mul_26O
Cos_5Cos
mul_26:z:0*
T0*#
_output_shapes
:?????????2
Cos_5?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24Y
mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_27/ys
mul_27Mulstrided_slice_24:output:0mul_27/y:output:0*
T0*#
_output_shapes
:?????????2
mul_27O
Sin_5Sin
mul_27:z:0*
T0*#
_output_shapes
:?????????2
Sin_5[
Mul_28Mul	Exp_5:y:0	Cos_5:y:0*
T0*#
_output_shapes
:?????????2
Mul_28[
Mul_29Mul	Exp_5:y:0	Sin_5:y:0*
T0*#
_output_shapes
:?????????2
Mul_29O
Neg_5Neg
Mul_29:z:0*
T0*#
_output_shapes
:?????????2
Neg_5?
stack_5Pack
Mul_28:z:0	Neg_5:y:0
Mul_29:z:0
Mul_28:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_5w
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_5/shape?
	Reshape_5Reshapestack_5:output:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_5?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_25?
einsum_5/EinsumEinsumstrided_slice_25:output:0Reshape_5:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_5/Einsum?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26Y
mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_30/ys
mul_30Mulstrided_slice_26:output:0mul_30/y:output:0*
T0*#
_output_shapes
:?????????2
mul_30O
Exp_6Exp
mul_30:z:0*
T0*#
_output_shapes
:?????????2
Exp_6?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27Y
mul_31/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_31/ys
mul_31Mulstrided_slice_27:output:0mul_31/y:output:0*
T0*#
_output_shapes
:?????????2
mul_31O
Cos_6Cos
mul_31:z:0*
T0*#
_output_shapes
:?????????2
Cos_6?
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_28/stack?
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_28/stack_1?
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_28/stack_2?
strided_slice_28StridedSlicestrided_slice_1:output:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_28Y
mul_32/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_32/ys
mul_32Mulstrided_slice_28:output:0mul_32/y:output:0*
T0*#
_output_shapes
:?????????2
mul_32O
Sin_6Sin
mul_32:z:0*
T0*#
_output_shapes
:?????????2
Sin_6[
Mul_33Mul	Exp_6:y:0	Cos_6:y:0*
T0*#
_output_shapes
:?????????2
Mul_33[
Mul_34Mul	Exp_6:y:0	Sin_6:y:0*
T0*#
_output_shapes
:?????????2
Mul_34O
Neg_6Neg
Mul_34:z:0*
T0*#
_output_shapes
:?????????2
Neg_6?
stack_6Pack
Mul_33:z:0	Neg_6:y:0
Mul_34:z:0
Mul_33:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_6w
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_6/shape?
	Reshape_6Reshapestack_6:output:0Reshape_6/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_6?
strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_29/stack?
strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_29/stack_1?
strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_29/stack_2?
strided_slice_29StridedSlicestrided_slice:output:0strided_slice_29/stack:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_29?
einsum_6/EinsumEinsumstrided_slice_29:output:0Reshape_6:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_6/Einsum?
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_30/stack?
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_30/stack_1?
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_30/stack_2?
strided_slice_30StridedSlicestrided_slice_1:output:0strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_30Y
mul_35/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_35/ys
mul_35Mulstrided_slice_30:output:0mul_35/y:output:0*
T0*#
_output_shapes
:?????????2
mul_35O
Exp_7Exp
mul_35:z:0*
T0*#
_output_shapes
:?????????2
Exp_7?
strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_31/stack?
strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_31/stack_1?
strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_31/stack_2?
strided_slice_31StridedSlicestrided_slice_1:output:0strided_slice_31/stack:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_31Y
mul_36/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_36/ys
mul_36Mulstrided_slice_31:output:0mul_36/y:output:0*
T0*#
_output_shapes
:?????????2
mul_36O
Cos_7Cos
mul_36:z:0*
T0*#
_output_shapes
:?????????2
Cos_7?
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_32/stack?
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_32/stack_1?
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_32/stack_2?
strided_slice_32StridedSlicestrided_slice_1:output:0strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_32Y
mul_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_37/ys
mul_37Mulstrided_slice_32:output:0mul_37/y:output:0*
T0*#
_output_shapes
:?????????2
mul_37O
Sin_7Sin
mul_37:z:0*
T0*#
_output_shapes
:?????????2
Sin_7[
Mul_38Mul	Exp_7:y:0	Cos_7:y:0*
T0*#
_output_shapes
:?????????2
Mul_38[
Mul_39Mul	Exp_7:y:0	Sin_7:y:0*
T0*#
_output_shapes
:?????????2
Mul_39O
Neg_7Neg
Mul_39:z:0*
T0*#
_output_shapes
:?????????2
Neg_7?
stack_7Pack
Mul_38:z:0	Neg_7:y:0
Mul_39:z:0
Mul_38:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_7w
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_7/shape?
	Reshape_7Reshapestack_7:output:0Reshape_7/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_7?
strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_33/stack?
strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_33/stack_1?
strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_33/stack_2?
strided_slice_33StridedSlicestrided_slice:output:0strided_slice_33/stack:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_33?
einsum_7/EinsumEinsumstrided_slice_33:output:0Reshape_7:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_7/Einsum?
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_34/stack?
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_34/stack_1?
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_34/stack_2?
strided_slice_34StridedSlicestrided_slice_1:output:0strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_34Y
mul_40/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_40/ys
mul_40Mulstrided_slice_34:output:0mul_40/y:output:0*
T0*#
_output_shapes
:?????????2
mul_40O
Exp_8Exp
mul_40:z:0*
T0*#
_output_shapes
:?????????2
Exp_8?
strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_35/stack?
strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_35/stack_1?
strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_35/stack_2?
strided_slice_35StridedSlicestrided_slice_1:output:0strided_slice_35/stack:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_35Y
mul_41/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_41/ys
mul_41Mulstrided_slice_35:output:0mul_41/y:output:0*
T0*#
_output_shapes
:?????????2
mul_41O
Cos_8Cos
mul_41:z:0*
T0*#
_output_shapes
:?????????2
Cos_8?
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_36/stack?
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_36/stack_1?
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_36/stack_2?
strided_slice_36StridedSlicestrided_slice_1:output:0strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_36Y
mul_42/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_42/ys
mul_42Mulstrided_slice_36:output:0mul_42/y:output:0*
T0*#
_output_shapes
:?????????2
mul_42O
Sin_8Sin
mul_42:z:0*
T0*#
_output_shapes
:?????????2
Sin_8[
Mul_43Mul	Exp_8:y:0	Cos_8:y:0*
T0*#
_output_shapes
:?????????2
Mul_43[
Mul_44Mul	Exp_8:y:0	Sin_8:y:0*
T0*#
_output_shapes
:?????????2
Mul_44O
Neg_8Neg
Mul_44:z:0*
T0*#
_output_shapes
:?????????2
Neg_8?
stack_8Pack
Mul_43:z:0	Neg_8:y:0
Mul_44:z:0
Mul_43:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_8w
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_8/shape?
	Reshape_8Reshapestack_8:output:0Reshape_8/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_8?
strided_slice_37/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_37/stack?
strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_37/stack_1?
strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_37/stack_2?
strided_slice_37StridedSlicestrided_slice:output:0strided_slice_37/stack:output:0!strided_slice_37/stack_1:output:0!strided_slice_37/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_37?
einsum_8/EinsumEinsumstrided_slice_37:output:0Reshape_8:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_8/Einsum?
strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_38/stack?
strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_38/stack_1?
strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_38/stack_2?
strided_slice_38StridedSlicestrided_slice_1:output:0strided_slice_38/stack:output:0!strided_slice_38/stack_1:output:0!strided_slice_38/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_38Y
mul_45/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_45/ys
mul_45Mulstrided_slice_38:output:0mul_45/y:output:0*
T0*#
_output_shapes
:?????????2
mul_45O
Exp_9Exp
mul_45:z:0*
T0*#
_output_shapes
:?????????2
Exp_9?
strided_slice_39/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_39/stack?
strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_39/stack_1?
strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_39/stack_2?
strided_slice_39StridedSlicestrided_slice_1:output:0strided_slice_39/stack:output:0!strided_slice_39/stack_1:output:0!strided_slice_39/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_39Y
mul_46/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_46/ys
mul_46Mulstrided_slice_39:output:0mul_46/y:output:0*
T0*#
_output_shapes
:?????????2
mul_46O
Cos_9Cos
mul_46:z:0*
T0*#
_output_shapes
:?????????2
Cos_9?
strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_40/stack?
strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_40/stack_1?
strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_40/stack_2?
strided_slice_40StridedSlicestrided_slice_1:output:0strided_slice_40/stack:output:0!strided_slice_40/stack_1:output:0!strided_slice_40/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_40Y
mul_47/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_47/ys
mul_47Mulstrided_slice_40:output:0mul_47/y:output:0*
T0*#
_output_shapes
:?????????2
mul_47O
Sin_9Sin
mul_47:z:0*
T0*#
_output_shapes
:?????????2
Sin_9[
Mul_48Mul	Exp_9:y:0	Cos_9:y:0*
T0*#
_output_shapes
:?????????2
Mul_48[
Mul_49Mul	Exp_9:y:0	Sin_9:y:0*
T0*#
_output_shapes
:?????????2
Mul_49O
Neg_9Neg
Mul_49:z:0*
T0*#
_output_shapes
:?????????2
Neg_9?
stack_9Pack
Mul_48:z:0	Neg_9:y:0
Mul_49:z:0
Mul_48:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_9w
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_9/shape?
	Reshape_9Reshapestack_9:output:0Reshape_9/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_9?
strided_slice_41/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_41/stack?
strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_41/stack_1?
strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_41/stack_2?
strided_slice_41StridedSlicestrided_slice:output:0strided_slice_41/stack:output:0!strided_slice_41/stack_1:output:0!strided_slice_41/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_41?
einsum_9/EinsumEinsumstrided_slice_41:output:0Reshape_9:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_9/Einsum?
stack_10Packeinsum/Einsum:output:0einsum_1/Einsum:output:0einsum_2/Einsum:output:0einsum_3/Einsum:output:0einsum_4/Einsum:output:0einsum_5/Einsum:output:0einsum_6/Einsum:output:0einsum_7/Einsum:output:0einsum_8/Einsum:output:0einsum_9/Einsum:output:0*
N
*
T0*+
_output_shapes
:?????????
*

axis2

stack_10u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10Reshapestack_10:output:0Reshape_10/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_10|
stack_11PackReshape_10:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2

stack_11y
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_11/shape?

Reshape_11Reshapestack_11:output:0Reshape_11/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_11k
IdentityIdentityReshape_11:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????:?????????:P L
+
_output_shapes
:?????????

_user_specified_namex/0:PL
+
_output_shapes
:?????????

_user_specified_namex/1
?	
?
'__inference_model_4_layer_call_fn_36434

inputs
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_354332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_34858

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:??????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
'__inference_model_4_layer_call_fn_35442
input_3
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3dense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_354332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?	
?
#__inference_signature_wrapper_35543
input_3
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3dense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_347392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
F__inference_dense_4_layer_call_and_return_all_conditional_losses_36595

inputs
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_349082
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
GPU2*0J 8? *7
f2R0
.__inference_dense_4_activity_regularizer_349342
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
.__inference_dense_3_activity_regularizer_34752
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
?
?
!__inference__traced_restore_37042
file_prefix#
assignvariableop_dense_3_kernel#
assignvariableop_1_dense_3_bias%
!assignvariableop_2_dense_4_kernel#
assignvariableop_3_dense_4_bias

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
F__inference_dense_3_layer_call_and_return_all_conditional_losses_36515

inputs
dense_3_kernel
dense_3_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_kerneldense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_348112
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
GPU2*0J 8? *7
f2R0
.__inference_dense_3_activity_regularizer_348372
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_36520

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:??????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?Z
?
B__inference_model_4_layer_call_and_return_conditional_losses_35377
input_3
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_348112!
dense_3/StatefulPartitionedCall?
+dense_3/ActivityRegularizer/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_3_activity_regularizer_348372-
+dense_3/ActivityRegularizer/PartitionedCall?
!dense_3/ActivityRegularizer/ShapeShape(dense_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_3/ActivityRegularizer/Shape?
/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_3/ActivityRegularizer/strided_slice/stack?
1dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_1?
1dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_2?
)dense_3/ActivityRegularizer/strided_sliceStridedSlice*dense_3/ActivityRegularizer/Shape:output:08dense_3/ActivityRegularizer/strided_slice/stack:output:0:dense_3/ActivityRegularizer/strided_slice/stack_1:output:0:dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_3/ActivityRegularizer/strided_slice?
 dense_3/ActivityRegularizer/CastCast2dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_3/ActivityRegularizer/Cast?
#dense_3/ActivityRegularizer/truedivRealDiv4dense_3/ActivityRegularizer/PartitionedCall:output:0$dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_3/ActivityRegularizer/truediv?
activation_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_348582
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_349082!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_4_activity_regularizer_349342-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
linear_update/PartitionedCallPartitionedCallinput_3(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_352892
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
H
.__inference_dense_3_activity_regularizer_34837
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
??
?
 __inference__wrapped_model_34739
input_3;
7model_4_dense_3_tensordot_readvariableop_dense_3_kernel7
3model_4_dense_3_biasadd_readvariableop_dense_3_bias;
7model_4_dense_4_tensordot_readvariableop_dense_4_kernel7
3model_4_dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??&model_4/dense_3/BiasAdd/ReadVariableOp?(model_4/dense_3/Tensordot/ReadVariableOp?&model_4/dense_4/BiasAdd/ReadVariableOp?(model_4/dense_4/Tensordot/ReadVariableOp?
(model_4/dense_3/Tensordot/ReadVariableOpReadVariableOp7model_4_dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02*
(model_4/dense_3/Tensordot/ReadVariableOp?
model_4/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_4/dense_3/Tensordot/axes?
model_4/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_4/dense_3/Tensordot/freey
model_4/dense_3/Tensordot/ShapeShapeinput_3*
T0*
_output_shapes
:2!
model_4/dense_3/Tensordot/Shape?
'model_4/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_4/dense_3/Tensordot/GatherV2/axis?
"model_4/dense_3/Tensordot/GatherV2GatherV2(model_4/dense_3/Tensordot/Shape:output:0'model_4/dense_3/Tensordot/free:output:00model_4/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_4/dense_3/Tensordot/GatherV2?
)model_4/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/dense_3/Tensordot/GatherV2_1/axis?
$model_4/dense_3/Tensordot/GatherV2_1GatherV2(model_4/dense_3/Tensordot/Shape:output:0'model_4/dense_3/Tensordot/axes:output:02model_4/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_4/dense_3/Tensordot/GatherV2_1?
model_4/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_4/dense_3/Tensordot/Const?
model_4/dense_3/Tensordot/ProdProd+model_4/dense_3/Tensordot/GatherV2:output:0(model_4/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_4/dense_3/Tensordot/Prod?
!model_4/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_4/dense_3/Tensordot/Const_1?
 model_4/dense_3/Tensordot/Prod_1Prod-model_4/dense_3/Tensordot/GatherV2_1:output:0*model_4/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_4/dense_3/Tensordot/Prod_1?
%model_4/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_4/dense_3/Tensordot/concat/axis?
 model_4/dense_3/Tensordot/concatConcatV2'model_4/dense_3/Tensordot/free:output:0'model_4/dense_3/Tensordot/axes:output:0.model_4/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_4/dense_3/Tensordot/concat?
model_4/dense_3/Tensordot/stackPack'model_4/dense_3/Tensordot/Prod:output:0)model_4/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_4/dense_3/Tensordot/stack?
#model_4/dense_3/Tensordot/transpose	Transposeinput_3)model_4/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2%
#model_4/dense_3/Tensordot/transpose?
!model_4/dense_3/Tensordot/ReshapeReshape'model_4/dense_3/Tensordot/transpose:y:0(model_4/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_4/dense_3/Tensordot/Reshape?
 model_4/dense_3/Tensordot/MatMulMatMul*model_4/dense_3/Tensordot/Reshape:output:00model_4/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model_4/dense_3/Tensordot/MatMul?
!model_4/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2#
!model_4/dense_3/Tensordot/Const_2?
'model_4/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_4/dense_3/Tensordot/concat_1/axis?
"model_4/dense_3/Tensordot/concat_1ConcatV2+model_4/dense_3/Tensordot/GatherV2:output:0*model_4/dense_3/Tensordot/Const_2:output:00model_4/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_4/dense_3/Tensordot/concat_1?
model_4/dense_3/TensordotReshape*model_4/dense_3/Tensordot/MatMul:product:0+model_4/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
model_4/dense_3/Tensordot?
&model_4/dense_3/BiasAdd/ReadVariableOpReadVariableOp3model_4_dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype02(
&model_4/dense_3/BiasAdd/ReadVariableOp?
model_4/dense_3/BiasAddBiasAdd"model_4/dense_3/Tensordot:output:0.model_4/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model_4/dense_3/BiasAdd?
*model_4/dense_3/ActivityRegularizer/SquareSquare model_4/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2,
*model_4/dense_3/ActivityRegularizer/Square?
)model_4/dense_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_4/dense_3/ActivityRegularizer/Const?
'model_4/dense_3/ActivityRegularizer/SumSum.model_4/dense_3/ActivityRegularizer/Square:y:02model_4/dense_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_4/dense_3/ActivityRegularizer/Sum?
)model_4/dense_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_4/dense_3/ActivityRegularizer/mul/x?
'model_4/dense_3/ActivityRegularizer/mulMul2model_4/dense_3/ActivityRegularizer/mul/x:output:00model_4/dense_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_4/dense_3/ActivityRegularizer/mul?
)model_4/dense_3/ActivityRegularizer/ShapeShape model_4/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_4/dense_3/ActivityRegularizer/Shape?
7model_4/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_4/dense_3/ActivityRegularizer/strided_slice/stack?
9model_4/dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_4/dense_3/ActivityRegularizer/strided_slice/stack_1?
9model_4/dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_4/dense_3/ActivityRegularizer/strided_slice/stack_2?
1model_4/dense_3/ActivityRegularizer/strided_sliceStridedSlice2model_4/dense_3/ActivityRegularizer/Shape:output:0@model_4/dense_3/ActivityRegularizer/strided_slice/stack:output:0Bmodel_4/dense_3/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_4/dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_4/dense_3/ActivityRegularizer/strided_slice?
(model_4/dense_3/ActivityRegularizer/CastCast:model_4/dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_4/dense_3/ActivityRegularizer/Cast?
+model_4/dense_3/ActivityRegularizer/truedivRealDiv+model_4/dense_3/ActivityRegularizer/mul:z:0,model_4/dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_4/dense_3/ActivityRegularizer/truediv?
model_4/activation_2/ReluRelu model_4/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model_4/activation_2/Relu?
(model_4/dense_4/Tensordot/ReadVariableOpReadVariableOp7model_4_dense_4_tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype02*
(model_4/dense_4/Tensordot/ReadVariableOp?
model_4/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_4/dense_4/Tensordot/axes?
model_4/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_4/dense_4/Tensordot/free?
model_4/dense_4/Tensordot/ShapeShape'model_4/activation_2/Relu:activations:0*
T0*
_output_shapes
:2!
model_4/dense_4/Tensordot/Shape?
'model_4/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_4/dense_4/Tensordot/GatherV2/axis?
"model_4/dense_4/Tensordot/GatherV2GatherV2(model_4/dense_4/Tensordot/Shape:output:0'model_4/dense_4/Tensordot/free:output:00model_4/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_4/dense_4/Tensordot/GatherV2?
)model_4/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_4/dense_4/Tensordot/GatherV2_1/axis?
$model_4/dense_4/Tensordot/GatherV2_1GatherV2(model_4/dense_4/Tensordot/Shape:output:0'model_4/dense_4/Tensordot/axes:output:02model_4/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_4/dense_4/Tensordot/GatherV2_1?
model_4/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_4/dense_4/Tensordot/Const?
model_4/dense_4/Tensordot/ProdProd+model_4/dense_4/Tensordot/GatherV2:output:0(model_4/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_4/dense_4/Tensordot/Prod?
!model_4/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_4/dense_4/Tensordot/Const_1?
 model_4/dense_4/Tensordot/Prod_1Prod-model_4/dense_4/Tensordot/GatherV2_1:output:0*model_4/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_4/dense_4/Tensordot/Prod_1?
%model_4/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_4/dense_4/Tensordot/concat/axis?
 model_4/dense_4/Tensordot/concatConcatV2'model_4/dense_4/Tensordot/free:output:0'model_4/dense_4/Tensordot/axes:output:0.model_4/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_4/dense_4/Tensordot/concat?
model_4/dense_4/Tensordot/stackPack'model_4/dense_4/Tensordot/Prod:output:0)model_4/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_4/dense_4/Tensordot/stack?
#model_4/dense_4/Tensordot/transpose	Transpose'model_4/activation_2/Relu:activations:0)model_4/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2%
#model_4/dense_4/Tensordot/transpose?
!model_4/dense_4/Tensordot/ReshapeReshape'model_4/dense_4/Tensordot/transpose:y:0(model_4/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_4/dense_4/Tensordot/Reshape?
 model_4/dense_4/Tensordot/MatMulMatMul*model_4/dense_4/Tensordot/Reshape:output:00model_4/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 model_4/dense_4/Tensordot/MatMul?
!model_4/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model_4/dense_4/Tensordot/Const_2?
'model_4/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_4/dense_4/Tensordot/concat_1/axis?
"model_4/dense_4/Tensordot/concat_1ConcatV2+model_4/dense_4/Tensordot/GatherV2:output:0*model_4/dense_4/Tensordot/Const_2:output:00model_4/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_4/dense_4/Tensordot/concat_1?
model_4/dense_4/TensordotReshape*model_4/dense_4/Tensordot/MatMul:product:0+model_4/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
model_4/dense_4/Tensordot?
&model_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp3model_4_dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02(
&model_4/dense_4/BiasAdd/ReadVariableOp?
model_4/dense_4/BiasAddBiasAdd"model_4/dense_4/Tensordot:output:0.model_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model_4/dense_4/BiasAdd?
*model_4/dense_4/ActivityRegularizer/SquareSquare model_4/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2,
*model_4/dense_4/ActivityRegularizer/Square?
)model_4/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_4/dense_4/ActivityRegularizer/Const?
'model_4/dense_4/ActivityRegularizer/SumSum.model_4/dense_4/ActivityRegularizer/Square:y:02model_4/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_4/dense_4/ActivityRegularizer/Sum?
)model_4/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_4/dense_4/ActivityRegularizer/mul/x?
'model_4/dense_4/ActivityRegularizer/mulMul2model_4/dense_4/ActivityRegularizer/mul/x:output:00model_4/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_4/dense_4/ActivityRegularizer/mul?
)model_4/dense_4/ActivityRegularizer/ShapeShape model_4/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_4/dense_4/ActivityRegularizer/Shape?
7model_4/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_4/dense_4/ActivityRegularizer/strided_slice/stack?
9model_4/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_4/dense_4/ActivityRegularizer/strided_slice/stack_1?
9model_4/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_4/dense_4/ActivityRegularizer/strided_slice/stack_2?
1model_4/dense_4/ActivityRegularizer/strided_sliceStridedSlice2model_4/dense_4/ActivityRegularizer/Shape:output:0@model_4/dense_4/ActivityRegularizer/strided_slice/stack:output:0Bmodel_4/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_4/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_4/dense_4/ActivityRegularizer/strided_slice?
(model_4/dense_4/ActivityRegularizer/CastCast:model_4/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_4/dense_4/ActivityRegularizer/Cast?
+model_4/dense_4/ActivityRegularizer/truedivRealDiv+model_4/dense_4/ActivityRegularizer/mul:z:0,model_4/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_4/dense_4/ActivityRegularizer/truediv?
)model_4/linear_update/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)model_4/linear_update/strided_slice/stack?
+model_4/linear_update/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+model_4/linear_update/strided_slice/stack_1?
+model_4/linear_update/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+model_4/linear_update/strided_slice/stack_2?
#model_4/linear_update/strided_sliceStridedSliceinput_32model_4/linear_update/strided_slice/stack:output:04model_4/linear_update/strided_slice/stack_1:output:04model_4/linear_update/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#model_4/linear_update/strided_slice?
+model_4/linear_update/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+model_4/linear_update/strided_slice_1/stack?
-model_4/linear_update/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_1/stack_1?
-model_4/linear_update/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_1/stack_2?
%model_4/linear_update/strided_slice_1StridedSlice model_4/dense_4/BiasAdd:output:04model_4/linear_update/strided_slice_1/stack:output:06model_4/linear_update/strided_slice_1/stack_1:output:06model_4/linear_update/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_4/linear_update/strided_slice_1?
+model_4/linear_update/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+model_4/linear_update/strided_slice_2/stack?
-model_4/linear_update/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_2/stack_1?
-model_4/linear_update/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_2/stack_2?
%model_4/linear_update/strided_slice_2StridedSlice.model_4/linear_update/strided_slice_1:output:04model_4/linear_update/strided_slice_2/stack:output:06model_4/linear_update/strided_slice_2/stack_1:output:06model_4/linear_update/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_4/linear_update/strided_slice_2
model_4/linear_update/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_4/linear_update/mul/y?
model_4/linear_update/mulMul.model_4/linear_update/strided_slice_2:output:0$model_4/linear_update/mul/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul?
model_4/linear_update/ExpExpmodel_4/linear_update/mul:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp?
+model_4/linear_update/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2-
+model_4/linear_update/strided_slice_3/stack?
-model_4/linear_update/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_3/stack_1?
-model_4/linear_update/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_3/stack_2?
%model_4/linear_update/strided_slice_3StridedSlice.model_4/linear_update/strided_slice_1:output:04model_4/linear_update/strided_slice_3/stack:output:06model_4/linear_update/strided_slice_3/stack_1:output:06model_4/linear_update/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_4/linear_update/strided_slice_3?
model_4/linear_update/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_4/linear_update/mul_1/y?
model_4/linear_update/mul_1Mul.model_4/linear_update/strided_slice_3:output:0&model_4/linear_update/mul_1/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_1?
model_4/linear_update/CosCosmodel_4/linear_update/mul_1:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos?
+model_4/linear_update/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2-
+model_4/linear_update/strided_slice_4/stack?
-model_4/linear_update/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_4/stack_1?
-model_4/linear_update/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_4/stack_2?
%model_4/linear_update/strided_slice_4StridedSlice.model_4/linear_update/strided_slice_1:output:04model_4/linear_update/strided_slice_4/stack:output:06model_4/linear_update/strided_slice_4/stack_1:output:06model_4/linear_update/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_4/linear_update/strided_slice_4?
model_4/linear_update/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_4/linear_update/mul_2/y?
model_4/linear_update/mul_2Mul.model_4/linear_update/strided_slice_4:output:0&model_4/linear_update/mul_2/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_2?
model_4/linear_update/SinSinmodel_4/linear_update/mul_2:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin?
model_4/linear_update/Mul_3Mulmodel_4/linear_update/Exp:y:0model_4/linear_update/Cos:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_3?
model_4/linear_update/Mul_4Mulmodel_4/linear_update/Exp:y:0model_4/linear_update/Sin:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_4?
model_4/linear_update/NegNegmodel_4/linear_update/Mul_4:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg?
model_4/linear_update/stackPackmodel_4/linear_update/Mul_3:z:0model_4/linear_update/Neg:y:0model_4/linear_update/Mul_4:z:0model_4/linear_update/Mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack?
#model_4/linear_update/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2%
#model_4/linear_update/Reshape/shape?
model_4/linear_update/ReshapeReshape$model_4/linear_update/stack:output:0,model_4/linear_update/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
model_4/linear_update/Reshape?
+model_4/linear_update/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+model_4/linear_update/strided_slice_5/stack?
-model_4/linear_update/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_5/stack_1?
-model_4/linear_update/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_5/stack_2?
%model_4/linear_update/strided_slice_5StridedSlice,model_4/linear_update/strided_slice:output:04model_4/linear_update/strided_slice_5/stack:output:06model_4/linear_update/strided_slice_5/stack_1:output:06model_4/linear_update/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2'
%model_4/linear_update/strided_slice_5?
#model_4/linear_update/einsum/EinsumEinsum.model_4/linear_update/strided_slice_5:output:0&model_4/linear_update/Reshape:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2%
#model_4/linear_update/einsum/Einsum?
+model_4/linear_update/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+model_4/linear_update/strided_slice_6/stack?
-model_4/linear_update/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_6/stack_1?
-model_4/linear_update/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_6/stack_2?
%model_4/linear_update/strided_slice_6StridedSlice.model_4/linear_update/strided_slice_1:output:04model_4/linear_update/strided_slice_6/stack:output:06model_4/linear_update/strided_slice_6/stack_1:output:06model_4/linear_update/strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_4/linear_update/strided_slice_6?
model_4/linear_update/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_4/linear_update/mul_5/y?
model_4/linear_update/mul_5Mul.model_4/linear_update/strided_slice_6:output:0&model_4/linear_update/mul_5/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_5?
model_4/linear_update/Exp_1Expmodel_4/linear_update/mul_5:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_1?
+model_4/linear_update/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+model_4/linear_update/strided_slice_7/stack?
-model_4/linear_update/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_7/stack_1?
-model_4/linear_update/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_7/stack_2?
%model_4/linear_update/strided_slice_7StridedSlice.model_4/linear_update/strided_slice_1:output:04model_4/linear_update/strided_slice_7/stack:output:06model_4/linear_update/strided_slice_7/stack_1:output:06model_4/linear_update/strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_4/linear_update/strided_slice_7?
model_4/linear_update/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_4/linear_update/mul_6/y?
model_4/linear_update/mul_6Mul.model_4/linear_update/strided_slice_7:output:0&model_4/linear_update/mul_6/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_6?
model_4/linear_update/Cos_1Cosmodel_4/linear_update/mul_6:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_1?
+model_4/linear_update/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+model_4/linear_update/strided_slice_8/stack?
-model_4/linear_update/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_8/stack_1?
-model_4/linear_update/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_8/stack_2?
%model_4/linear_update/strided_slice_8StridedSlice.model_4/linear_update/strided_slice_1:output:04model_4/linear_update/strided_slice_8/stack:output:06model_4/linear_update/strided_slice_8/stack_1:output:06model_4/linear_update/strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_4/linear_update/strided_slice_8?
model_4/linear_update/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_4/linear_update/mul_7/y?
model_4/linear_update/mul_7Mul.model_4/linear_update/strided_slice_8:output:0&model_4/linear_update/mul_7/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_7?
model_4/linear_update/Sin_1Sinmodel_4/linear_update/mul_7:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_1?
model_4/linear_update/Mul_8Mulmodel_4/linear_update/Exp_1:y:0model_4/linear_update/Cos_1:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_8?
model_4/linear_update/Mul_9Mulmodel_4/linear_update/Exp_1:y:0model_4/linear_update/Sin_1:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_9?
model_4/linear_update/Neg_1Negmodel_4/linear_update/Mul_9:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_1?
model_4/linear_update/stack_1Packmodel_4/linear_update/Mul_8:z:0model_4/linear_update/Neg_1:y:0model_4/linear_update/Mul_9:z:0model_4/linear_update/Mul_8:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_1?
%model_4/linear_update/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_1/shape?
model_4/linear_update/Reshape_1Reshape&model_4/linear_update/stack_1:output:0.model_4/linear_update/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_1?
+model_4/linear_update/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+model_4/linear_update/strided_slice_9/stack?
-model_4/linear_update/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_4/linear_update/strided_slice_9/stack_1?
-model_4/linear_update/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_4/linear_update/strided_slice_9/stack_2?
%model_4/linear_update/strided_slice_9StridedSlice,model_4/linear_update/strided_slice:output:04model_4/linear_update/strided_slice_9/stack:output:06model_4/linear_update/strided_slice_9/stack_1:output:06model_4/linear_update/strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2'
%model_4/linear_update/strided_slice_9?
%model_4/linear_update/einsum_1/EinsumEinsum.model_4/linear_update/strided_slice_9:output:0(model_4/linear_update/Reshape_1:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_1/Einsum?
,model_4/linear_update/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_10/stack?
.model_4/linear_update/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_10/stack_1?
.model_4/linear_update/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_10/stack_2?
&model_4/linear_update/strided_slice_10StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_10/stack:output:07model_4/linear_update/strided_slice_10/stack_1:output:07model_4/linear_update/strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_10?
model_4/linear_update/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_10/y?
model_4/linear_update/mul_10Mul/model_4/linear_update/strided_slice_10:output:0'model_4/linear_update/mul_10/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_10?
model_4/linear_update/Exp_2Exp model_4/linear_update/mul_10:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_2?
,model_4/linear_update/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_11/stack?
.model_4/linear_update/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_11/stack_1?
.model_4/linear_update/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_11/stack_2?
&model_4/linear_update/strided_slice_11StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_11/stack:output:07model_4/linear_update/strided_slice_11/stack_1:output:07model_4/linear_update/strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_11?
model_4/linear_update/mul_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_11/y?
model_4/linear_update/mul_11Mul/model_4/linear_update/strided_slice_11:output:0'model_4/linear_update/mul_11/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_11?
model_4/linear_update/Cos_2Cos model_4/linear_update/mul_11:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_2?
,model_4/linear_update/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_12/stack?
.model_4/linear_update/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_12/stack_1?
.model_4/linear_update/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_12/stack_2?
&model_4/linear_update/strided_slice_12StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_12/stack:output:07model_4/linear_update/strided_slice_12/stack_1:output:07model_4/linear_update/strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_12?
model_4/linear_update/mul_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_12/y?
model_4/linear_update/mul_12Mul/model_4/linear_update/strided_slice_12:output:0'model_4/linear_update/mul_12/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_12?
model_4/linear_update/Sin_2Sin model_4/linear_update/mul_12:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_2?
model_4/linear_update/Mul_13Mulmodel_4/linear_update/Exp_2:y:0model_4/linear_update/Cos_2:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_13?
model_4/linear_update/Mul_14Mulmodel_4/linear_update/Exp_2:y:0model_4/linear_update/Sin_2:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_14?
model_4/linear_update/Neg_2Neg model_4/linear_update/Mul_14:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_2?
model_4/linear_update/stack_2Pack model_4/linear_update/Mul_13:z:0model_4/linear_update/Neg_2:y:0 model_4/linear_update/Mul_14:z:0 model_4/linear_update/Mul_13:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_2?
%model_4/linear_update/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_2/shape?
model_4/linear_update/Reshape_2Reshape&model_4/linear_update/stack_2:output:0.model_4/linear_update/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_2?
,model_4/linear_update/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_13/stack?
.model_4/linear_update/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_13/stack_1?
.model_4/linear_update/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_13/stack_2?
&model_4/linear_update/strided_slice_13StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_13/stack:output:07model_4/linear_update/strided_slice_13/stack_1:output:07model_4/linear_update/strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_13?
%model_4/linear_update/einsum_2/EinsumEinsum/model_4/linear_update/strided_slice_13:output:0(model_4/linear_update/Reshape_2:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_2/Einsum?
,model_4/linear_update/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_14/stack?
.model_4/linear_update/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_14/stack_1?
.model_4/linear_update/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_14/stack_2?
&model_4/linear_update/strided_slice_14StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_14/stack:output:07model_4/linear_update/strided_slice_14/stack_1:output:07model_4/linear_update/strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_14?
model_4/linear_update/mul_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_15/y?
model_4/linear_update/mul_15Mul/model_4/linear_update/strided_slice_14:output:0'model_4/linear_update/mul_15/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_15?
model_4/linear_update/Exp_3Exp model_4/linear_update/mul_15:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_3?
,model_4/linear_update/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_15/stack?
.model_4/linear_update/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_15/stack_1?
.model_4/linear_update/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_15/stack_2?
&model_4/linear_update/strided_slice_15StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_15/stack:output:07model_4/linear_update/strided_slice_15/stack_1:output:07model_4/linear_update/strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_15?
model_4/linear_update/mul_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_16/y?
model_4/linear_update/mul_16Mul/model_4/linear_update/strided_slice_15:output:0'model_4/linear_update/mul_16/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_16?
model_4/linear_update/Cos_3Cos model_4/linear_update/mul_16:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_3?
,model_4/linear_update/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_16/stack?
.model_4/linear_update/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_16/stack_1?
.model_4/linear_update/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_16/stack_2?
&model_4/linear_update/strided_slice_16StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_16/stack:output:07model_4/linear_update/strided_slice_16/stack_1:output:07model_4/linear_update/strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_16?
model_4/linear_update/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_17/y?
model_4/linear_update/mul_17Mul/model_4/linear_update/strided_slice_16:output:0'model_4/linear_update/mul_17/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_17?
model_4/linear_update/Sin_3Sin model_4/linear_update/mul_17:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_3?
model_4/linear_update/Mul_18Mulmodel_4/linear_update/Exp_3:y:0model_4/linear_update/Cos_3:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_18?
model_4/linear_update/Mul_19Mulmodel_4/linear_update/Exp_3:y:0model_4/linear_update/Sin_3:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_19?
model_4/linear_update/Neg_3Neg model_4/linear_update/Mul_19:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_3?
model_4/linear_update/stack_3Pack model_4/linear_update/Mul_18:z:0model_4/linear_update/Neg_3:y:0 model_4/linear_update/Mul_19:z:0 model_4/linear_update/Mul_18:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_3?
%model_4/linear_update/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_3/shape?
model_4/linear_update/Reshape_3Reshape&model_4/linear_update/stack_3:output:0.model_4/linear_update/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_3?
,model_4/linear_update/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_17/stack?
.model_4/linear_update/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_17/stack_1?
.model_4/linear_update/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_17/stack_2?
&model_4/linear_update/strided_slice_17StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_17/stack:output:07model_4/linear_update/strided_slice_17/stack_1:output:07model_4/linear_update/strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_17?
%model_4/linear_update/einsum_3/EinsumEinsum/model_4/linear_update/strided_slice_17:output:0(model_4/linear_update/Reshape_3:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_3/Einsum?
,model_4/linear_update/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_18/stack?
.model_4/linear_update/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_18/stack_1?
.model_4/linear_update/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_18/stack_2?
&model_4/linear_update/strided_slice_18StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_18/stack:output:07model_4/linear_update/strided_slice_18/stack_1:output:07model_4/linear_update/strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_18?
model_4/linear_update/mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_20/y?
model_4/linear_update/mul_20Mul/model_4/linear_update/strided_slice_18:output:0'model_4/linear_update/mul_20/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_20?
model_4/linear_update/Exp_4Exp model_4/linear_update/mul_20:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_4?
,model_4/linear_update/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_19/stack?
.model_4/linear_update/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_19/stack_1?
.model_4/linear_update/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_19/stack_2?
&model_4/linear_update/strided_slice_19StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_19/stack:output:07model_4/linear_update/strided_slice_19/stack_1:output:07model_4/linear_update/strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_19?
model_4/linear_update/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_21/y?
model_4/linear_update/mul_21Mul/model_4/linear_update/strided_slice_19:output:0'model_4/linear_update/mul_21/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_21?
model_4/linear_update/Cos_4Cos model_4/linear_update/mul_21:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_4?
,model_4/linear_update/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_20/stack?
.model_4/linear_update/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_20/stack_1?
.model_4/linear_update/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_20/stack_2?
&model_4/linear_update/strided_slice_20StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_20/stack:output:07model_4/linear_update/strided_slice_20/stack_1:output:07model_4/linear_update/strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_20?
model_4/linear_update/mul_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_22/y?
model_4/linear_update/mul_22Mul/model_4/linear_update/strided_slice_20:output:0'model_4/linear_update/mul_22/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_22?
model_4/linear_update/Sin_4Sin model_4/linear_update/mul_22:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_4?
model_4/linear_update/Mul_23Mulmodel_4/linear_update/Exp_4:y:0model_4/linear_update/Cos_4:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_23?
model_4/linear_update/Mul_24Mulmodel_4/linear_update/Exp_4:y:0model_4/linear_update/Sin_4:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_24?
model_4/linear_update/Neg_4Neg model_4/linear_update/Mul_24:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_4?
model_4/linear_update/stack_4Pack model_4/linear_update/Mul_23:z:0model_4/linear_update/Neg_4:y:0 model_4/linear_update/Mul_24:z:0 model_4/linear_update/Mul_23:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_4?
%model_4/linear_update/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_4/shape?
model_4/linear_update/Reshape_4Reshape&model_4/linear_update/stack_4:output:0.model_4/linear_update/Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_4?
,model_4/linear_update/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_21/stack?
.model_4/linear_update/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   20
.model_4/linear_update/strided_slice_21/stack_1?
.model_4/linear_update/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_21/stack_2?
&model_4/linear_update/strided_slice_21StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_21/stack:output:07model_4/linear_update/strided_slice_21/stack_1:output:07model_4/linear_update/strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_21?
%model_4/linear_update/einsum_4/EinsumEinsum/model_4/linear_update/strided_slice_21:output:0(model_4/linear_update/Reshape_4:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_4/Einsum?
,model_4/linear_update/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_22/stack?
.model_4/linear_update/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_22/stack_1?
.model_4/linear_update/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_22/stack_2?
&model_4/linear_update/strided_slice_22StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_22/stack:output:07model_4/linear_update/strided_slice_22/stack_1:output:07model_4/linear_update/strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_22?
model_4/linear_update/mul_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_25/y?
model_4/linear_update/mul_25Mul/model_4/linear_update/strided_slice_22:output:0'model_4/linear_update/mul_25/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_25?
model_4/linear_update/Exp_5Exp model_4/linear_update/mul_25:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_5?
,model_4/linear_update/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_23/stack?
.model_4/linear_update/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_23/stack_1?
.model_4/linear_update/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_23/stack_2?
&model_4/linear_update/strided_slice_23StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_23/stack:output:07model_4/linear_update/strided_slice_23/stack_1:output:07model_4/linear_update/strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_23?
model_4/linear_update/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_26/y?
model_4/linear_update/mul_26Mul/model_4/linear_update/strided_slice_23:output:0'model_4/linear_update/mul_26/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_26?
model_4/linear_update/Cos_5Cos model_4/linear_update/mul_26:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_5?
,model_4/linear_update/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_24/stack?
.model_4/linear_update/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_24/stack_1?
.model_4/linear_update/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_24/stack_2?
&model_4/linear_update/strided_slice_24StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_24/stack:output:07model_4/linear_update/strided_slice_24/stack_1:output:07model_4/linear_update/strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_24?
model_4/linear_update/mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_27/y?
model_4/linear_update/mul_27Mul/model_4/linear_update/strided_slice_24:output:0'model_4/linear_update/mul_27/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_27?
model_4/linear_update/Sin_5Sin model_4/linear_update/mul_27:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_5?
model_4/linear_update/Mul_28Mulmodel_4/linear_update/Exp_5:y:0model_4/linear_update/Cos_5:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_28?
model_4/linear_update/Mul_29Mulmodel_4/linear_update/Exp_5:y:0model_4/linear_update/Sin_5:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_29?
model_4/linear_update/Neg_5Neg model_4/linear_update/Mul_29:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_5?
model_4/linear_update/stack_5Pack model_4/linear_update/Mul_28:z:0model_4/linear_update/Neg_5:y:0 model_4/linear_update/Mul_29:z:0 model_4/linear_update/Mul_28:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_5?
%model_4/linear_update/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_5/shape?
model_4/linear_update/Reshape_5Reshape&model_4/linear_update/stack_5:output:0.model_4/linear_update/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_5?
,model_4/linear_update/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2.
,model_4/linear_update/strided_slice_25/stack?
.model_4/linear_update/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_25/stack_1?
.model_4/linear_update/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_25/stack_2?
&model_4/linear_update/strided_slice_25StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_25/stack:output:07model_4/linear_update/strided_slice_25/stack_1:output:07model_4/linear_update/strided_slice_25/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_25?
%model_4/linear_update/einsum_5/EinsumEinsum/model_4/linear_update/strided_slice_25:output:0(model_4/linear_update/Reshape_5:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_5/Einsum?
,model_4/linear_update/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_26/stack?
.model_4/linear_update/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_26/stack_1?
.model_4/linear_update/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_26/stack_2?
&model_4/linear_update/strided_slice_26StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_26/stack:output:07model_4/linear_update/strided_slice_26/stack_1:output:07model_4/linear_update/strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_26?
model_4/linear_update/mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_30/y?
model_4/linear_update/mul_30Mul/model_4/linear_update/strided_slice_26:output:0'model_4/linear_update/mul_30/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_30?
model_4/linear_update/Exp_6Exp model_4/linear_update/mul_30:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_6?
,model_4/linear_update/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_27/stack?
.model_4/linear_update/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_27/stack_1?
.model_4/linear_update/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_27/stack_2?
&model_4/linear_update/strided_slice_27StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_27/stack:output:07model_4/linear_update/strided_slice_27/stack_1:output:07model_4/linear_update/strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_27?
model_4/linear_update/mul_31/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_31/y?
model_4/linear_update/mul_31Mul/model_4/linear_update/strided_slice_27:output:0'model_4/linear_update/mul_31/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_31?
model_4/linear_update/Cos_6Cos model_4/linear_update/mul_31:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_6?
,model_4/linear_update/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_28/stack?
.model_4/linear_update/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_28/stack_1?
.model_4/linear_update/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_28/stack_2?
&model_4/linear_update/strided_slice_28StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_28/stack:output:07model_4/linear_update/strided_slice_28/stack_1:output:07model_4/linear_update/strided_slice_28/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_28?
model_4/linear_update/mul_32/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_32/y?
model_4/linear_update/mul_32Mul/model_4/linear_update/strided_slice_28:output:0'model_4/linear_update/mul_32/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_32?
model_4/linear_update/Sin_6Sin model_4/linear_update/mul_32:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_6?
model_4/linear_update/Mul_33Mulmodel_4/linear_update/Exp_6:y:0model_4/linear_update/Cos_6:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_33?
model_4/linear_update/Mul_34Mulmodel_4/linear_update/Exp_6:y:0model_4/linear_update/Sin_6:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_34?
model_4/linear_update/Neg_6Neg model_4/linear_update/Mul_34:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_6?
model_4/linear_update/stack_6Pack model_4/linear_update/Mul_33:z:0model_4/linear_update/Neg_6:y:0 model_4/linear_update/Mul_34:z:0 model_4/linear_update/Mul_33:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_6?
%model_4/linear_update/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_6/shape?
model_4/linear_update/Reshape_6Reshape&model_4/linear_update/stack_6:output:0.model_4/linear_update/Reshape_6/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_6?
,model_4/linear_update/strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_29/stack?
.model_4/linear_update/strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_29/stack_1?
.model_4/linear_update/strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_29/stack_2?
&model_4/linear_update/strided_slice_29StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_29/stack:output:07model_4/linear_update/strided_slice_29/stack_1:output:07model_4/linear_update/strided_slice_29/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_29?
%model_4/linear_update/einsum_6/EinsumEinsum/model_4/linear_update/strided_slice_29:output:0(model_4/linear_update/Reshape_6:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_6/Einsum?
,model_4/linear_update/strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_30/stack?
.model_4/linear_update/strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_30/stack_1?
.model_4/linear_update/strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_30/stack_2?
&model_4/linear_update/strided_slice_30StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_30/stack:output:07model_4/linear_update/strided_slice_30/stack_1:output:07model_4/linear_update/strided_slice_30/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_30?
model_4/linear_update/mul_35/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_35/y?
model_4/linear_update/mul_35Mul/model_4/linear_update/strided_slice_30:output:0'model_4/linear_update/mul_35/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_35?
model_4/linear_update/Exp_7Exp model_4/linear_update/mul_35:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_7?
,model_4/linear_update/strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_31/stack?
.model_4/linear_update/strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_31/stack_1?
.model_4/linear_update/strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_31/stack_2?
&model_4/linear_update/strided_slice_31StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_31/stack:output:07model_4/linear_update/strided_slice_31/stack_1:output:07model_4/linear_update/strided_slice_31/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_31?
model_4/linear_update/mul_36/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_36/y?
model_4/linear_update/mul_36Mul/model_4/linear_update/strided_slice_31:output:0'model_4/linear_update/mul_36/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_36?
model_4/linear_update/Cos_7Cos model_4/linear_update/mul_36:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_7?
,model_4/linear_update/strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_32/stack?
.model_4/linear_update/strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_32/stack_1?
.model_4/linear_update/strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_32/stack_2?
&model_4/linear_update/strided_slice_32StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_32/stack:output:07model_4/linear_update/strided_slice_32/stack_1:output:07model_4/linear_update/strided_slice_32/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_32?
model_4/linear_update/mul_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_37/y?
model_4/linear_update/mul_37Mul/model_4/linear_update/strided_slice_32:output:0'model_4/linear_update/mul_37/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_37?
model_4/linear_update/Sin_7Sin model_4/linear_update/mul_37:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_7?
model_4/linear_update/Mul_38Mulmodel_4/linear_update/Exp_7:y:0model_4/linear_update/Cos_7:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_38?
model_4/linear_update/Mul_39Mulmodel_4/linear_update/Exp_7:y:0model_4/linear_update/Sin_7:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_39?
model_4/linear_update/Neg_7Neg model_4/linear_update/Mul_39:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_7?
model_4/linear_update/stack_7Pack model_4/linear_update/Mul_38:z:0model_4/linear_update/Neg_7:y:0 model_4/linear_update/Mul_39:z:0 model_4/linear_update/Mul_38:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_7?
%model_4/linear_update/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_7/shape?
model_4/linear_update/Reshape_7Reshape&model_4/linear_update/stack_7:output:0.model_4/linear_update/Reshape_7/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_7?
,model_4/linear_update/strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_33/stack?
.model_4/linear_update/strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_33/stack_1?
.model_4/linear_update/strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_33/stack_2?
&model_4/linear_update/strided_slice_33StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_33/stack:output:07model_4/linear_update/strided_slice_33/stack_1:output:07model_4/linear_update/strided_slice_33/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_33?
%model_4/linear_update/einsum_7/EinsumEinsum/model_4/linear_update/strided_slice_33:output:0(model_4/linear_update/Reshape_7:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_7/Einsum?
,model_4/linear_update/strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_34/stack?
.model_4/linear_update/strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   20
.model_4/linear_update/strided_slice_34/stack_1?
.model_4/linear_update/strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_34/stack_2?
&model_4/linear_update/strided_slice_34StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_34/stack:output:07model_4/linear_update/strided_slice_34/stack_1:output:07model_4/linear_update/strided_slice_34/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_34?
model_4/linear_update/mul_40/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_40/y?
model_4/linear_update/mul_40Mul/model_4/linear_update/strided_slice_34:output:0'model_4/linear_update/mul_40/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_40?
model_4/linear_update/Exp_8Exp model_4/linear_update/mul_40:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_8?
,model_4/linear_update/strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_35/stack?
.model_4/linear_update/strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_35/stack_1?
.model_4/linear_update/strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_35/stack_2?
&model_4/linear_update/strided_slice_35StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_35/stack:output:07model_4/linear_update/strided_slice_35/stack_1:output:07model_4/linear_update/strided_slice_35/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_35?
model_4/linear_update/mul_41/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_41/y?
model_4/linear_update/mul_41Mul/model_4/linear_update/strided_slice_35:output:0'model_4/linear_update/mul_41/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_41?
model_4/linear_update/Cos_8Cos model_4/linear_update/mul_41:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_8?
,model_4/linear_update/strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_36/stack?
.model_4/linear_update/strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_36/stack_1?
.model_4/linear_update/strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_36/stack_2?
&model_4/linear_update/strided_slice_36StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_36/stack:output:07model_4/linear_update/strided_slice_36/stack_1:output:07model_4/linear_update/strided_slice_36/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_36?
model_4/linear_update/mul_42/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_42/y?
model_4/linear_update/mul_42Mul/model_4/linear_update/strided_slice_36:output:0'model_4/linear_update/mul_42/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_42?
model_4/linear_update/Sin_8Sin model_4/linear_update/mul_42:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_8?
model_4/linear_update/Mul_43Mulmodel_4/linear_update/Exp_8:y:0model_4/linear_update/Cos_8:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_43?
model_4/linear_update/Mul_44Mulmodel_4/linear_update/Exp_8:y:0model_4/linear_update/Sin_8:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_44?
model_4/linear_update/Neg_8Neg model_4/linear_update/Mul_44:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_8?
model_4/linear_update/stack_8Pack model_4/linear_update/Mul_43:z:0model_4/linear_update/Neg_8:y:0 model_4/linear_update/Mul_44:z:0 model_4/linear_update/Mul_43:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_8?
%model_4/linear_update/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_8/shape?
model_4/linear_update/Reshape_8Reshape&model_4/linear_update/stack_8:output:0.model_4/linear_update/Reshape_8/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_8?
,model_4/linear_update/strided_slice_37/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_37/stack?
.model_4/linear_update/strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_37/stack_1?
.model_4/linear_update/strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_37/stack_2?
&model_4/linear_update/strided_slice_37StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_37/stack:output:07model_4/linear_update/strided_slice_37/stack_1:output:07model_4/linear_update/strided_slice_37/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_37?
%model_4/linear_update/einsum_8/EinsumEinsum/model_4/linear_update/strided_slice_37:output:0(model_4/linear_update/Reshape_8:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_8/Einsum?
,model_4/linear_update/strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2.
,model_4/linear_update/strided_slice_38/stack?
.model_4/linear_update/strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   20
.model_4/linear_update/strided_slice_38/stack_1?
.model_4/linear_update/strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_38/stack_2?
&model_4/linear_update/strided_slice_38StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_38/stack:output:07model_4/linear_update/strided_slice_38/stack_1:output:07model_4/linear_update/strided_slice_38/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_38?
model_4/linear_update/mul_45/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_45/y?
model_4/linear_update/mul_45Mul/model_4/linear_update/strided_slice_38:output:0'model_4/linear_update/mul_45/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_45?
model_4/linear_update/Exp_9Exp model_4/linear_update/mul_45:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Exp_9?
,model_4/linear_update/strided_slice_39/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_39/stack?
.model_4/linear_update/strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_39/stack_1?
.model_4/linear_update/strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_39/stack_2?
&model_4/linear_update/strided_slice_39StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_39/stack:output:07model_4/linear_update/strided_slice_39/stack_1:output:07model_4/linear_update/strided_slice_39/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_39?
model_4/linear_update/mul_46/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_46/y?
model_4/linear_update/mul_46Mul/model_4/linear_update/strided_slice_39:output:0'model_4/linear_update/mul_46/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_46?
model_4/linear_update/Cos_9Cos model_4/linear_update/mul_46:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Cos_9?
,model_4/linear_update/strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_40/stack?
.model_4/linear_update/strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_40/stack_1?
.model_4/linear_update/strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_40/stack_2?
&model_4/linear_update/strided_slice_40StridedSlice.model_4/linear_update/strided_slice_1:output:05model_4/linear_update/strided_slice_40/stack:output:07model_4/linear_update/strided_slice_40/stack_1:output:07model_4/linear_update/strided_slice_40/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2(
&model_4/linear_update/strided_slice_40?
model_4/linear_update/mul_47/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2 
model_4/linear_update/mul_47/y?
model_4/linear_update/mul_47Mul/model_4/linear_update/strided_slice_40:output:0'model_4/linear_update/mul_47/y:output:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/mul_47?
model_4/linear_update/Sin_9Sin model_4/linear_update/mul_47:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Sin_9?
model_4/linear_update/Mul_48Mulmodel_4/linear_update/Exp_9:y:0model_4/linear_update/Cos_9:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_48?
model_4/linear_update/Mul_49Mulmodel_4/linear_update/Exp_9:y:0model_4/linear_update/Sin_9:y:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Mul_49?
model_4/linear_update/Neg_9Neg model_4/linear_update/Mul_49:z:0*
T0*#
_output_shapes
:?????????2
model_4/linear_update/Neg_9?
model_4/linear_update/stack_9Pack model_4/linear_update/Mul_48:z:0model_4/linear_update/Neg_9:y:0 model_4/linear_update/Mul_49:z:0 model_4/linear_update/Mul_48:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_9?
%model_4/linear_update/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_9/shape?
model_4/linear_update/Reshape_9Reshape&model_4/linear_update/stack_9:output:0.model_4/linear_update/Reshape_9/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_9?
,model_4/linear_update/strided_slice_41/stackConst*
_output_shapes
:*
dtype0*
valueB"       2.
,model_4/linear_update/strided_slice_41/stack?
.model_4/linear_update/strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model_4/linear_update/strided_slice_41/stack_1?
.model_4/linear_update/strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model_4/linear_update/strided_slice_41/stack_2?
&model_4/linear_update/strided_slice_41StridedSlice,model_4/linear_update/strided_slice:output:05model_4/linear_update/strided_slice_41/stack:output:07model_4/linear_update/strided_slice_41/stack_1:output:07model_4/linear_update/strided_slice_41/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&model_4/linear_update/strided_slice_41?
%model_4/linear_update/einsum_9/EinsumEinsum/model_4/linear_update/strided_slice_41:output:0(model_4/linear_update/Reshape_9:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2'
%model_4/linear_update/einsum_9/Einsum?
model_4/linear_update/stack_10Pack,model_4/linear_update/einsum/Einsum:output:0.model_4/linear_update/einsum_1/Einsum:output:0.model_4/linear_update/einsum_2/Einsum:output:0.model_4/linear_update/einsum_3/Einsum:output:0.model_4/linear_update/einsum_4/Einsum:output:0.model_4/linear_update/einsum_5/Einsum:output:0.model_4/linear_update/einsum_6/Einsum:output:0.model_4/linear_update/einsum_7/Einsum:output:0.model_4/linear_update/einsum_8/Einsum:output:0.model_4/linear_update/einsum_9/Einsum:output:0*
N
*
T0*+
_output_shapes
:?????????
*

axis2 
model_4/linear_update/stack_10?
&model_4/linear_update/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&model_4/linear_update/Reshape_10/shape?
 model_4/linear_update/Reshape_10Reshape'model_4/linear_update/stack_10:output:0/model_4/linear_update/Reshape_10/shape:output:0*
T0*'
_output_shapes
:?????????2"
 model_4/linear_update/Reshape_10?
model_4/linear_update/stack_11Pack)model_4/linear_update/Reshape_10:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2 
model_4/linear_update/stack_11?
&model_4/linear_update/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2(
&model_4/linear_update/Reshape_11/shape?
 model_4/linear_update/Reshape_11Reshape'model_4/linear_update/stack_11:output:0/model_4/linear_update/Reshape_11/shape:output:0*
T0*+
_output_shapes
:?????????2"
 model_4/linear_update/Reshape_11?
IdentityIdentity model_4/dense_4/BiasAdd:output:0'^model_4/dense_3/BiasAdd/ReadVariableOp)^model_4/dense_3/Tensordot/ReadVariableOp'^model_4/dense_4/BiasAdd/ReadVariableOp)^model_4/dense_4/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity)model_4/linear_update/Reshape_11:output:0'^model_4/dense_3/BiasAdd/ReadVariableOp)^model_4/dense_3/Tensordot/ReadVariableOp'^model_4/dense_4/BiasAdd/ReadVariableOp)^model_4/dense_4/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2P
&model_4/dense_3/BiasAdd/ReadVariableOp&model_4/dense_3/BiasAdd/ReadVariableOp2T
(model_4/dense_3/Tensordot/ReadVariableOp(model_4/dense_3/Tensordot/ReadVariableOp2P
&model_4/dense_4/BiasAdd/ReadVariableOp&model_4/dense_4/BiasAdd/ReadVariableOp2T
(model_4/dense_4/Tensordot/ReadVariableOp(model_4/dense_4/Tensordot/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
H
,__inference_activation_2_layer_call_fn_36525

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_348582
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
h
H__inference_linear_update_layer_call_and_return_conditional_losses_35289
x
x_1
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlicex_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2S
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
mul/yi
mulMulstrided_slice_2:output:0mul/y:output:0*
T0*#
_output_shapes
:?????????2
mulH
ExpExpmul:z:0*
T0*#
_output_shapes
:?????????2
Exp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3W
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_1/yo
mul_1Mulstrided_slice_3:output:0mul_1/y:output:0*
T0*#
_output_shapes
:?????????2
mul_1J
CosCos	mul_1:z:0*
T0*#
_output_shapes
:?????????2
Cos
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4W
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_2/yo
mul_2Mulstrided_slice_4:output:0mul_2/y:output:0*
T0*#
_output_shapes
:?????????2
mul_2J
SinSin	mul_2:z:0*
T0*#
_output_shapes
:?????????2
SinU
Mul_3MulExp:y:0Cos:y:0*
T0*#
_output_shapes
:?????????2
Mul_3U
Mul_4MulExp:y:0Sin:y:0*
T0*#
_output_shapes
:?????????2
Mul_4J
NegNeg	Mul_4:z:0*
T0*#
_output_shapes
:?????????2
Neg?
stackPack	Mul_3:z:0Neg:y:0	Mul_4:z:0	Mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
stacks
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shape{
ReshapeReshapestack:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshape
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_5?
einsum/EinsumEinsumstrided_slice_5:output:0Reshape:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum/Einsum
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6W
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_5/yo
mul_5Mulstrided_slice_6:output:0mul_5/y:output:0*
T0*#
_output_shapes
:?????????2
mul_5N
Exp_1Exp	mul_5:z:0*
T0*#
_output_shapes
:?????????2
Exp_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7W
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_6/yo
mul_6Mulstrided_slice_7:output:0mul_6/y:output:0*
T0*#
_output_shapes
:?????????2
mul_6N
Cos_1Cos	mul_6:z:0*
T0*#
_output_shapes
:?????????2
Cos_1
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8W
mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2	
mul_7/yo
mul_7Mulstrided_slice_8:output:0mul_7/y:output:0*
T0*#
_output_shapes
:?????????2
mul_7N
Sin_1Sin	mul_7:z:0*
T0*#
_output_shapes
:?????????2
Sin_1Y
Mul_8Mul	Exp_1:y:0	Cos_1:y:0*
T0*#
_output_shapes
:?????????2
Mul_8Y
Mul_9Mul	Exp_1:y:0	Sin_1:y:0*
T0*#
_output_shapes
:?????????2
Mul_9N
Neg_1Neg	Mul_9:z:0*
T0*#
_output_shapes
:?????????2
Neg_1?
stack_1Pack	Mul_8:z:0	Neg_1:y:0	Mul_9:z:0	Mul_8:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_1w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_1/shape?
	Reshape_1Reshapestack_1:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_1
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_9?
einsum_1/EinsumEinsumstrided_slice_9:output:0Reshape_1:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_1/Einsum?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10Y
mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_10/ys
mul_10Mulstrided_slice_10:output:0mul_10/y:output:0*
T0*#
_output_shapes
:?????????2
mul_10O
Exp_2Exp
mul_10:z:0*
T0*#
_output_shapes
:?????????2
Exp_2?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11Y
mul_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_11/ys
mul_11Mulstrided_slice_11:output:0mul_11/y:output:0*
T0*#
_output_shapes
:?????????2
mul_11O
Cos_2Cos
mul_11:z:0*
T0*#
_output_shapes
:?????????2
Cos_2?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12Y
mul_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_12/ys
mul_12Mulstrided_slice_12:output:0mul_12/y:output:0*
T0*#
_output_shapes
:?????????2
mul_12O
Sin_2Sin
mul_12:z:0*
T0*#
_output_shapes
:?????????2
Sin_2[
Mul_13Mul	Exp_2:y:0	Cos_2:y:0*
T0*#
_output_shapes
:?????????2
Mul_13[
Mul_14Mul	Exp_2:y:0	Sin_2:y:0*
T0*#
_output_shapes
:?????????2
Mul_14O
Neg_2Neg
Mul_14:z:0*
T0*#
_output_shapes
:?????????2
Neg_2?
stack_2Pack
Mul_13:z:0	Neg_2:y:0
Mul_14:z:0
Mul_13:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_2w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_2/shape?
	Reshape_2Reshapestack_2:output:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_2?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_13?
einsum_2/EinsumEinsumstrided_slice_13:output:0Reshape_2:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_2/Einsum?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14Y
mul_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_15/ys
mul_15Mulstrided_slice_14:output:0mul_15/y:output:0*
T0*#
_output_shapes
:?????????2
mul_15O
Exp_3Exp
mul_15:z:0*
T0*#
_output_shapes
:?????????2
Exp_3?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15Y
mul_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_16/ys
mul_16Mulstrided_slice_15:output:0mul_16/y:output:0*
T0*#
_output_shapes
:?????????2
mul_16O
Cos_3Cos
mul_16:z:0*
T0*#
_output_shapes
:?????????2
Cos_3?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16Y
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_17/ys
mul_17Mulstrided_slice_16:output:0mul_17/y:output:0*
T0*#
_output_shapes
:?????????2
mul_17O
Sin_3Sin
mul_17:z:0*
T0*#
_output_shapes
:?????????2
Sin_3[
Mul_18Mul	Exp_3:y:0	Cos_3:y:0*
T0*#
_output_shapes
:?????????2
Mul_18[
Mul_19Mul	Exp_3:y:0	Sin_3:y:0*
T0*#
_output_shapes
:?????????2
Mul_19O
Neg_3Neg
Mul_19:z:0*
T0*#
_output_shapes
:?????????2
Neg_3?
stack_3Pack
Mul_18:z:0	Neg_3:y:0
Mul_19:z:0
Mul_18:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_3w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_3/shape?
	Reshape_3Reshapestack_3:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_3?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_17?
einsum_3/EinsumEinsumstrided_slice_17:output:0Reshape_3:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_3/Einsum?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18Y
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_20/ys
mul_20Mulstrided_slice_18:output:0mul_20/y:output:0*
T0*#
_output_shapes
:?????????2
mul_20O
Exp_4Exp
mul_20:z:0*
T0*#
_output_shapes
:?????????2
Exp_4?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19Y
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_21/ys
mul_21Mulstrided_slice_19:output:0mul_21/y:output:0*
T0*#
_output_shapes
:?????????2
mul_21O
Cos_4Cos
mul_21:z:0*
T0*#
_output_shapes
:?????????2
Cos_4?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20Y
mul_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_22/ys
mul_22Mulstrided_slice_20:output:0mul_22/y:output:0*
T0*#
_output_shapes
:?????????2
mul_22O
Sin_4Sin
mul_22:z:0*
T0*#
_output_shapes
:?????????2
Sin_4[
Mul_23Mul	Exp_4:y:0	Cos_4:y:0*
T0*#
_output_shapes
:?????????2
Mul_23[
Mul_24Mul	Exp_4:y:0	Sin_4:y:0*
T0*#
_output_shapes
:?????????2
Mul_24O
Neg_4Neg
Mul_24:z:0*
T0*#
_output_shapes
:?????????2
Neg_4?
stack_4Pack
Mul_23:z:0	Neg_4:y:0
Mul_24:z:0
Mul_23:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_4w
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_4/shape?
	Reshape_4Reshapestack_4:output:0Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_4?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_21?
einsum_4/EinsumEinsumstrided_slice_21:output:0Reshape_4:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_4/Einsum?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22Y
mul_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_25/ys
mul_25Mulstrided_slice_22:output:0mul_25/y:output:0*
T0*#
_output_shapes
:?????????2
mul_25O
Exp_5Exp
mul_25:z:0*
T0*#
_output_shapes
:?????????2
Exp_5?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23Y
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_26/ys
mul_26Mulstrided_slice_23:output:0mul_26/y:output:0*
T0*#
_output_shapes
:?????????2
mul_26O
Cos_5Cos
mul_26:z:0*
T0*#
_output_shapes
:?????????2
Cos_5?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24Y
mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_27/ys
mul_27Mulstrided_slice_24:output:0mul_27/y:output:0*
T0*#
_output_shapes
:?????????2
mul_27O
Sin_5Sin
mul_27:z:0*
T0*#
_output_shapes
:?????????2
Sin_5[
Mul_28Mul	Exp_5:y:0	Cos_5:y:0*
T0*#
_output_shapes
:?????????2
Mul_28[
Mul_29Mul	Exp_5:y:0	Sin_5:y:0*
T0*#
_output_shapes
:?????????2
Mul_29O
Neg_5Neg
Mul_29:z:0*
T0*#
_output_shapes
:?????????2
Neg_5?
stack_5Pack
Mul_28:z:0	Neg_5:y:0
Mul_29:z:0
Mul_28:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_5w
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_5/shape?
	Reshape_5Reshapestack_5:output:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_5?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_25?
einsum_5/EinsumEinsumstrided_slice_25:output:0Reshape_5:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_5/Einsum?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26Y
mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_30/ys
mul_30Mulstrided_slice_26:output:0mul_30/y:output:0*
T0*#
_output_shapes
:?????????2
mul_30O
Exp_6Exp
mul_30:z:0*
T0*#
_output_shapes
:?????????2
Exp_6?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27Y
mul_31/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_31/ys
mul_31Mulstrided_slice_27:output:0mul_31/y:output:0*
T0*#
_output_shapes
:?????????2
mul_31O
Cos_6Cos
mul_31:z:0*
T0*#
_output_shapes
:?????????2
Cos_6?
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_28/stack?
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_28/stack_1?
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_28/stack_2?
strided_slice_28StridedSlicestrided_slice_1:output:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_28Y
mul_32/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_32/ys
mul_32Mulstrided_slice_28:output:0mul_32/y:output:0*
T0*#
_output_shapes
:?????????2
mul_32O
Sin_6Sin
mul_32:z:0*
T0*#
_output_shapes
:?????????2
Sin_6[
Mul_33Mul	Exp_6:y:0	Cos_6:y:0*
T0*#
_output_shapes
:?????????2
Mul_33[
Mul_34Mul	Exp_6:y:0	Sin_6:y:0*
T0*#
_output_shapes
:?????????2
Mul_34O
Neg_6Neg
Mul_34:z:0*
T0*#
_output_shapes
:?????????2
Neg_6?
stack_6Pack
Mul_33:z:0	Neg_6:y:0
Mul_34:z:0
Mul_33:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_6w
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_6/shape?
	Reshape_6Reshapestack_6:output:0Reshape_6/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_6?
strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_29/stack?
strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_29/stack_1?
strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_29/stack_2?
strided_slice_29StridedSlicestrided_slice:output:0strided_slice_29/stack:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_29?
einsum_6/EinsumEinsumstrided_slice_29:output:0Reshape_6:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_6/Einsum?
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_30/stack?
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_30/stack_1?
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_30/stack_2?
strided_slice_30StridedSlicestrided_slice_1:output:0strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_30Y
mul_35/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_35/ys
mul_35Mulstrided_slice_30:output:0mul_35/y:output:0*
T0*#
_output_shapes
:?????????2
mul_35O
Exp_7Exp
mul_35:z:0*
T0*#
_output_shapes
:?????????2
Exp_7?
strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_31/stack?
strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_31/stack_1?
strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_31/stack_2?
strided_slice_31StridedSlicestrided_slice_1:output:0strided_slice_31/stack:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_31Y
mul_36/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_36/ys
mul_36Mulstrided_slice_31:output:0mul_36/y:output:0*
T0*#
_output_shapes
:?????????2
mul_36O
Cos_7Cos
mul_36:z:0*
T0*#
_output_shapes
:?????????2
Cos_7?
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_32/stack?
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_32/stack_1?
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_32/stack_2?
strided_slice_32StridedSlicestrided_slice_1:output:0strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_32Y
mul_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_37/ys
mul_37Mulstrided_slice_32:output:0mul_37/y:output:0*
T0*#
_output_shapes
:?????????2
mul_37O
Sin_7Sin
mul_37:z:0*
T0*#
_output_shapes
:?????????2
Sin_7[
Mul_38Mul	Exp_7:y:0	Cos_7:y:0*
T0*#
_output_shapes
:?????????2
Mul_38[
Mul_39Mul	Exp_7:y:0	Sin_7:y:0*
T0*#
_output_shapes
:?????????2
Mul_39O
Neg_7Neg
Mul_39:z:0*
T0*#
_output_shapes
:?????????2
Neg_7?
stack_7Pack
Mul_38:z:0	Neg_7:y:0
Mul_39:z:0
Mul_38:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_7w
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_7/shape?
	Reshape_7Reshapestack_7:output:0Reshape_7/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_7?
strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_33/stack?
strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_33/stack_1?
strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_33/stack_2?
strided_slice_33StridedSlicestrided_slice:output:0strided_slice_33/stack:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_33?
einsum_7/EinsumEinsumstrided_slice_33:output:0Reshape_7:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_7/Einsum?
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_34/stack?
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_34/stack_1?
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_34/stack_2?
strided_slice_34StridedSlicestrided_slice_1:output:0strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_34Y
mul_40/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_40/ys
mul_40Mulstrided_slice_34:output:0mul_40/y:output:0*
T0*#
_output_shapes
:?????????2
mul_40O
Exp_8Exp
mul_40:z:0*
T0*#
_output_shapes
:?????????2
Exp_8?
strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_35/stack?
strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_35/stack_1?
strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_35/stack_2?
strided_slice_35StridedSlicestrided_slice_1:output:0strided_slice_35/stack:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_35Y
mul_41/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_41/ys
mul_41Mulstrided_slice_35:output:0mul_41/y:output:0*
T0*#
_output_shapes
:?????????2
mul_41O
Cos_8Cos
mul_41:z:0*
T0*#
_output_shapes
:?????????2
Cos_8?
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_36/stack?
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_36/stack_1?
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_36/stack_2?
strided_slice_36StridedSlicestrided_slice_1:output:0strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_36Y
mul_42/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_42/ys
mul_42Mulstrided_slice_36:output:0mul_42/y:output:0*
T0*#
_output_shapes
:?????????2
mul_42O
Sin_8Sin
mul_42:z:0*
T0*#
_output_shapes
:?????????2
Sin_8[
Mul_43Mul	Exp_8:y:0	Cos_8:y:0*
T0*#
_output_shapes
:?????????2
Mul_43[
Mul_44Mul	Exp_8:y:0	Sin_8:y:0*
T0*#
_output_shapes
:?????????2
Mul_44O
Neg_8Neg
Mul_44:z:0*
T0*#
_output_shapes
:?????????2
Neg_8?
stack_8Pack
Mul_43:z:0	Neg_8:y:0
Mul_44:z:0
Mul_43:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_8w
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_8/shape?
	Reshape_8Reshapestack_8:output:0Reshape_8/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_8?
strided_slice_37/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_37/stack?
strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_37/stack_1?
strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_37/stack_2?
strided_slice_37StridedSlicestrided_slice:output:0strided_slice_37/stack:output:0!strided_slice_37/stack_1:output:0!strided_slice_37/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_37?
einsum_8/EinsumEinsumstrided_slice_37:output:0Reshape_8:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_8/Einsum?
strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_38/stack?
strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_38/stack_1?
strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_38/stack_2?
strided_slice_38StridedSlicestrided_slice_1:output:0strided_slice_38/stack:output:0!strided_slice_38/stack_1:output:0!strided_slice_38/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_38Y
mul_45/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_45/ys
mul_45Mulstrided_slice_38:output:0mul_45/y:output:0*
T0*#
_output_shapes
:?????????2
mul_45O
Exp_9Exp
mul_45:z:0*
T0*#
_output_shapes
:?????????2
Exp_9?
strided_slice_39/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_39/stack?
strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_39/stack_1?
strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_39/stack_2?
strided_slice_39StridedSlicestrided_slice_1:output:0strided_slice_39/stack:output:0!strided_slice_39/stack_1:output:0!strided_slice_39/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_39Y
mul_46/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_46/ys
mul_46Mulstrided_slice_39:output:0mul_46/y:output:0*
T0*#
_output_shapes
:?????????2
mul_46O
Cos_9Cos
mul_46:z:0*
T0*#
_output_shapes
:?????????2
Cos_9?
strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_40/stack?
strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_40/stack_1?
strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_40/stack_2?
strided_slice_40StridedSlicestrided_slice_1:output:0strided_slice_40/stack:output:0!strided_slice_40/stack_1:output:0!strided_slice_40/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_40Y
mul_47/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2

mul_47/ys
mul_47Mulstrided_slice_40:output:0mul_47/y:output:0*
T0*#
_output_shapes
:?????????2
mul_47O
Sin_9Sin
mul_47:z:0*
T0*#
_output_shapes
:?????????2
Sin_9[
Mul_48Mul	Exp_9:y:0	Cos_9:y:0*
T0*#
_output_shapes
:?????????2
Mul_48[
Mul_49Mul	Exp_9:y:0	Sin_9:y:0*
T0*#
_output_shapes
:?????????2
Mul_49O
Neg_9Neg
Mul_49:z:0*
T0*#
_output_shapes
:?????????2
Neg_9?
stack_9Pack
Mul_48:z:0	Neg_9:y:0
Mul_49:z:0
Mul_48:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2	
stack_9w
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_9/shape?
	Reshape_9Reshapestack_9:output:0Reshape_9/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_9?
strided_slice_41/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_41/stack?
strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_41/stack_1?
strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_41/stack_2?
strided_slice_41StridedSlicestrided_slice:output:0strided_slice_41/stack:output:0!strided_slice_41/stack_1:output:0!strided_slice_41/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_41?
einsum_9/EinsumEinsumstrided_slice_41:output:0Reshape_9:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
einsum_9/Einsum?
stack_10Packeinsum/Einsum:output:0einsum_1/Einsum:output:0einsum_2/Einsum:output:0einsum_3/Einsum:output:0einsum_4/Einsum:output:0einsum_5/Einsum:output:0einsum_6/Einsum:output:0einsum_7/Einsum:output:0einsum_8/Einsum:output:0einsum_9/Einsum:output:0*
N
*
T0*+
_output_shapes
:?????????
*

axis2

stack_10u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10Reshapestack_10:output:0Reshape_10/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_10|
stack_11PackReshape_10:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2

stack_11y
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_11/shape?

Reshape_11Reshapestack_11:output:0Reshape_11/shape:output:0*
T0*+
_output_shapes
:?????????2

Reshape_11k
IdentityIdentityReshape_11:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????:?????????:N J
+
_output_shapes
:?????????

_user_specified_namex:NJ
+
_output_shapes
:?????????

_user_specified_namex
?
?
__inference__traced_save_37020
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
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

identity_1Identity_1:output:0*:
_input_shapes)
': :	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
??
?
B__inference_model_4_layer_call_and_return_conditional_losses_35983

inputs3
/dense_3_tensordot_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias3
/dense_4_tensordot_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
 dense_3/Tensordot/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes?
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/freeh
dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape?
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2?
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod?
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1?
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack?
dense_3/Tensordot/transpose	Transposeinputs!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_3/Tensordot/transpose?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_3/Tensordot/Reshape?
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/Tensordot/MatMul?
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_3/Tensordot/Const_2?
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_3/Tensordot?
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_3/BiasAdd?
"dense_3/ActivityRegularizer/SquareSquaredense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2$
"dense_3/ActivityRegularizer/Square?
!dense_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_3/ActivityRegularizer/Const?
dense_3/ActivityRegularizer/SumSum&dense_3/ActivityRegularizer/Square:y:0*dense_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_3/ActivityRegularizer/Sum?
!dense_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_3/ActivityRegularizer/mul/x?
dense_3/ActivityRegularizer/mulMul*dense_3/ActivityRegularizer/mul/x:output:0(dense_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_3/ActivityRegularizer/mul?
!dense_3/ActivityRegularizer/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_3/ActivityRegularizer/Shape?
/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_3/ActivityRegularizer/strided_slice/stack?
1dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_1?
1dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_2?
)dense_3/ActivityRegularizer/strided_sliceStridedSlice*dense_3/ActivityRegularizer/Shape:output:08dense_3/ActivityRegularizer/strided_slice/stack:output:0:dense_3/ActivityRegularizer/strided_slice/stack_1:output:0:dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_3/ActivityRegularizer/strided_slice?
 dense_3/ActivityRegularizer/CastCast2dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_3/ActivityRegularizer/Cast?
#dense_3/ActivityRegularizer/truedivRealDiv#dense_3/ActivityRegularizer/mul:z:0$dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_3/ActivityRegularizer/truediv
activation_2/ReluReludense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
activation_2/Relu?
 dense_4/Tensordot/ReadVariableOpReadVariableOp/dense_4_tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes?
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/free?
dense_4/Tensordot/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape?
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axis?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2?
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod?
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1?
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axis?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack?
dense_4/Tensordot/transpose	Transposeactivation_2/Relu:activations:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_4/Tensordot/transpose?
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_4/Tensordot/Reshape?
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/Tensordot/MatMul?
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/Const_2?
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axis?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_4/Tensordot?
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_4/BiasAdd?
"dense_4/ActivityRegularizer/SquareSquaredense_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
"dense_4/ActivityRegularizer/Square?
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_4/ActivityRegularizer/Const?
dense_4/ActivityRegularizer/SumSum&dense_4/ActivityRegularizer/Square:y:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_4/ActivityRegularizer/Sum?
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_4/ActivityRegularizer/mul/x?
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_4/ActivityRegularizer/mul?
!dense_4/ActivityRegularizer/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv#dense_4/ActivityRegularizer/mul:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
!linear_update/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!linear_update/strided_slice/stack?
#linear_update/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice/stack_1?
#linear_update/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#linear_update/strided_slice/stack_2?
linear_update/strided_sliceStridedSliceinputs*linear_update/strided_slice/stack:output:0,linear_update/strided_slice/stack_1:output:0,linear_update/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice?
#linear_update/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#linear_update/strided_slice_1/stack?
%linear_update/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_1/stack_1?
%linear_update/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_1/stack_2?
linear_update/strided_slice_1StridedSlicedense_4/BiasAdd:output:0,linear_update/strided_slice_1/stack:output:0.linear_update/strided_slice_1/stack_1:output:0.linear_update/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_1?
#linear_update/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#linear_update/strided_slice_2/stack?
%linear_update/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_2/stack_1?
%linear_update/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_2/stack_2?
linear_update/strided_slice_2StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_2/stack:output:0.linear_update/strided_slice_2/stack_1:output:0.linear_update/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_2o
linear_update/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul/y?
linear_update/mulMul&linear_update/strided_slice_2:output:0linear_update/mul/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mulr
linear_update/ExpExplinear_update/mul:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp?
#linear_update/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2%
#linear_update/strided_slice_3/stack?
%linear_update/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_3/stack_1?
%linear_update/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_3/stack_2?
linear_update/strided_slice_3StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_3/stack:output:0.linear_update/strided_slice_3/stack_1:output:0.linear_update/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_3s
linear_update/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_1/y?
linear_update/mul_1Mul&linear_update/strided_slice_3:output:0linear_update/mul_1/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_1t
linear_update/CosCoslinear_update/mul_1:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos?
#linear_update/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2%
#linear_update/strided_slice_4/stack?
%linear_update/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_4/stack_1?
%linear_update/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_4/stack_2?
linear_update/strided_slice_4StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_4/stack:output:0.linear_update/strided_slice_4/stack_1:output:0.linear_update/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_4s
linear_update/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_2/y?
linear_update/mul_2Mul&linear_update/strided_slice_4:output:0linear_update/mul_2/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_2t
linear_update/SinSinlinear_update/mul_2:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin?
linear_update/Mul_3Mullinear_update/Exp:y:0linear_update/Cos:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_3?
linear_update/Mul_4Mullinear_update/Exp:y:0linear_update/Sin:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_4t
linear_update/NegNeglinear_update/Mul_4:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg?
linear_update/stackPacklinear_update/Mul_3:z:0linear_update/Neg:y:0linear_update/Mul_4:z:0linear_update/Mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack?
linear_update/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape/shape?
linear_update/ReshapeReshapelinear_update/stack:output:0$linear_update/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape?
#linear_update/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#linear_update/strided_slice_5/stack?
%linear_update/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_5/stack_1?
%linear_update/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_5/stack_2?
linear_update/strided_slice_5StridedSlice$linear_update/strided_slice:output:0,linear_update/strided_slice_5/stack:output:0.linear_update/strided_slice_5/stack_1:output:0.linear_update/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
linear_update/strided_slice_5?
linear_update/einsum/EinsumEinsum&linear_update/strided_slice_5:output:0linear_update/Reshape:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum/Einsum?
#linear_update/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_6/stack?
%linear_update/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_6/stack_1?
%linear_update/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_6/stack_2?
linear_update/strided_slice_6StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_6/stack:output:0.linear_update/strided_slice_6/stack_1:output:0.linear_update/strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_6s
linear_update/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_5/y?
linear_update/mul_5Mul&linear_update/strided_slice_6:output:0linear_update/mul_5/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_5x
linear_update/Exp_1Explinear_update/mul_5:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_1?
#linear_update/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_7/stack?
%linear_update/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_7/stack_1?
%linear_update/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_7/stack_2?
linear_update/strided_slice_7StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_7/stack:output:0.linear_update/strided_slice_7/stack_1:output:0.linear_update/strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_7s
linear_update/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_6/y?
linear_update/mul_6Mul&linear_update/strided_slice_7:output:0linear_update/mul_6/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_6x
linear_update/Cos_1Coslinear_update/mul_6:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_1?
#linear_update/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_8/stack?
%linear_update/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_8/stack_1?
%linear_update/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_8/stack_2?
linear_update/strided_slice_8StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_8/stack:output:0.linear_update/strided_slice_8/stack_1:output:0.linear_update/strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_8s
linear_update/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_7/y?
linear_update/mul_7Mul&linear_update/strided_slice_8:output:0linear_update/mul_7/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_7x
linear_update/Sin_1Sinlinear_update/mul_7:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_1?
linear_update/Mul_8Mullinear_update/Exp_1:y:0linear_update/Cos_1:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_8?
linear_update/Mul_9Mullinear_update/Exp_1:y:0linear_update/Sin_1:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_9x
linear_update/Neg_1Neglinear_update/Mul_9:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_1?
linear_update/stack_1Packlinear_update/Mul_8:z:0linear_update/Neg_1:y:0linear_update/Mul_9:z:0linear_update/Mul_8:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_1?
linear_update/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_1/shape?
linear_update/Reshape_1Reshapelinear_update/stack_1:output:0&linear_update/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_1?
#linear_update/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_9/stack?
%linear_update/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_9/stack_1?
%linear_update/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_9/stack_2?
linear_update/strided_slice_9StridedSlice$linear_update/strided_slice:output:0,linear_update/strided_slice_9/stack:output:0.linear_update/strided_slice_9/stack_1:output:0.linear_update/strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
linear_update/strided_slice_9?
linear_update/einsum_1/EinsumEinsum&linear_update/strided_slice_9:output:0 linear_update/Reshape_1:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_1/Einsum?
$linear_update/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_10/stack?
&linear_update/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_10/stack_1?
&linear_update/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_10/stack_2?
linear_update/strided_slice_10StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_10/stack:output:0/linear_update/strided_slice_10/stack_1:output:0/linear_update/strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_10u
linear_update/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_10/y?
linear_update/mul_10Mul'linear_update/strided_slice_10:output:0linear_update/mul_10/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_10y
linear_update/Exp_2Explinear_update/mul_10:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_2?
$linear_update/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_11/stack?
&linear_update/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_11/stack_1?
&linear_update/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_11/stack_2?
linear_update/strided_slice_11StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_11/stack:output:0/linear_update/strided_slice_11/stack_1:output:0/linear_update/strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_11u
linear_update/mul_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_11/y?
linear_update/mul_11Mul'linear_update/strided_slice_11:output:0linear_update/mul_11/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_11y
linear_update/Cos_2Coslinear_update/mul_11:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_2?
$linear_update/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_12/stack?
&linear_update/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_12/stack_1?
&linear_update/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_12/stack_2?
linear_update/strided_slice_12StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_12/stack:output:0/linear_update/strided_slice_12/stack_1:output:0/linear_update/strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_12u
linear_update/mul_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_12/y?
linear_update/mul_12Mul'linear_update/strided_slice_12:output:0linear_update/mul_12/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_12y
linear_update/Sin_2Sinlinear_update/mul_12:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_2?
linear_update/Mul_13Mullinear_update/Exp_2:y:0linear_update/Cos_2:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_13?
linear_update/Mul_14Mullinear_update/Exp_2:y:0linear_update/Sin_2:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_14y
linear_update/Neg_2Neglinear_update/Mul_14:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_2?
linear_update/stack_2Packlinear_update/Mul_13:z:0linear_update/Neg_2:y:0linear_update/Mul_14:z:0linear_update/Mul_13:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_2?
linear_update/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_2/shape?
linear_update/Reshape_2Reshapelinear_update/stack_2:output:0&linear_update/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_2?
$linear_update/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_13/stack?
&linear_update/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_13/stack_1?
&linear_update/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_13/stack_2?
linear_update/strided_slice_13StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_13/stack:output:0/linear_update/strided_slice_13/stack_1:output:0/linear_update/strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_13?
linear_update/einsum_2/EinsumEinsum'linear_update/strided_slice_13:output:0 linear_update/Reshape_2:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_2/Einsum?
$linear_update/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_14/stack?
&linear_update/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_14/stack_1?
&linear_update/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_14/stack_2?
linear_update/strided_slice_14StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_14/stack:output:0/linear_update/strided_slice_14/stack_1:output:0/linear_update/strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_14u
linear_update/mul_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_15/y?
linear_update/mul_15Mul'linear_update/strided_slice_14:output:0linear_update/mul_15/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_15y
linear_update/Exp_3Explinear_update/mul_15:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_3?
$linear_update/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_15/stack?
&linear_update/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_15/stack_1?
&linear_update/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_15/stack_2?
linear_update/strided_slice_15StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_15/stack:output:0/linear_update/strided_slice_15/stack_1:output:0/linear_update/strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_15u
linear_update/mul_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_16/y?
linear_update/mul_16Mul'linear_update/strided_slice_15:output:0linear_update/mul_16/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_16y
linear_update/Cos_3Coslinear_update/mul_16:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_3?
$linear_update/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_16/stack?
&linear_update/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_16/stack_1?
&linear_update/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_16/stack_2?
linear_update/strided_slice_16StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_16/stack:output:0/linear_update/strided_slice_16/stack_1:output:0/linear_update/strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_16u
linear_update/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_17/y?
linear_update/mul_17Mul'linear_update/strided_slice_16:output:0linear_update/mul_17/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_17y
linear_update/Sin_3Sinlinear_update/mul_17:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_3?
linear_update/Mul_18Mullinear_update/Exp_3:y:0linear_update/Cos_3:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_18?
linear_update/Mul_19Mullinear_update/Exp_3:y:0linear_update/Sin_3:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_19y
linear_update/Neg_3Neglinear_update/Mul_19:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_3?
linear_update/stack_3Packlinear_update/Mul_18:z:0linear_update/Neg_3:y:0linear_update/Mul_19:z:0linear_update/Mul_18:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_3?
linear_update/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_3/shape?
linear_update/Reshape_3Reshapelinear_update/stack_3:output:0&linear_update/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_3?
$linear_update/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_17/stack?
&linear_update/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_17/stack_1?
&linear_update/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_17/stack_2?
linear_update/strided_slice_17StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_17/stack:output:0/linear_update/strided_slice_17/stack_1:output:0/linear_update/strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_17?
linear_update/einsum_3/EinsumEinsum'linear_update/strided_slice_17:output:0 linear_update/Reshape_3:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_3/Einsum?
$linear_update/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_18/stack?
&linear_update/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_18/stack_1?
&linear_update/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_18/stack_2?
linear_update/strided_slice_18StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_18/stack:output:0/linear_update/strided_slice_18/stack_1:output:0/linear_update/strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_18u
linear_update/mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_20/y?
linear_update/mul_20Mul'linear_update/strided_slice_18:output:0linear_update/mul_20/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_20y
linear_update/Exp_4Explinear_update/mul_20:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_4?
$linear_update/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_19/stack?
&linear_update/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_19/stack_1?
&linear_update/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_19/stack_2?
linear_update/strided_slice_19StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_19/stack:output:0/linear_update/strided_slice_19/stack_1:output:0/linear_update/strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_19u
linear_update/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_21/y?
linear_update/mul_21Mul'linear_update/strided_slice_19:output:0linear_update/mul_21/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_21y
linear_update/Cos_4Coslinear_update/mul_21:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_4?
$linear_update/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_20/stack?
&linear_update/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_20/stack_1?
&linear_update/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_20/stack_2?
linear_update/strided_slice_20StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_20/stack:output:0/linear_update/strided_slice_20/stack_1:output:0/linear_update/strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_20u
linear_update/mul_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_22/y?
linear_update/mul_22Mul'linear_update/strided_slice_20:output:0linear_update/mul_22/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_22y
linear_update/Sin_4Sinlinear_update/mul_22:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_4?
linear_update/Mul_23Mullinear_update/Exp_4:y:0linear_update/Cos_4:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_23?
linear_update/Mul_24Mullinear_update/Exp_4:y:0linear_update/Sin_4:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_24y
linear_update/Neg_4Neglinear_update/Mul_24:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_4?
linear_update/stack_4Packlinear_update/Mul_23:z:0linear_update/Neg_4:y:0linear_update/Mul_24:z:0linear_update/Mul_23:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_4?
linear_update/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_4/shape?
linear_update/Reshape_4Reshapelinear_update/stack_4:output:0&linear_update/Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_4?
$linear_update/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_21/stack?
&linear_update/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2(
&linear_update/strided_slice_21/stack_1?
&linear_update/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_21/stack_2?
linear_update/strided_slice_21StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_21/stack:output:0/linear_update/strided_slice_21/stack_1:output:0/linear_update/strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_21?
linear_update/einsum_4/EinsumEinsum'linear_update/strided_slice_21:output:0 linear_update/Reshape_4:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_4/Einsum?
$linear_update/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_22/stack?
&linear_update/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_22/stack_1?
&linear_update/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_22/stack_2?
linear_update/strided_slice_22StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_22/stack:output:0/linear_update/strided_slice_22/stack_1:output:0/linear_update/strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_22u
linear_update/mul_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_25/y?
linear_update/mul_25Mul'linear_update/strided_slice_22:output:0linear_update/mul_25/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_25y
linear_update/Exp_5Explinear_update/mul_25:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_5?
$linear_update/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_23/stack?
&linear_update/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_23/stack_1?
&linear_update/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_23/stack_2?
linear_update/strided_slice_23StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_23/stack:output:0/linear_update/strided_slice_23/stack_1:output:0/linear_update/strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_23u
linear_update/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_26/y?
linear_update/mul_26Mul'linear_update/strided_slice_23:output:0linear_update/mul_26/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_26y
linear_update/Cos_5Coslinear_update/mul_26:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_5?
$linear_update/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_24/stack?
&linear_update/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_24/stack_1?
&linear_update/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_24/stack_2?
linear_update/strided_slice_24StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_24/stack:output:0/linear_update/strided_slice_24/stack_1:output:0/linear_update/strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_24u
linear_update/mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_27/y?
linear_update/mul_27Mul'linear_update/strided_slice_24:output:0linear_update/mul_27/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_27y
linear_update/Sin_5Sinlinear_update/mul_27:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_5?
linear_update/Mul_28Mullinear_update/Exp_5:y:0linear_update/Cos_5:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_28?
linear_update/Mul_29Mullinear_update/Exp_5:y:0linear_update/Sin_5:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_29y
linear_update/Neg_5Neglinear_update/Mul_29:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_5?
linear_update/stack_5Packlinear_update/Mul_28:z:0linear_update/Neg_5:y:0linear_update/Mul_29:z:0linear_update/Mul_28:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_5?
linear_update/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_5/shape?
linear_update/Reshape_5Reshapelinear_update/stack_5:output:0&linear_update/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_5?
$linear_update/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2&
$linear_update/strided_slice_25/stack?
&linear_update/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_25/stack_1?
&linear_update/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_25/stack_2?
linear_update/strided_slice_25StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_25/stack:output:0/linear_update/strided_slice_25/stack_1:output:0/linear_update/strided_slice_25/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_25?
linear_update/einsum_5/EinsumEinsum'linear_update/strided_slice_25:output:0 linear_update/Reshape_5:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_5/Einsum?
$linear_update/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_26/stack?
&linear_update/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_26/stack_1?
&linear_update/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_26/stack_2?
linear_update/strided_slice_26StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_26/stack:output:0/linear_update/strided_slice_26/stack_1:output:0/linear_update/strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_26u
linear_update/mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_30/y?
linear_update/mul_30Mul'linear_update/strided_slice_26:output:0linear_update/mul_30/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_30y
linear_update/Exp_6Explinear_update/mul_30:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_6?
$linear_update/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_27/stack?
&linear_update/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_27/stack_1?
&linear_update/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_27/stack_2?
linear_update/strided_slice_27StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_27/stack:output:0/linear_update/strided_slice_27/stack_1:output:0/linear_update/strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_27u
linear_update/mul_31/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_31/y?
linear_update/mul_31Mul'linear_update/strided_slice_27:output:0linear_update/mul_31/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_31y
linear_update/Cos_6Coslinear_update/mul_31:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_6?
$linear_update/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_28/stack?
&linear_update/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_28/stack_1?
&linear_update/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_28/stack_2?
linear_update/strided_slice_28StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_28/stack:output:0/linear_update/strided_slice_28/stack_1:output:0/linear_update/strided_slice_28/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_28u
linear_update/mul_32/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_32/y?
linear_update/mul_32Mul'linear_update/strided_slice_28:output:0linear_update/mul_32/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_32y
linear_update/Sin_6Sinlinear_update/mul_32:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_6?
linear_update/Mul_33Mullinear_update/Exp_6:y:0linear_update/Cos_6:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_33?
linear_update/Mul_34Mullinear_update/Exp_6:y:0linear_update/Sin_6:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_34y
linear_update/Neg_6Neglinear_update/Mul_34:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_6?
linear_update/stack_6Packlinear_update/Mul_33:z:0linear_update/Neg_6:y:0linear_update/Mul_34:z:0linear_update/Mul_33:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_6?
linear_update/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_6/shape?
linear_update/Reshape_6Reshapelinear_update/stack_6:output:0&linear_update/Reshape_6/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_6?
$linear_update/strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_29/stack?
&linear_update/strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_29/stack_1?
&linear_update/strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_29/stack_2?
linear_update/strided_slice_29StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_29/stack:output:0/linear_update/strided_slice_29/stack_1:output:0/linear_update/strided_slice_29/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_29?
linear_update/einsum_6/EinsumEinsum'linear_update/strided_slice_29:output:0 linear_update/Reshape_6:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_6/Einsum?
$linear_update/strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_30/stack?
&linear_update/strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_30/stack_1?
&linear_update/strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_30/stack_2?
linear_update/strided_slice_30StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_30/stack:output:0/linear_update/strided_slice_30/stack_1:output:0/linear_update/strided_slice_30/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_30u
linear_update/mul_35/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_35/y?
linear_update/mul_35Mul'linear_update/strided_slice_30:output:0linear_update/mul_35/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_35y
linear_update/Exp_7Explinear_update/mul_35:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_7?
$linear_update/strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_31/stack?
&linear_update/strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_31/stack_1?
&linear_update/strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_31/stack_2?
linear_update/strided_slice_31StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_31/stack:output:0/linear_update/strided_slice_31/stack_1:output:0/linear_update/strided_slice_31/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_31u
linear_update/mul_36/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_36/y?
linear_update/mul_36Mul'linear_update/strided_slice_31:output:0linear_update/mul_36/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_36y
linear_update/Cos_7Coslinear_update/mul_36:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_7?
$linear_update/strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_32/stack?
&linear_update/strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_32/stack_1?
&linear_update/strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_32/stack_2?
linear_update/strided_slice_32StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_32/stack:output:0/linear_update/strided_slice_32/stack_1:output:0/linear_update/strided_slice_32/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_32u
linear_update/mul_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_37/y?
linear_update/mul_37Mul'linear_update/strided_slice_32:output:0linear_update/mul_37/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_37y
linear_update/Sin_7Sinlinear_update/mul_37:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_7?
linear_update/Mul_38Mullinear_update/Exp_7:y:0linear_update/Cos_7:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_38?
linear_update/Mul_39Mullinear_update/Exp_7:y:0linear_update/Sin_7:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_39y
linear_update/Neg_7Neglinear_update/Mul_39:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_7?
linear_update/stack_7Packlinear_update/Mul_38:z:0linear_update/Neg_7:y:0linear_update/Mul_39:z:0linear_update/Mul_38:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_7?
linear_update/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_7/shape?
linear_update/Reshape_7Reshapelinear_update/stack_7:output:0&linear_update/Reshape_7/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_7?
$linear_update/strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_33/stack?
&linear_update/strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_33/stack_1?
&linear_update/strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_33/stack_2?
linear_update/strided_slice_33StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_33/stack:output:0/linear_update/strided_slice_33/stack_1:output:0/linear_update/strided_slice_33/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_33?
linear_update/einsum_7/EinsumEinsum'linear_update/strided_slice_33:output:0 linear_update/Reshape_7:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_7/Einsum?
$linear_update/strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_34/stack?
&linear_update/strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2(
&linear_update/strided_slice_34/stack_1?
&linear_update/strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_34/stack_2?
linear_update/strided_slice_34StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_34/stack:output:0/linear_update/strided_slice_34/stack_1:output:0/linear_update/strided_slice_34/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_34u
linear_update/mul_40/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_40/y?
linear_update/mul_40Mul'linear_update/strided_slice_34:output:0linear_update/mul_40/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_40y
linear_update/Exp_8Explinear_update/mul_40:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_8?
$linear_update/strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_35/stack?
&linear_update/strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_35/stack_1?
&linear_update/strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_35/stack_2?
linear_update/strided_slice_35StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_35/stack:output:0/linear_update/strided_slice_35/stack_1:output:0/linear_update/strided_slice_35/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_35u
linear_update/mul_41/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_41/y?
linear_update/mul_41Mul'linear_update/strided_slice_35:output:0linear_update/mul_41/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_41y
linear_update/Cos_8Coslinear_update/mul_41:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_8?
$linear_update/strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_36/stack?
&linear_update/strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_36/stack_1?
&linear_update/strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_36/stack_2?
linear_update/strided_slice_36StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_36/stack:output:0/linear_update/strided_slice_36/stack_1:output:0/linear_update/strided_slice_36/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_36u
linear_update/mul_42/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_42/y?
linear_update/mul_42Mul'linear_update/strided_slice_36:output:0linear_update/mul_42/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_42y
linear_update/Sin_8Sinlinear_update/mul_42:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_8?
linear_update/Mul_43Mullinear_update/Exp_8:y:0linear_update/Cos_8:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_43?
linear_update/Mul_44Mullinear_update/Exp_8:y:0linear_update/Sin_8:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_44y
linear_update/Neg_8Neglinear_update/Mul_44:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_8?
linear_update/stack_8Packlinear_update/Mul_43:z:0linear_update/Neg_8:y:0linear_update/Mul_44:z:0linear_update/Mul_43:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_8?
linear_update/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_8/shape?
linear_update/Reshape_8Reshapelinear_update/stack_8:output:0&linear_update/Reshape_8/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_8?
$linear_update/strided_slice_37/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_37/stack?
&linear_update/strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_37/stack_1?
&linear_update/strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_37/stack_2?
linear_update/strided_slice_37StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_37/stack:output:0/linear_update/strided_slice_37/stack_1:output:0/linear_update/strided_slice_37/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_37?
linear_update/einsum_8/EinsumEinsum'linear_update/strided_slice_37:output:0 linear_update/Reshape_8:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_8/Einsum?
$linear_update/strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2&
$linear_update/strided_slice_38/stack?
&linear_update/strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2(
&linear_update/strided_slice_38/stack_1?
&linear_update/strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_38/stack_2?
linear_update/strided_slice_38StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_38/stack:output:0/linear_update/strided_slice_38/stack_1:output:0/linear_update/strided_slice_38/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_38u
linear_update/mul_45/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_45/y?
linear_update/mul_45Mul'linear_update/strided_slice_38:output:0linear_update/mul_45/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_45y
linear_update/Exp_9Explinear_update/mul_45:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_9?
$linear_update/strided_slice_39/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_39/stack?
&linear_update/strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_39/stack_1?
&linear_update/strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_39/stack_2?
linear_update/strided_slice_39StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_39/stack:output:0/linear_update/strided_slice_39/stack_1:output:0/linear_update/strided_slice_39/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_39u
linear_update/mul_46/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_46/y?
linear_update/mul_46Mul'linear_update/strided_slice_39:output:0linear_update/mul_46/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_46y
linear_update/Cos_9Coslinear_update/mul_46:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_9?
$linear_update/strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_40/stack?
&linear_update/strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_40/stack_1?
&linear_update/strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_40/stack_2?
linear_update/strided_slice_40StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_40/stack:output:0/linear_update/strided_slice_40/stack_1:output:0/linear_update/strided_slice_40/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_40u
linear_update/mul_47/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_47/y?
linear_update/mul_47Mul'linear_update/strided_slice_40:output:0linear_update/mul_47/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_47y
linear_update/Sin_9Sinlinear_update/mul_47:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_9?
linear_update/Mul_48Mullinear_update/Exp_9:y:0linear_update/Cos_9:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_48?
linear_update/Mul_49Mullinear_update/Exp_9:y:0linear_update/Sin_9:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_49y
linear_update/Neg_9Neglinear_update/Mul_49:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_9?
linear_update/stack_9Packlinear_update/Mul_48:z:0linear_update/Neg_9:y:0linear_update/Mul_49:z:0linear_update/Mul_48:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_9?
linear_update/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_9/shape?
linear_update/Reshape_9Reshapelinear_update/stack_9:output:0&linear_update/Reshape_9/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_9?
$linear_update/strided_slice_41/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_41/stack?
&linear_update/strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_41/stack_1?
&linear_update/strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_41/stack_2?
linear_update/strided_slice_41StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_41/stack:output:0/linear_update/strided_slice_41/stack_1:output:0/linear_update/strided_slice_41/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_41?
linear_update/einsum_9/EinsumEinsum'linear_update/strided_slice_41:output:0 linear_update/Reshape_9:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_9/Einsum?
linear_update/stack_10Pack$linear_update/einsum/Einsum:output:0&linear_update/einsum_1/Einsum:output:0&linear_update/einsum_2/Einsum:output:0&linear_update/einsum_3/Einsum:output:0&linear_update/einsum_4/Einsum:output:0&linear_update/einsum_5/Einsum:output:0&linear_update/einsum_6/Einsum:output:0&linear_update/einsum_7/Einsum:output:0&linear_update/einsum_8/Einsum:output:0&linear_update/einsum_9/Einsum:output:0*
N
*
T0*+
_output_shapes
:?????????
*

axis2
linear_update/stack_10?
linear_update/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
linear_update/Reshape_10/shape?
linear_update/Reshape_10Reshapelinear_update/stack_10:output:0'linear_update/Reshape_10/shape:output:0*
T0*'
_output_shapes
:?????????2
linear_update/Reshape_10?
linear_update/stack_11Pack!linear_update/Reshape_10:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
linear_update/stack_11?
linear_update/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2 
linear_update/Reshape_11/shape?
linear_update/Reshape_11Reshapelinear_update/stack_11:output:0'linear_update/Reshape_11/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_11?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_4_tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentitydense_4/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity!linear_update/Reshape_11:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_36973C
?dense_4_kernel_regularizer_square_readvariableop_dense_4_kernel
identity??0dense_4/kernel/Regularizer/Square/ReadVariableOp?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_4_kernel_regularizer_square_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentity"dense_4/kernel/Regularizer/mul:z:01^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp
?
H
.__inference_dense_4_activity_regularizer_34765
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
?Z
?
B__inference_model_4_layer_call_and_return_conditional_losses_35324
input_3
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_348112!
dense_3/StatefulPartitionedCall?
+dense_3/ActivityRegularizer/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_3_activity_regularizer_348372-
+dense_3/ActivityRegularizer/PartitionedCall?
!dense_3/ActivityRegularizer/ShapeShape(dense_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_3/ActivityRegularizer/Shape?
/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_3/ActivityRegularizer/strided_slice/stack?
1dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_1?
1dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_2?
)dense_3/ActivityRegularizer/strided_sliceStridedSlice*dense_3/ActivityRegularizer/Shape:output:08dense_3/ActivityRegularizer/strided_slice/stack:output:0:dense_3/ActivityRegularizer/strided_slice/stack_1:output:0:dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_3/ActivityRegularizer/strided_slice?
 dense_3/ActivityRegularizer/CastCast2dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_3/ActivityRegularizer/Cast?
#dense_3/ActivityRegularizer/truedivRealDiv4dense_3/ActivityRegularizer/PartitionedCall:output:0$dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_3/ActivityRegularizer/truediv?
activation_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_348582
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_349082!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_4_activity_regularizer_349342-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
linear_update/PartitionedCallPartitionedCallinput_3(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_352892
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
'__inference_dense_3_layer_call_fn_36506

inputs
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_kerneldense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_348112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?Z
?
B__inference_model_4_layer_call_and_return_conditional_losses_35433

inputs
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_348112!
dense_3/StatefulPartitionedCall?
+dense_3/ActivityRegularizer/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_3_activity_regularizer_348372-
+dense_3/ActivityRegularizer/PartitionedCall?
!dense_3/ActivityRegularizer/ShapeShape(dense_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_3/ActivityRegularizer/Shape?
/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_3/ActivityRegularizer/strided_slice/stack?
1dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_1?
1dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_2?
)dense_3/ActivityRegularizer/strided_sliceStridedSlice*dense_3/ActivityRegularizer/Shape:output:08dense_3/ActivityRegularizer/strided_slice/stack:output:0:dense_3/ActivityRegularizer/strided_slice/stack_1:output:0:dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_3/ActivityRegularizer/strided_slice?
 dense_3/ActivityRegularizer/CastCast2dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_3/ActivityRegularizer/Cast?
#dense_3/ActivityRegularizer/truedivRealDiv4dense_3/ActivityRegularizer/PartitionedCall:output:0$dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_3/ActivityRegularizer/truediv?
activation_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_348582
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_349082!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *7
f2R0
.__inference_dense_4_activity_regularizer_349342-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
linear_update/PartitionedCallPartitionedCallinputs(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_352892
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_36951C
?dense_3_kernel_regularizer_square_readvariableop_dense_3_kernel
identity??0dense_3/kernel/Regularizer/Square/ReadVariableOp?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_3_kernel_regularizer_square_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity"dense_3/kernel/Regularizer/mul:z:01^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp
?3
?
B__inference_dense_3_layer_call_and_return_conditional_losses_36499

inputs+
'tensordot_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
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
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_model_4_layer_call_and_return_conditional_losses_36423

inputs3
/dense_3_tensordot_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias3
/dense_4_tensordot_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
 dense_3/Tensordot/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes?
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/freeh
dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape?
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2?
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod?
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1?
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack?
dense_3/Tensordot/transpose	Transposeinputs!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_3/Tensordot/transpose?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_3/Tensordot/Reshape?
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/Tensordot/MatMul?
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_3/Tensordot/Const_2?
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_3/Tensordot?
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_3/BiasAdd?
"dense_3/ActivityRegularizer/SquareSquaredense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2$
"dense_3/ActivityRegularizer/Square?
!dense_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_3/ActivityRegularizer/Const?
dense_3/ActivityRegularizer/SumSum&dense_3/ActivityRegularizer/Square:y:0*dense_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_3/ActivityRegularizer/Sum?
!dense_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_3/ActivityRegularizer/mul/x?
dense_3/ActivityRegularizer/mulMul*dense_3/ActivityRegularizer/mul/x:output:0(dense_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_3/ActivityRegularizer/mul?
!dense_3/ActivityRegularizer/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_3/ActivityRegularizer/Shape?
/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_3/ActivityRegularizer/strided_slice/stack?
1dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_1?
1dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_3/ActivityRegularizer/strided_slice/stack_2?
)dense_3/ActivityRegularizer/strided_sliceStridedSlice*dense_3/ActivityRegularizer/Shape:output:08dense_3/ActivityRegularizer/strided_slice/stack:output:0:dense_3/ActivityRegularizer/strided_slice/stack_1:output:0:dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_3/ActivityRegularizer/strided_slice?
 dense_3/ActivityRegularizer/CastCast2dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_3/ActivityRegularizer/Cast?
#dense_3/ActivityRegularizer/truedivRealDiv#dense_3/ActivityRegularizer/mul:z:0$dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_3/ActivityRegularizer/truediv
activation_2/ReluReludense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
activation_2/Relu?
 dense_4/Tensordot/ReadVariableOpReadVariableOp/dense_4_tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes?
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/free?
dense_4/Tensordot/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape?
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axis?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2?
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod?
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1?
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axis?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack?
dense_4/Tensordot/transpose	Transposeactivation_2/Relu:activations:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_4/Tensordot/transpose?
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_4/Tensordot/Reshape?
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/Tensordot/MatMul?
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/Const_2?
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axis?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_4/Tensordot?
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_4/BiasAdd?
"dense_4/ActivityRegularizer/SquareSquaredense_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
"dense_4/ActivityRegularizer/Square?
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_4/ActivityRegularizer/Const?
dense_4/ActivityRegularizer/SumSum&dense_4/ActivityRegularizer/Square:y:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_4/ActivityRegularizer/Sum?
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_4/ActivityRegularizer/mul/x?
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_4/ActivityRegularizer/mul?
!dense_4/ActivityRegularizer/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv#dense_4/ActivityRegularizer/mul:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
!linear_update/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!linear_update/strided_slice/stack?
#linear_update/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice/stack_1?
#linear_update/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#linear_update/strided_slice/stack_2?
linear_update/strided_sliceStridedSliceinputs*linear_update/strided_slice/stack:output:0,linear_update/strided_slice/stack_1:output:0,linear_update/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice?
#linear_update/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#linear_update/strided_slice_1/stack?
%linear_update/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_1/stack_1?
%linear_update/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_1/stack_2?
linear_update/strided_slice_1StridedSlicedense_4/BiasAdd:output:0,linear_update/strided_slice_1/stack:output:0.linear_update/strided_slice_1/stack_1:output:0.linear_update/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_1?
#linear_update/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#linear_update/strided_slice_2/stack?
%linear_update/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_2/stack_1?
%linear_update/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_2/stack_2?
linear_update/strided_slice_2StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_2/stack:output:0.linear_update/strided_slice_2/stack_1:output:0.linear_update/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_2o
linear_update/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul/y?
linear_update/mulMul&linear_update/strided_slice_2:output:0linear_update/mul/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mulr
linear_update/ExpExplinear_update/mul:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp?
#linear_update/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2%
#linear_update/strided_slice_3/stack?
%linear_update/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_3/stack_1?
%linear_update/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_3/stack_2?
linear_update/strided_slice_3StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_3/stack:output:0.linear_update/strided_slice_3/stack_1:output:0.linear_update/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_3s
linear_update/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_1/y?
linear_update/mul_1Mul&linear_update/strided_slice_3:output:0linear_update/mul_1/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_1t
linear_update/CosCoslinear_update/mul_1:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos?
#linear_update/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2%
#linear_update/strided_slice_4/stack?
%linear_update/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_4/stack_1?
%linear_update/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_4/stack_2?
linear_update/strided_slice_4StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_4/stack:output:0.linear_update/strided_slice_4/stack_1:output:0.linear_update/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_4s
linear_update/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_2/y?
linear_update/mul_2Mul&linear_update/strided_slice_4:output:0linear_update/mul_2/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_2t
linear_update/SinSinlinear_update/mul_2:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin?
linear_update/Mul_3Mullinear_update/Exp:y:0linear_update/Cos:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_3?
linear_update/Mul_4Mullinear_update/Exp:y:0linear_update/Sin:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_4t
linear_update/NegNeglinear_update/Mul_4:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg?
linear_update/stackPacklinear_update/Mul_3:z:0linear_update/Neg:y:0linear_update/Mul_4:z:0linear_update/Mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack?
linear_update/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape/shape?
linear_update/ReshapeReshapelinear_update/stack:output:0$linear_update/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape?
#linear_update/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#linear_update/strided_slice_5/stack?
%linear_update/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_5/stack_1?
%linear_update/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_5/stack_2?
linear_update/strided_slice_5StridedSlice$linear_update/strided_slice:output:0,linear_update/strided_slice_5/stack:output:0.linear_update/strided_slice_5/stack_1:output:0.linear_update/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
linear_update/strided_slice_5?
linear_update/einsum/EinsumEinsum&linear_update/strided_slice_5:output:0linear_update/Reshape:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum/Einsum?
#linear_update/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_6/stack?
%linear_update/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_6/stack_1?
%linear_update/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_6/stack_2?
linear_update/strided_slice_6StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_6/stack:output:0.linear_update/strided_slice_6/stack_1:output:0.linear_update/strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_6s
linear_update/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_5/y?
linear_update/mul_5Mul&linear_update/strided_slice_6:output:0linear_update/mul_5/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_5x
linear_update/Exp_1Explinear_update/mul_5:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_1?
#linear_update/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_7/stack?
%linear_update/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_7/stack_1?
%linear_update/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_7/stack_2?
linear_update/strided_slice_7StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_7/stack:output:0.linear_update/strided_slice_7/stack_1:output:0.linear_update/strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_7s
linear_update/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_6/y?
linear_update/mul_6Mul&linear_update/strided_slice_7:output:0linear_update/mul_6/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_6x
linear_update/Cos_1Coslinear_update/mul_6:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_1?
#linear_update/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_8/stack?
%linear_update/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_8/stack_1?
%linear_update/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_8/stack_2?
linear_update/strided_slice_8StridedSlice&linear_update/strided_slice_1:output:0,linear_update/strided_slice_8/stack:output:0.linear_update/strided_slice_8/stack_1:output:0.linear_update/strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
linear_update/strided_slice_8s
linear_update/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_7/y?
linear_update/mul_7Mul&linear_update/strided_slice_8:output:0linear_update/mul_7/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_7x
linear_update/Sin_1Sinlinear_update/mul_7:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_1?
linear_update/Mul_8Mullinear_update/Exp_1:y:0linear_update/Cos_1:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_8?
linear_update/Mul_9Mullinear_update/Exp_1:y:0linear_update/Sin_1:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_9x
linear_update/Neg_1Neglinear_update/Mul_9:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_1?
linear_update/stack_1Packlinear_update/Mul_8:z:0linear_update/Neg_1:y:0linear_update/Mul_9:z:0linear_update/Mul_8:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_1?
linear_update/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_1/shape?
linear_update/Reshape_1Reshapelinear_update/stack_1:output:0&linear_update/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_1?
#linear_update/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#linear_update/strided_slice_9/stack?
%linear_update/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%linear_update/strided_slice_9/stack_1?
%linear_update/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%linear_update/strided_slice_9/stack_2?
linear_update/strided_slice_9StridedSlice$linear_update/strided_slice:output:0,linear_update/strided_slice_9/stack:output:0.linear_update/strided_slice_9/stack_1:output:0.linear_update/strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
linear_update/strided_slice_9?
linear_update/einsum_1/EinsumEinsum&linear_update/strided_slice_9:output:0 linear_update/Reshape_1:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_1/Einsum?
$linear_update/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_10/stack?
&linear_update/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_10/stack_1?
&linear_update/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_10/stack_2?
linear_update/strided_slice_10StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_10/stack:output:0/linear_update/strided_slice_10/stack_1:output:0/linear_update/strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_10u
linear_update/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_10/y?
linear_update/mul_10Mul'linear_update/strided_slice_10:output:0linear_update/mul_10/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_10y
linear_update/Exp_2Explinear_update/mul_10:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_2?
$linear_update/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_11/stack?
&linear_update/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_11/stack_1?
&linear_update/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_11/stack_2?
linear_update/strided_slice_11StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_11/stack:output:0/linear_update/strided_slice_11/stack_1:output:0/linear_update/strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_11u
linear_update/mul_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_11/y?
linear_update/mul_11Mul'linear_update/strided_slice_11:output:0linear_update/mul_11/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_11y
linear_update/Cos_2Coslinear_update/mul_11:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_2?
$linear_update/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_12/stack?
&linear_update/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_12/stack_1?
&linear_update/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_12/stack_2?
linear_update/strided_slice_12StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_12/stack:output:0/linear_update/strided_slice_12/stack_1:output:0/linear_update/strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_12u
linear_update/mul_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_12/y?
linear_update/mul_12Mul'linear_update/strided_slice_12:output:0linear_update/mul_12/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_12y
linear_update/Sin_2Sinlinear_update/mul_12:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_2?
linear_update/Mul_13Mullinear_update/Exp_2:y:0linear_update/Cos_2:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_13?
linear_update/Mul_14Mullinear_update/Exp_2:y:0linear_update/Sin_2:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_14y
linear_update/Neg_2Neglinear_update/Mul_14:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_2?
linear_update/stack_2Packlinear_update/Mul_13:z:0linear_update/Neg_2:y:0linear_update/Mul_14:z:0linear_update/Mul_13:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_2?
linear_update/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_2/shape?
linear_update/Reshape_2Reshapelinear_update/stack_2:output:0&linear_update/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_2?
$linear_update/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_13/stack?
&linear_update/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_13/stack_1?
&linear_update/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_13/stack_2?
linear_update/strided_slice_13StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_13/stack:output:0/linear_update/strided_slice_13/stack_1:output:0/linear_update/strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_13?
linear_update/einsum_2/EinsumEinsum'linear_update/strided_slice_13:output:0 linear_update/Reshape_2:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_2/Einsum?
$linear_update/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_14/stack?
&linear_update/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_14/stack_1?
&linear_update/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_14/stack_2?
linear_update/strided_slice_14StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_14/stack:output:0/linear_update/strided_slice_14/stack_1:output:0/linear_update/strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_14u
linear_update/mul_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_15/y?
linear_update/mul_15Mul'linear_update/strided_slice_14:output:0linear_update/mul_15/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_15y
linear_update/Exp_3Explinear_update/mul_15:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_3?
$linear_update/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_15/stack?
&linear_update/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_15/stack_1?
&linear_update/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_15/stack_2?
linear_update/strided_slice_15StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_15/stack:output:0/linear_update/strided_slice_15/stack_1:output:0/linear_update/strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_15u
linear_update/mul_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_16/y?
linear_update/mul_16Mul'linear_update/strided_slice_15:output:0linear_update/mul_16/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_16y
linear_update/Cos_3Coslinear_update/mul_16:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_3?
$linear_update/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_16/stack?
&linear_update/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_16/stack_1?
&linear_update/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_16/stack_2?
linear_update/strided_slice_16StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_16/stack:output:0/linear_update/strided_slice_16/stack_1:output:0/linear_update/strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_16u
linear_update/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_17/y?
linear_update/mul_17Mul'linear_update/strided_slice_16:output:0linear_update/mul_17/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_17y
linear_update/Sin_3Sinlinear_update/mul_17:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_3?
linear_update/Mul_18Mullinear_update/Exp_3:y:0linear_update/Cos_3:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_18?
linear_update/Mul_19Mullinear_update/Exp_3:y:0linear_update/Sin_3:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_19y
linear_update/Neg_3Neglinear_update/Mul_19:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_3?
linear_update/stack_3Packlinear_update/Mul_18:z:0linear_update/Neg_3:y:0linear_update/Mul_19:z:0linear_update/Mul_18:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_3?
linear_update/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_3/shape?
linear_update/Reshape_3Reshapelinear_update/stack_3:output:0&linear_update/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_3?
$linear_update/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_17/stack?
&linear_update/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_17/stack_1?
&linear_update/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_17/stack_2?
linear_update/strided_slice_17StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_17/stack:output:0/linear_update/strided_slice_17/stack_1:output:0/linear_update/strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_17?
linear_update/einsum_3/EinsumEinsum'linear_update/strided_slice_17:output:0 linear_update/Reshape_3:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_3/Einsum?
$linear_update/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_18/stack?
&linear_update/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_18/stack_1?
&linear_update/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_18/stack_2?
linear_update/strided_slice_18StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_18/stack:output:0/linear_update/strided_slice_18/stack_1:output:0/linear_update/strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_18u
linear_update/mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_20/y?
linear_update/mul_20Mul'linear_update/strided_slice_18:output:0linear_update/mul_20/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_20y
linear_update/Exp_4Explinear_update/mul_20:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_4?
$linear_update/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_19/stack?
&linear_update/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_19/stack_1?
&linear_update/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_19/stack_2?
linear_update/strided_slice_19StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_19/stack:output:0/linear_update/strided_slice_19/stack_1:output:0/linear_update/strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_19u
linear_update/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_21/y?
linear_update/mul_21Mul'linear_update/strided_slice_19:output:0linear_update/mul_21/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_21y
linear_update/Cos_4Coslinear_update/mul_21:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_4?
$linear_update/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_20/stack?
&linear_update/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_20/stack_1?
&linear_update/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_20/stack_2?
linear_update/strided_slice_20StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_20/stack:output:0/linear_update/strided_slice_20/stack_1:output:0/linear_update/strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_20u
linear_update/mul_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_22/y?
linear_update/mul_22Mul'linear_update/strided_slice_20:output:0linear_update/mul_22/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_22y
linear_update/Sin_4Sinlinear_update/mul_22:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_4?
linear_update/Mul_23Mullinear_update/Exp_4:y:0linear_update/Cos_4:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_23?
linear_update/Mul_24Mullinear_update/Exp_4:y:0linear_update/Sin_4:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_24y
linear_update/Neg_4Neglinear_update/Mul_24:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_4?
linear_update/stack_4Packlinear_update/Mul_23:z:0linear_update/Neg_4:y:0linear_update/Mul_24:z:0linear_update/Mul_23:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_4?
linear_update/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_4/shape?
linear_update/Reshape_4Reshapelinear_update/stack_4:output:0&linear_update/Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_4?
$linear_update/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_21/stack?
&linear_update/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2(
&linear_update/strided_slice_21/stack_1?
&linear_update/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_21/stack_2?
linear_update/strided_slice_21StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_21/stack:output:0/linear_update/strided_slice_21/stack_1:output:0/linear_update/strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_21?
linear_update/einsum_4/EinsumEinsum'linear_update/strided_slice_21:output:0 linear_update/Reshape_4:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_4/Einsum?
$linear_update/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_22/stack?
&linear_update/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_22/stack_1?
&linear_update/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_22/stack_2?
linear_update/strided_slice_22StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_22/stack:output:0/linear_update/strided_slice_22/stack_1:output:0/linear_update/strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_22u
linear_update/mul_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_25/y?
linear_update/mul_25Mul'linear_update/strided_slice_22:output:0linear_update/mul_25/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_25y
linear_update/Exp_5Explinear_update/mul_25:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_5?
$linear_update/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_23/stack?
&linear_update/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_23/stack_1?
&linear_update/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_23/stack_2?
linear_update/strided_slice_23StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_23/stack:output:0/linear_update/strided_slice_23/stack_1:output:0/linear_update/strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_23u
linear_update/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_26/y?
linear_update/mul_26Mul'linear_update/strided_slice_23:output:0linear_update/mul_26/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_26y
linear_update/Cos_5Coslinear_update/mul_26:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_5?
$linear_update/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_24/stack?
&linear_update/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_24/stack_1?
&linear_update/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_24/stack_2?
linear_update/strided_slice_24StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_24/stack:output:0/linear_update/strided_slice_24/stack_1:output:0/linear_update/strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_24u
linear_update/mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_27/y?
linear_update/mul_27Mul'linear_update/strided_slice_24:output:0linear_update/mul_27/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_27y
linear_update/Sin_5Sinlinear_update/mul_27:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_5?
linear_update/Mul_28Mullinear_update/Exp_5:y:0linear_update/Cos_5:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_28?
linear_update/Mul_29Mullinear_update/Exp_5:y:0linear_update/Sin_5:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_29y
linear_update/Neg_5Neglinear_update/Mul_29:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_5?
linear_update/stack_5Packlinear_update/Mul_28:z:0linear_update/Neg_5:y:0linear_update/Mul_29:z:0linear_update/Mul_28:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_5?
linear_update/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_5/shape?
linear_update/Reshape_5Reshapelinear_update/stack_5:output:0&linear_update/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_5?
$linear_update/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2&
$linear_update/strided_slice_25/stack?
&linear_update/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_25/stack_1?
&linear_update/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_25/stack_2?
linear_update/strided_slice_25StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_25/stack:output:0/linear_update/strided_slice_25/stack_1:output:0/linear_update/strided_slice_25/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_25?
linear_update/einsum_5/EinsumEinsum'linear_update/strided_slice_25:output:0 linear_update/Reshape_5:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_5/Einsum?
$linear_update/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_26/stack?
&linear_update/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_26/stack_1?
&linear_update/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_26/stack_2?
linear_update/strided_slice_26StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_26/stack:output:0/linear_update/strided_slice_26/stack_1:output:0/linear_update/strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_26u
linear_update/mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_30/y?
linear_update/mul_30Mul'linear_update/strided_slice_26:output:0linear_update/mul_30/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_30y
linear_update/Exp_6Explinear_update/mul_30:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_6?
$linear_update/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_27/stack?
&linear_update/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_27/stack_1?
&linear_update/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_27/stack_2?
linear_update/strided_slice_27StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_27/stack:output:0/linear_update/strided_slice_27/stack_1:output:0/linear_update/strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_27u
linear_update/mul_31/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_31/y?
linear_update/mul_31Mul'linear_update/strided_slice_27:output:0linear_update/mul_31/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_31y
linear_update/Cos_6Coslinear_update/mul_31:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_6?
$linear_update/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_28/stack?
&linear_update/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_28/stack_1?
&linear_update/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_28/stack_2?
linear_update/strided_slice_28StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_28/stack:output:0/linear_update/strided_slice_28/stack_1:output:0/linear_update/strided_slice_28/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_28u
linear_update/mul_32/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_32/y?
linear_update/mul_32Mul'linear_update/strided_slice_28:output:0linear_update/mul_32/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_32y
linear_update/Sin_6Sinlinear_update/mul_32:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_6?
linear_update/Mul_33Mullinear_update/Exp_6:y:0linear_update/Cos_6:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_33?
linear_update/Mul_34Mullinear_update/Exp_6:y:0linear_update/Sin_6:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_34y
linear_update/Neg_6Neglinear_update/Mul_34:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_6?
linear_update/stack_6Packlinear_update/Mul_33:z:0linear_update/Neg_6:y:0linear_update/Mul_34:z:0linear_update/Mul_33:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_6?
linear_update/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_6/shape?
linear_update/Reshape_6Reshapelinear_update/stack_6:output:0&linear_update/Reshape_6/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_6?
$linear_update/strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_29/stack?
&linear_update/strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_29/stack_1?
&linear_update/strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_29/stack_2?
linear_update/strided_slice_29StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_29/stack:output:0/linear_update/strided_slice_29/stack_1:output:0/linear_update/strided_slice_29/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_29?
linear_update/einsum_6/EinsumEinsum'linear_update/strided_slice_29:output:0 linear_update/Reshape_6:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_6/Einsum?
$linear_update/strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_30/stack?
&linear_update/strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_30/stack_1?
&linear_update/strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_30/stack_2?
linear_update/strided_slice_30StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_30/stack:output:0/linear_update/strided_slice_30/stack_1:output:0/linear_update/strided_slice_30/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_30u
linear_update/mul_35/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_35/y?
linear_update/mul_35Mul'linear_update/strided_slice_30:output:0linear_update/mul_35/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_35y
linear_update/Exp_7Explinear_update/mul_35:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_7?
$linear_update/strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_31/stack?
&linear_update/strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_31/stack_1?
&linear_update/strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_31/stack_2?
linear_update/strided_slice_31StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_31/stack:output:0/linear_update/strided_slice_31/stack_1:output:0/linear_update/strided_slice_31/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_31u
linear_update/mul_36/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_36/y?
linear_update/mul_36Mul'linear_update/strided_slice_31:output:0linear_update/mul_36/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_36y
linear_update/Cos_7Coslinear_update/mul_36:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_7?
$linear_update/strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_32/stack?
&linear_update/strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_32/stack_1?
&linear_update/strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_32/stack_2?
linear_update/strided_slice_32StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_32/stack:output:0/linear_update/strided_slice_32/stack_1:output:0/linear_update/strided_slice_32/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_32u
linear_update/mul_37/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_37/y?
linear_update/mul_37Mul'linear_update/strided_slice_32:output:0linear_update/mul_37/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_37y
linear_update/Sin_7Sinlinear_update/mul_37:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_7?
linear_update/Mul_38Mullinear_update/Exp_7:y:0linear_update/Cos_7:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_38?
linear_update/Mul_39Mullinear_update/Exp_7:y:0linear_update/Sin_7:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_39y
linear_update/Neg_7Neglinear_update/Mul_39:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_7?
linear_update/stack_7Packlinear_update/Mul_38:z:0linear_update/Neg_7:y:0linear_update/Mul_39:z:0linear_update/Mul_38:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_7?
linear_update/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_7/shape?
linear_update/Reshape_7Reshapelinear_update/stack_7:output:0&linear_update/Reshape_7/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_7?
$linear_update/strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_33/stack?
&linear_update/strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_33/stack_1?
&linear_update/strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_33/stack_2?
linear_update/strided_slice_33StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_33/stack:output:0/linear_update/strided_slice_33/stack_1:output:0/linear_update/strided_slice_33/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_33?
linear_update/einsum_7/EinsumEinsum'linear_update/strided_slice_33:output:0 linear_update/Reshape_7:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_7/Einsum?
$linear_update/strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_34/stack?
&linear_update/strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2(
&linear_update/strided_slice_34/stack_1?
&linear_update/strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_34/stack_2?
linear_update/strided_slice_34StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_34/stack:output:0/linear_update/strided_slice_34/stack_1:output:0/linear_update/strided_slice_34/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_34u
linear_update/mul_40/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_40/y?
linear_update/mul_40Mul'linear_update/strided_slice_34:output:0linear_update/mul_40/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_40y
linear_update/Exp_8Explinear_update/mul_40:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_8?
$linear_update/strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_35/stack?
&linear_update/strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_35/stack_1?
&linear_update/strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_35/stack_2?
linear_update/strided_slice_35StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_35/stack:output:0/linear_update/strided_slice_35/stack_1:output:0/linear_update/strided_slice_35/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_35u
linear_update/mul_41/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_41/y?
linear_update/mul_41Mul'linear_update/strided_slice_35:output:0linear_update/mul_41/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_41y
linear_update/Cos_8Coslinear_update/mul_41:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_8?
$linear_update/strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_36/stack?
&linear_update/strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_36/stack_1?
&linear_update/strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_36/stack_2?
linear_update/strided_slice_36StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_36/stack:output:0/linear_update/strided_slice_36/stack_1:output:0/linear_update/strided_slice_36/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_36u
linear_update/mul_42/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_42/y?
linear_update/mul_42Mul'linear_update/strided_slice_36:output:0linear_update/mul_42/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_42y
linear_update/Sin_8Sinlinear_update/mul_42:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_8?
linear_update/Mul_43Mullinear_update/Exp_8:y:0linear_update/Cos_8:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_43?
linear_update/Mul_44Mullinear_update/Exp_8:y:0linear_update/Sin_8:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_44y
linear_update/Neg_8Neglinear_update/Mul_44:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_8?
linear_update/stack_8Packlinear_update/Mul_43:z:0linear_update/Neg_8:y:0linear_update/Mul_44:z:0linear_update/Mul_43:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_8?
linear_update/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_8/shape?
linear_update/Reshape_8Reshapelinear_update/stack_8:output:0&linear_update/Reshape_8/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_8?
$linear_update/strided_slice_37/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_37/stack?
&linear_update/strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_37/stack_1?
&linear_update/strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_37/stack_2?
linear_update/strided_slice_37StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_37/stack:output:0/linear_update/strided_slice_37/stack_1:output:0/linear_update/strided_slice_37/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_37?
linear_update/einsum_8/EinsumEinsum'linear_update/strided_slice_37:output:0 linear_update/Reshape_8:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_8/Einsum?
$linear_update/strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2&
$linear_update/strided_slice_38/stack?
&linear_update/strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2(
&linear_update/strided_slice_38/stack_1?
&linear_update/strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_38/stack_2?
linear_update/strided_slice_38StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_38/stack:output:0/linear_update/strided_slice_38/stack_1:output:0/linear_update/strided_slice_38/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_38u
linear_update/mul_45/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_45/y?
linear_update/mul_45Mul'linear_update/strided_slice_38:output:0linear_update/mul_45/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_45y
linear_update/Exp_9Explinear_update/mul_45:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Exp_9?
$linear_update/strided_slice_39/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_39/stack?
&linear_update/strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_39/stack_1?
&linear_update/strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_39/stack_2?
linear_update/strided_slice_39StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_39/stack:output:0/linear_update/strided_slice_39/stack_1:output:0/linear_update/strided_slice_39/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_39u
linear_update/mul_46/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_46/y?
linear_update/mul_46Mul'linear_update/strided_slice_39:output:0linear_update/mul_46/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_46y
linear_update/Cos_9Coslinear_update/mul_46:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Cos_9?
$linear_update/strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_40/stack?
&linear_update/strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_40/stack_1?
&linear_update/strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_40/stack_2?
linear_update/strided_slice_40StridedSlice&linear_update/strided_slice_1:output:0-linear_update/strided_slice_40/stack:output:0/linear_update/strided_slice_40/stack_1:output:0/linear_update/strided_slice_40/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
linear_update/strided_slice_40u
linear_update/mul_47/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
linear_update/mul_47/y?
linear_update/mul_47Mul'linear_update/strided_slice_40:output:0linear_update/mul_47/y:output:0*
T0*#
_output_shapes
:?????????2
linear_update/mul_47y
linear_update/Sin_9Sinlinear_update/mul_47:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Sin_9?
linear_update/Mul_48Mullinear_update/Exp_9:y:0linear_update/Cos_9:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_48?
linear_update/Mul_49Mullinear_update/Exp_9:y:0linear_update/Sin_9:y:0*
T0*#
_output_shapes
:?????????2
linear_update/Mul_49y
linear_update/Neg_9Neglinear_update/Mul_49:z:0*
T0*#
_output_shapes
:?????????2
linear_update/Neg_9?
linear_update/stack_9Packlinear_update/Mul_48:z:0linear_update/Neg_9:y:0linear_update/Mul_49:z:0linear_update/Mul_48:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
linear_update/stack_9?
linear_update/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_9/shape?
linear_update/Reshape_9Reshapelinear_update/stack_9:output:0&linear_update/Reshape_9/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_9?
$linear_update/strided_slice_41/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$linear_update/strided_slice_41/stack?
&linear_update/strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&linear_update/strided_slice_41/stack_1?
&linear_update/strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&linear_update/strided_slice_41/stack_2?
linear_update/strided_slice_41StridedSlice$linear_update/strided_slice:output:0-linear_update/strided_slice_41/stack:output:0/linear_update/strided_slice_41/stack_1:output:0/linear_update/strided_slice_41/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2 
linear_update/strided_slice_41?
linear_update/einsum_9/EinsumEinsum'linear_update/strided_slice_41:output:0 linear_update/Reshape_9:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2
linear_update/einsum_9/Einsum?
linear_update/stack_10Pack$linear_update/einsum/Einsum:output:0&linear_update/einsum_1/Einsum:output:0&linear_update/einsum_2/Einsum:output:0&linear_update/einsum_3/Einsum:output:0&linear_update/einsum_4/Einsum:output:0&linear_update/einsum_5/Einsum:output:0&linear_update/einsum_6/Einsum:output:0&linear_update/einsum_7/Einsum:output:0&linear_update/einsum_8/Einsum:output:0&linear_update/einsum_9/Einsum:output:0*
N
*
T0*+
_output_shapes
:?????????
*

axis2
linear_update/stack_10?
linear_update/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2 
linear_update/Reshape_10/shape?
linear_update/Reshape_10Reshapelinear_update/stack_10:output:0'linear_update/Reshape_10/shape:output:0*
T0*'
_output_shapes
:?????????2
linear_update/Reshape_10?
linear_update/stack_11Pack!linear_update/Reshape_10:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
linear_update/stack_11?
linear_update/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2 
linear_update/Reshape_11/shape?
linear_update/Reshape_11Reshapelinear_update/stack_11:output:0'linear_update/Reshape_11/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_11?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_4_tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentitydense_4/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity!linear_update/Reshape_11:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_36984?
;dense_4_bias_regularizer_square_readvariableop_dense_4_bias
identity??.dense_4/bias/Regularizer/Square/ReadVariableOp?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_4_bias_regularizer_square_readvariableop_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_4/bias/Regularizer/Square?
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/Const?
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum?
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_4/bias/Regularizer/mul/x?
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul?
IdentityIdentity dense_4/bias/Regularizer/mul:z:0/^dense_4/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp
?3
?
B__inference_dense_3_layer_call_and_return_conditional_losses_34811

inputs+
'tensordot_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
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
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOp?
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2!
dense_3/bias/Regularizer/Square?
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/Const?
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum?
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_3/bias/Regularizer/mul/x?
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
.__inference_dense_4_activity_regularizer_34934
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
?	
?
'__inference_model_4_layer_call_fn_36445

inputs
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_354972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_34
serving_default_input_3:0??????????
dense_44
StatefulPartitionedCall:0?????????E
linear_update4
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?+
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
trainable_variables
regularization_losses
	variables
		keras_api


signatures
*9&call_and_return_all_conditional_losses
:__call__
;_default_save_signature"?)
_tf_keras_network?({"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.22360679774997896, "maxval": 0.22360679774997896, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "linear_update", "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 20]}, "kernels": [], "num_complex": 10, "num_real": 0, "dt": 0.05}, "name": "linear_update", "inbound_nodes": [[["input_3", 0, 0, {}], ["dense_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_4", 0, 0], ["linear_update", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 20]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.22360679774997896, "maxval": 0.22360679774997896, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "linear_update", "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 20]}, "kernels": [], "num_complex": 10, "num_real": 0, "dt": 0.05}, "name": "linear_update", "inbound_nodes": [[["input_3", 0, 0, {}], ["dense_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_4", 0, 0], ["linear_update", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 20]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*<&call_and_return_all_conditional_losses
=__call__"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.22360679774997896, "maxval": 0.22360679774997896, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 20]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*>&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 170}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 170]}}
?
kernels
trainable_variables
regularization_losses
	variables
	keras_api
*B&call_and_return_all_conditional_losses
C__call__"?
_tf_keras_layer?{"class_name": "linear_update", "name": "linear_update", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 20]}, "kernels": [], "num_complex": 10, "num_real": 0, "dt": 0.05}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 20]}, {"class_name": "TensorShape", "items": [null, 1, 20]}]}
<
0
1
2
3"
trackable_list_wrapper
<
D0
E1
F2
G3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
 layer_metrics
!metrics

"layers
#non_trainable_variables
trainable_variables
$layer_regularization_losses
regularization_losses
	variables
:__call__
;_default_save_signature
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
,
Hserving_default"
signature_map
!:	?2dense_3/kernel
:?2dense_3/bias
.
0
1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
%layer_metrics
&metrics

'layers
(non_trainable_variables
trainable_variables
)layer_regularization_losses
regularization_losses
	variables
=__call__
Iactivity_regularizer_fn
*<&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
*layer_metrics
+metrics

,layers
-non_trainable_variables
trainable_variables
.layer_regularization_losses
regularization_losses
	variables
?__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_4/kernel
:2dense_4/bias
.
0
1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
/layer_metrics
0metrics

1layers
2non_trainable_variables
trainable_variables
3layer_regularization_losses
regularization_losses
	variables
A__call__
Kactivity_regularizer_fn
*@&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4layer_metrics
5metrics

6layers
7non_trainable_variables
trainable_variables
8layer_regularization_losses
regularization_losses
	variables
C__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
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
D0
E1"
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
F0
G1"
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
?2?
B__inference_model_4_layer_call_and_return_conditional_losses_35983
B__inference_model_4_layer_call_and_return_conditional_losses_36423
B__inference_model_4_layer_call_and_return_conditional_losses_35377
B__inference_model_4_layer_call_and_return_conditional_losses_35324?
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
?2?
'__inference_model_4_layer_call_fn_36445
'__inference_model_4_layer_call_fn_35442
'__inference_model_4_layer_call_fn_35506
'__inference_model_4_layer_call_fn_36434?
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
 __inference__wrapped_model_34739?
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
input_3?????????
?2?
F__inference_dense_3_layer_call_and_return_all_conditional_losses_36515?
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
'__inference_dense_3_layer_call_fn_36506?
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
G__inference_activation_2_layer_call_and_return_conditional_losses_36520?
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
,__inference_activation_2_layer_call_fn_36525?
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
F__inference_dense_4_layer_call_and_return_all_conditional_losses_36595?
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
'__inference_dense_4_layer_call_fn_36586?
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
H__inference_linear_update_layer_call_and_return_conditional_losses_36934?
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
annotations? *
 
?2?
-__inference_linear_update_layer_call_fn_36940?
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
annotations? *
 
?2?
__inference_loss_fn_0_36951?
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
__inference_loss_fn_1_36962?
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
__inference_loss_fn_2_36973?
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
__inference_loss_fn_3_36984?
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
#__inference_signature_wrapper_35543input_3"?
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
.__inference_dense_3_activity_regularizer_34752?
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
B__inference_dense_3_layer_call_and_return_conditional_losses_36499?
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
.__inference_dense_4_activity_regularizer_34765?
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
B__inference_dense_4_layer_call_and_return_conditional_losses_36579?
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
 __inference__wrapped_model_34739?4?1
*?'
%?"
input_3?????????
? "s?p
0
dense_4%?"
dense_4?????????
<
linear_update+?(
linear_update??????????
G__inference_activation_2_layer_call_and_return_conditional_losses_36520b4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_activation_2_layer_call_fn_36525U4?1
*?'
%?"
inputs??????????
? "???????????[
.__inference_dense_3_activity_regularizer_34752)?
?
?
self
? "? ?
F__inference_dense_3_layer_call_and_return_all_conditional_losses_36515s3?0
)?&
$?!
inputs?????????
? "8?5
 ?
0??????????
?
?	
1/0 ?
B__inference_dense_3_layer_call_and_return_conditional_losses_36499e3?0
)?&
$?!
inputs?????????
? "*?'
 ?
0??????????
? ?
'__inference_dense_3_layer_call_fn_36506X3?0
)?&
$?!
inputs?????????
? "???????????[
.__inference_dense_4_activity_regularizer_34765)?
?
?
self
? "? ?
F__inference_dense_4_layer_call_and_return_all_conditional_losses_36595s4?1
*?'
%?"
inputs??????????
? "7?4
?
0?????????
?
?	
1/0 ?
B__inference_dense_4_layer_call_and_return_conditional_losses_36579e4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
'__inference_dense_4_layer_call_fn_36586X4?1
*?'
%?"
inputs??????????
? "???????????
H__inference_linear_update_layer_call_and_return_conditional_losses_36934?X?U
N?K
I?F
!?
x/0?????????
!?
x/1?????????
? ")?&
?
0?????????
? ?
-__inference_linear_update_layer_call_fn_36940xX?U
N?K
I?F
!?
x/0?????????
!?
x/1?????????
? "??????????:
__inference_loss_fn_0_36951?

? 
? "? :
__inference_loss_fn_1_36962?

? 
? "? :
__inference_loss_fn_2_36973?

? 
? "? :
__inference_loss_fn_3_36984?

? 
? "? ?
B__inference_model_4_layer_call_and_return_conditional_losses_35324?<?9
2?/
%?"
input_3?????????
p

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_35377?<?9
2?/
%?"
input_3?????????
p 

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_35983?;?8
1?.
$?!
inputs?????????
p

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_36423?;?8
1?.
$?!
inputs?????????
p 

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
'__inference_model_4_layer_call_fn_35442?<?9
2?/
%?"
input_3?????????
p

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_4_layer_call_fn_35506?<?9
2?/
%?"
input_3?????????
p 

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_4_layer_call_fn_36434?;?8
1?.
$?!
inputs?????????
p

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_4_layer_call_fn_36445?;?8
1?.
$?!
inputs?????????
p 

 
? "E?B
?
0?????????
?
1??????????
#__inference_signature_wrapper_35543???<
? 
5?2
0
input_3%?"
input_3?????????"s?p
0
dense_4%?"
dense_4?????????
<
linear_update+?(
linear_update?????????