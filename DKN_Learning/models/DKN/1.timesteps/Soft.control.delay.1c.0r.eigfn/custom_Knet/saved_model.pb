??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
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
shape:	?*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
_
 kernels
!regularization_losses
"	variables
#trainable_variables
$	keras_api
 

0
1
2
3

0
1
2
3
?
regularization_losses
	variables
%metrics
	trainable_variables
&layer_regularization_losses
'layer_metrics

(layers
)non_trainable_variables
 
 
 
 
?
regularization_losses
	variables
*metrics
trainable_variables
+layer_regularization_losses
,layer_metrics

-layers
.non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
/metrics
trainable_variables
0layer_regularization_losses
1layer_metrics

2layers
3non_trainable_variables
 
 
 
?
regularization_losses
	variables
4metrics
trainable_variables
5layer_regularization_losses
6layer_metrics

7layers
8non_trainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
9metrics
trainable_variables
:layer_regularization_losses
;layer_metrics

<layers
=non_trainable_variables
 
 
 
 
?
!regularization_losses
"	variables
>metrics
#trainable_variables
?layer_regularization_losses
@layer_metrics

Alayers
Bnon_trainable_variables
 
 
 
*
0
1
2
3
4
5
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
serving_default_input_3Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_14326
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
__inference__traced_save_14975
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
!__inference__traced_restore_14997??
?
?
!__inference__traced_restore_14997
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
?
?
'__inference_dense_3_layer_call_fn_14749

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
B__inference_dense_3_layer_call_and_return_conditional_losses_138792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_14829

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_139762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
O
-__inference_linear_update_layer_call_fn_14895
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_140692
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????:?????????:P L
+
_output_shapes
:?????????

_user_specified_namex/0:PL
+
_output_shapes
:?????????

_user_specified_namex/1
?	
?
#__inference_signature_wrapper_14326
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
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_137862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?3
?
B__inference_dense_4_layer_call_and_return_conditional_losses_13976

inputs+
'tensordot_readvariableop_dense_4_kernel'
#biasadd_readvariableop_dense_4_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
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
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
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
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

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
??
?
B__inference_model_4_layer_call_and_return_conditional_losses_14648

inputs3
/dense_3_tensordot_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias3
/dense_4_tensordot_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
&compute_aux_inputs/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2(
&compute_aux_inputs/strided_slice/stack?
(compute_aux_inputs/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2*
(compute_aux_inputs/strided_slice/stack_1?
(compute_aux_inputs/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2*
(compute_aux_inputs/strided_slice/stack_2?
 compute_aux_inputs/strided_sliceStridedSliceinputs/compute_aux_inputs/strided_slice/stack:output:01compute_aux_inputs/strided_slice/stack_1:output:01compute_aux_inputs/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2"
 compute_aux_inputs/strided_slice?
compute_aux_inputs/SquareSquare)compute_aux_inputs/strided_slice:output:0*
T0*+
_output_shapes
:?????????2
compute_aux_inputs/Square?
 compute_aux_inputs/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         2"
 compute_aux_inputs/Reshape/shape?
compute_aux_inputs/ReshapeReshapecompute_aux_inputs/Square:y:0)compute_aux_inputs/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
compute_aux_inputs/Reshape?
(compute_aux_inputs/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(compute_aux_inputs/Sum/reduction_indices?
compute_aux_inputs/SumSum#compute_aux_inputs/Reshape:output:01compute_aux_inputs/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2
compute_aux_inputs/Sum?
 dense_3/Tensordot/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
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
dense_3/Tensordot/free?
dense_3/Tensordot/ShapeShapecompute_aux_inputs/Sum:output:0*
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
dense_3/Tensordot/transpose	Transposecompute_aux_inputs/Sum:output:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
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
:	?*
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
:?????????2
dense_4/Tensordot/MatMul?
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
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
:?????????2
dense_4/Tensordot?
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_4/BiasAdd?
"dense_4/ActivityRegularizer/SquareSquaredense_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
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
:?????????*

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
:?????????*

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
valueB"       2%
#linear_update/strided_slice_3/stack?
%linear_update/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
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
valueB"       2%
#linear_update/strided_slice_4/stack?
%linear_update/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
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
linear_update/stack_1Pack$linear_update/einsum/Einsum:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
linear_update/stack_1?
linear_update/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
linear_update/Reshape_1/shape?
linear_update/Reshape_1Reshapelinear_update/stack_1:output:0&linear_update/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
linear_update/Reshape_1?
linear_update/stack_2Pack linear_update/Reshape_1:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
linear_update/stack_2?
linear_update/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_2/shape?
linear_update/Reshape_2Reshapelinear_update/stack_2:output:0&linear_update/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_2?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

Identity?

Identity_1Identity linear_update/Reshape_2:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2@
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
:?????????
 
_user_specified_nameinputs
?3
?
B__inference_dense_4_layer_call_and_return_conditional_losses_14822

inputs+
'tensordot_readvariableop_dense_4_kernel'
#biasadd_readvariableop_dense_4_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
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
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
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
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

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
H
.__inference_dense_3_activity_regularizer_13799
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
'__inference_model_4_layer_call_fn_14659

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
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_142152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
'__inference_model_4_layer_call_fn_14670

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
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_142802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
'__inference_model_4_layer_call_fn_14289
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
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_142802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?]
?
B__inference_model_4_layer_call_and_return_conditional_losses_14280

inputs
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
"compute_aux_inputs/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_138292$
"compute_aux_inputs/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall+compute_aux_inputs/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_138792!
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
.__inference_dense_3_activity_regularizer_139052-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_139262
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_139762!
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
.__inference_dense_4_activity_regularizer_140022-
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_140692
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
.__inference_dense_3_activity_regularizer_13905
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
H
.__inference_dense_4_activity_regularizer_14002
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
?3
?
B__inference_dense_3_layer_call_and_return_conditional_losses_13879

inputs+
'tensordot_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
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
:?????????2
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
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_13926

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
F__inference_dense_4_layer_call_and_return_all_conditional_losses_14838

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_139762
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
.__inference_dense_4_activity_regularizer_140022
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

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
?
?
__inference__traced_save_14975
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
': :	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
??
?
 __inference__wrapped_model_13786
input_3;
7model_4_dense_3_tensordot_readvariableop_dense_3_kernel7
3model_4_dense_3_biasadd_readvariableop_dense_3_bias;
7model_4_dense_4_tensordot_readvariableop_dense_4_kernel7
3model_4_dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??&model_4/dense_3/BiasAdd/ReadVariableOp?(model_4/dense_3/Tensordot/ReadVariableOp?&model_4/dense_4/BiasAdd/ReadVariableOp?(model_4/dense_4/Tensordot/ReadVariableOp?
.model_4/compute_aux_inputs/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model_4/compute_aux_inputs/strided_slice/stack?
0model_4/compute_aux_inputs/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model_4/compute_aux_inputs/strided_slice/stack_1?
0model_4/compute_aux_inputs/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model_4/compute_aux_inputs/strided_slice/stack_2?
(model_4/compute_aux_inputs/strided_sliceStridedSliceinput_37model_4/compute_aux_inputs/strided_slice/stack:output:09model_4/compute_aux_inputs/strided_slice/stack_1:output:09model_4/compute_aux_inputs/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2*
(model_4/compute_aux_inputs/strided_slice?
!model_4/compute_aux_inputs/SquareSquare1model_4/compute_aux_inputs/strided_slice:output:0*
T0*+
_output_shapes
:?????????2#
!model_4/compute_aux_inputs/Square?
(model_4/compute_aux_inputs/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         2*
(model_4/compute_aux_inputs/Reshape/shape?
"model_4/compute_aux_inputs/ReshapeReshape%model_4/compute_aux_inputs/Square:y:01model_4/compute_aux_inputs/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2$
"model_4/compute_aux_inputs/Reshape?
0model_4/compute_aux_inputs/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0model_4/compute_aux_inputs/Sum/reduction_indices?
model_4/compute_aux_inputs/SumSum+model_4/compute_aux_inputs/Reshape:output:09model_4/compute_aux_inputs/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2 
model_4/compute_aux_inputs/Sum?
(model_4/dense_3/Tensordot/ReadVariableOpReadVariableOp7model_4_dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
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
model_4/dense_3/Tensordot/free?
model_4/dense_3/Tensordot/ShapeShape'model_4/compute_aux_inputs/Sum:output:0*
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
#model_4/dense_3/Tensordot/transpose	Transpose'model_4/compute_aux_inputs/Sum:output:0)model_4/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2%
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
:	?*
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
:?????????2"
 model_4/dense_4/Tensordot/MatMul?
!model_4/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
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
:?????????2
model_4/dense_4/Tensordot?
&model_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp3model_4_dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02(
&model_4/dense_4/BiasAdd/ReadVariableOp?
model_4/dense_4/BiasAddBiasAdd"model_4/dense_4/Tensordot:output:0.model_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model_4/dense_4/BiasAdd?
*model_4/dense_4/ActivityRegularizer/SquareSquare model_4/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2,
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
:?????????*

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
:?????????*

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
valueB"       2-
+model_4/linear_update/strided_slice_3/stack?
-model_4/linear_update/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
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
valueB"       2-
+model_4/linear_update/strided_slice_4/stack?
-model_4/linear_update/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
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
model_4/linear_update/stack_1Pack,model_4/linear_update/einsum/Einsum:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_1?
%model_4/linear_update/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2'
%model_4/linear_update/Reshape_1/shape?
model_4/linear_update/Reshape_1Reshape&model_4/linear_update/stack_1:output:0.model_4/linear_update/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2!
model_4/linear_update/Reshape_1?
model_4/linear_update/stack_2Pack(model_4/linear_update/Reshape_1:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
model_4/linear_update/stack_2?
%model_4/linear_update/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_4/linear_update/Reshape_2/shape?
model_4/linear_update/Reshape_2Reshape&model_4/linear_update/stack_2:output:0.model_4/linear_update/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_4/linear_update/Reshape_2?
IdentityIdentity model_4/dense_4/BiasAdd:output:0'^model_4/dense_3/BiasAdd/ReadVariableOp)^model_4/dense_3/Tensordot/ReadVariableOp'^model_4/dense_4/BiasAdd/ReadVariableOp)^model_4/dense_4/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity(model_4/linear_update/Reshape_2:output:0'^model_4/dense_3/BiasAdd/ReadVariableOp)^model_4/dense_3/Tensordot/ReadVariableOp'^model_4/dense_4/BiasAdd/ReadVariableOp)^model_4/dense_4/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2P
&model_4/dense_3/BiasAdd/ReadVariableOp&model_4/dense_3/BiasAdd/ReadVariableOp2T
(model_4/dense_3/Tensordot/ReadVariableOp(model_4/dense_3/Tensordot/ReadVariableOp2P
&model_4/dense_4/BiasAdd/ReadVariableOp&model_4/dense_4/BiasAdd/ReadVariableOp2T
(model_4/dense_4/Tensordot/ReadVariableOp(model_4/dense_4/Tensordot/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?6
h
H__inference_linear_update_layer_call_and_return_conditional_losses_14069
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
:?????????*

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
:?????????*

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
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
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
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
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
einsum/Einsum}
stack_1Packeinsum/Einsum:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2	
stack_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape
	Reshape_1Reshapestack_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1y
stack_2PackReshape_1:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2	
stack_2w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_2/shape?
	Reshape_2Reshapestack_2:output:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_2j
IdentityIdentityReshape_2:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????:?????????:N J
+
_output_shapes
:?????????

_user_specified_namex:NJ
+
_output_shapes
:?????????

_user_specified_namex
?
H
,__inference_activation_2_layer_call_fn_14768

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
G__inference_activation_2_layer_call_and_return_conditional_losses_139262
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
?
?
__inference_loss_fn_3_14939?
;dense_4_bias_regularizer_square_readvariableop_dense_4_bias
identity??.dense_4/bias/Regularizer/Square/ReadVariableOp?
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_4_bias_regularizer_square_readvariableop_dense_4_bias*
_output_shapes
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
?
I
2__inference_compute_aux_inputs_layer_call_fn_14688
x
identity?
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_138292
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:N J
+
_output_shapes
:?????????

_user_specified_namex
?
d
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_14683
x
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_sliceh
SquareSquarestrided_slice:output:0*
T0*+
_output_shapes
:?????????2
Squarew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         2
Reshape/shape{
ReshapeReshape
Square:y:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicesy
SumSumReshape:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2
Sumd
IdentityIdentitySum:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
F__inference_dense_3_layer_call_and_return_all_conditional_losses_14758

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
B__inference_dense_3_layer_call_and_return_conditional_losses_138792
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
.__inference_dense_3_activity_regularizer_139052
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
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?]
?
B__inference_model_4_layer_call_and_return_conditional_losses_14158
input_3
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
"compute_aux_inputs/PartitionedCallPartitionedCallinput_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_138292$
"compute_aux_inputs/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall+compute_aux_inputs/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_138792!
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
.__inference_dense_3_activity_regularizer_139052-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_139262
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_139762!
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
.__inference_dense_4_activity_regularizer_140022-
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_140692
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
__inference_loss_fn_2_14928C
?dense_4_kernel_regularizer_square_readvariableop_dense_4_kernel
identity??0dense_4/kernel/Regularizer/Square/ReadVariableOp?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_4_kernel_regularizer_square_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
?
d
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_13829
x
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_sliceh
SquareSquarestrided_slice:output:0*
T0*+
_output_shapes
:?????????2
Squarew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         2
Reshape/shape{
ReshapeReshape
Square:y:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicesy
SumSumReshape:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2
Sumd
IdentityIdentitySum:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:N J
+
_output_shapes
:?????????

_user_specified_namex
?]
?
B__inference_model_4_layer_call_and_return_conditional_losses_14215

inputs
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
"compute_aux_inputs/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_138292$
"compute_aux_inputs/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall+compute_aux_inputs/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_138792!
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
.__inference_dense_3_activity_regularizer_139052-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_139262
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_139762!
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
.__inference_dense_4_activity_regularizer_140022-
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_140692
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
.__inference_dense_4_activity_regularizer_13812
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
'__inference_model_4_layer_call_fn_14224
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
.:?????????:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_142152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?]
?
B__inference_model_4_layer_call_and_return_conditional_losses_14104
input_3
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
"compute_aux_inputs/PartitionedCallPartitionedCallinput_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_138292$
"compute_aux_inputs/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall+compute_aux_inputs/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_138792!
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
.__inference_dense_3_activity_regularizer_139052-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_139262
activation_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_139762!
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
.__inference_dense_4_activity_regularizer_140022-
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_linear_update_layer_call_and_return_conditional_losses_140692
linear_update/PartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?6
j
H__inference_linear_update_layer_call_and_return_conditional_losses_14889
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
:?????????*

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
:?????????*

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
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
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
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
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
einsum/Einsum}
stack_1Packeinsum/Einsum:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2	
stack_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape
	Reshape_1Reshapestack_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1y
stack_2PackReshape_1:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2	
stack_2w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape_2/shape?
	Reshape_2Reshapestack_2:output:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
	Reshape_2j
IdentityIdentityReshape_2:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????:?????????:P L
+
_output_shapes
:?????????

_user_specified_namex/0:PL
+
_output_shapes
:?????????

_user_specified_namex/1
?
?
__inference_loss_fn_1_14917?
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
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_14763

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
??
?
B__inference_model_4_layer_call_and_return_conditional_losses_14487

inputs3
/dense_3_tensordot_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias3
/dense_4_tensordot_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
&compute_aux_inputs/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2(
&compute_aux_inputs/strided_slice/stack?
(compute_aux_inputs/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2*
(compute_aux_inputs/strided_slice/stack_1?
(compute_aux_inputs/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2*
(compute_aux_inputs/strided_slice/stack_2?
 compute_aux_inputs/strided_sliceStridedSliceinputs/compute_aux_inputs/strided_slice/stack:output:01compute_aux_inputs/strided_slice/stack_1:output:01compute_aux_inputs/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2"
 compute_aux_inputs/strided_slice?
compute_aux_inputs/SquareSquare)compute_aux_inputs/strided_slice:output:0*
T0*+
_output_shapes
:?????????2
compute_aux_inputs/Square?
 compute_aux_inputs/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         2"
 compute_aux_inputs/Reshape/shape?
compute_aux_inputs/ReshapeReshapecompute_aux_inputs/Square:y:0)compute_aux_inputs/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
compute_aux_inputs/Reshape?
(compute_aux_inputs/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(compute_aux_inputs/Sum/reduction_indices?
compute_aux_inputs/SumSum#compute_aux_inputs/Reshape:output:01compute_aux_inputs/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2
compute_aux_inputs/Sum?
 dense_3/Tensordot/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
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
dense_3/Tensordot/free?
dense_3/Tensordot/ShapeShapecompute_aux_inputs/Sum:output:0*
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
dense_3/Tensordot/transpose	Transposecompute_aux_inputs/Sum:output:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
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
:	?*
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
:?????????2
dense_4/Tensordot/MatMul?
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
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
:?????????2
dense_4/Tensordot?
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_4/BiasAdd?
"dense_4/ActivityRegularizer/SquareSquaredense_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
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
:?????????*

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
:?????????*

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
valueB"       2%
#linear_update/strided_slice_3/stack?
%linear_update/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
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
valueB"       2%
#linear_update/strided_slice_4/stack?
%linear_update/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
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
linear_update/stack_1Pack$linear_update/einsum/Einsum:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
linear_update/stack_1?
linear_update/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
linear_update/Reshape_1/shape?
linear_update/Reshape_1Reshapelinear_update/stack_1:output:0&linear_update/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
linear_update/Reshape_1?
linear_update/stack_2Pack linear_update/Reshape_1:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
linear_update/stack_2?
linear_update/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
linear_update/Reshape_2/shape?
linear_update/Reshape_2Reshapelinear_update/stack_2:output:0&linear_update/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2
linear_update/Reshape_2?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:	?*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOp?
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
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
:?????????2

Identity?

Identity_1Identity linear_update/Reshape_2:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::2@
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
:?????????
 
_user_specified_nameinputs
?3
?
B__inference_dense_3_layer_call_and_return_conditional_losses_14742

inputs+
'tensordot_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
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
:?????????2
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
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_14906C
?dense_3_kernel_regularizer_square_readvariableop_dense_3_kernel
identity??0dense_3/kernel/Regularizer/Square/ReadVariableOp?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_3_kernel_regularizer_square_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2#
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
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp"?L
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
serving_default_input_3:0??????????
dense_44
StatefulPartitionedCall:0?????????E
linear_update4
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?.
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
C__call__
D_default_save_signature
*E&call_and_return_all_conditional_losses"?,
_tf_keras_network?,{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "compute_aux_inputs", "config": {"name": "compute_aux_inputs", "trainable": true, "dtype": "float32", "num_complex": 1, "num_real": 0}, "name": "compute_aux_inputs", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["compute_aux_inputs", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "linear_update", "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 2]}, "kernels": [], "num_complex": 1, "num_real": 0, "dt": 0.05}, "name": "linear_update", "inbound_nodes": [[["input_3", 0, 0, {}], ["dense_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_4", 0, 0], ["linear_update", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "compute_aux_inputs", "config": {"name": "compute_aux_inputs", "trainable": true, "dtype": "float32", "num_complex": 1, "num_real": 0}, "name": "compute_aux_inputs", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["compute_aux_inputs", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "linear_update", "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 2]}, "kernels": [], "num_complex": 1, "num_real": 0, "dt": 0.05}, "name": "linear_update", "inbound_nodes": [[["input_3", 0, 0, {}], ["dense_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense_4", 0, 0], ["linear_update", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "compute_aux_inputs", "name": "compute_aux_inputs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "compute_aux_inputs", "trainable": true, "dtype": "float32", "num_complex": 1, "num_real": 0}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 2]}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 170}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 170]}}
?
 kernels
!regularization_losses
"	variables
#trainable_variables
$	keras_api
N__call__
*O&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "linear_update", "name": "linear_update", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 2]}, "kernels": [], "num_complex": 1, "num_real": 0, "dt": 0.05}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 2]}, {"class_name": "TensorShape", "items": [null, 1, 2]}]}
<
P0
Q1
R2
S3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
regularization_losses
	variables
%metrics
	trainable_variables
&layer_regularization_losses
'layer_metrics

(layers
)non_trainable_variables
C__call__
D_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Tserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables
*metrics
trainable_variables
+layer_regularization_losses
,layer_metrics

-layers
.non_trainable_variables
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_3/kernel
:?2dense_3/bias
.
P0
Q1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
/metrics
trainable_variables
0layer_regularization_losses
1layer_metrics

2layers
3non_trainable_variables
H__call__
Uactivity_regularizer_fn
*I&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables
4metrics
trainable_variables
5layer_regularization_losses
6layer_metrics

7layers
8non_trainable_variables
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_4/kernel
:2dense_4/bias
.
R0
S1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
9metrics
trainable_variables
:layer_regularization_losses
;layer_metrics

<layers
=non_trainable_variables
L__call__
Wactivity_regularizer_fn
*M&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
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
!regularization_losses
"	variables
>metrics
#trainable_variables
?layer_regularization_losses
@layer_metrics

Alayers
Bnon_trainable_variables
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
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
P0
Q1"
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
R0
S1"
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
'__inference_model_4_layer_call_fn_14670
'__inference_model_4_layer_call_fn_14289
'__inference_model_4_layer_call_fn_14224
'__inference_model_4_layer_call_fn_14659?
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
 __inference__wrapped_model_13786?
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
input_3?????????
?2?
B__inference_model_4_layer_call_and_return_conditional_losses_14487
B__inference_model_4_layer_call_and_return_conditional_losses_14648
B__inference_model_4_layer_call_and_return_conditional_losses_14158
B__inference_model_4_layer_call_and_return_conditional_losses_14104?
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
2__inference_compute_aux_inputs_layer_call_fn_14688?
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
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_14683?
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
'__inference_dense_3_layer_call_fn_14749?
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
F__inference_dense_3_layer_call_and_return_all_conditional_losses_14758?
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
,__inference_activation_2_layer_call_fn_14768?
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
G__inference_activation_2_layer_call_and_return_conditional_losses_14763?
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
'__inference_dense_4_layer_call_fn_14829?
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
F__inference_dense_4_layer_call_and_return_all_conditional_losses_14838?
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
-__inference_linear_update_layer_call_fn_14895?
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
H__inference_linear_update_layer_call_and_return_conditional_losses_14889?
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
__inference_loss_fn_0_14906?
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
__inference_loss_fn_1_14917?
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
__inference_loss_fn_2_14928?
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
__inference_loss_fn_3_14939?
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
#__inference_signature_wrapper_14326input_3"?
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
.__inference_dense_3_activity_regularizer_13799?
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
B__inference_dense_3_layer_call_and_return_conditional_losses_14742?
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
.__inference_dense_4_activity_regularizer_13812?
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
B__inference_dense_4_layer_call_and_return_conditional_losses_14822?
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
 __inference__wrapped_model_13786?4?1
*?'
%?"
input_3?????????
? "s?p
0
dense_4%?"
dense_4?????????
<
linear_update+?(
linear_update??????????
G__inference_activation_2_layer_call_and_return_conditional_losses_14763b4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_activation_2_layer_call_fn_14768U4?1
*?'
%?"
inputs??????????
? "????????????
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_14683[.?+
$?!
?
x?????????
? ")?&
?
0?????????
? ?
2__inference_compute_aux_inputs_layer_call_fn_14688N.?+
$?!
?
x?????????
? "??????????[
.__inference_dense_3_activity_regularizer_13799)?
?
?
self
? "? ?
F__inference_dense_3_layer_call_and_return_all_conditional_losses_14758s3?0
)?&
$?!
inputs?????????
? "8?5
 ?
0??????????
?
?	
1/0 ?
B__inference_dense_3_layer_call_and_return_conditional_losses_14742e3?0
)?&
$?!
inputs?????????
? "*?'
 ?
0??????????
? ?
'__inference_dense_3_layer_call_fn_14749X3?0
)?&
$?!
inputs?????????
? "???????????[
.__inference_dense_4_activity_regularizer_13812)?
?
?
self
? "? ?
F__inference_dense_4_layer_call_and_return_all_conditional_losses_14838s4?1
*?'
%?"
inputs??????????
? "7?4
?
0?????????
?
?	
1/0 ?
B__inference_dense_4_layer_call_and_return_conditional_losses_14822e4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
'__inference_dense_4_layer_call_fn_14829X4?1
*?'
%?"
inputs??????????
? "???????????
H__inference_linear_update_layer_call_and_return_conditional_losses_14889?X?U
N?K
I?F
!?
x/0?????????
!?
x/1?????????
? ")?&
?
0?????????
? ?
-__inference_linear_update_layer_call_fn_14895xX?U
N?K
I?F
!?
x/0?????????
!?
x/1?????????
? "??????????:
__inference_loss_fn_0_14906?

? 
? "? :
__inference_loss_fn_1_14917?

? 
? "? :
__inference_loss_fn_2_14928?

? 
? "? :
__inference_loss_fn_3_14939?

? 
? "? ?
B__inference_model_4_layer_call_and_return_conditional_losses_14104?<?9
2?/
%?"
input_3?????????
p

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_14158?<?9
2?/
%?"
input_3?????????
p 

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_14487?;?8
1?.
$?!
inputs?????????
p

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_14648?;?8
1?.
$?!
inputs?????????
p 

 
? "S?P
I?F
!?
0/0?????????
!?
0/1?????????
? ?
'__inference_model_4_layer_call_fn_14224?<?9
2?/
%?"
input_3?????????
p

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_4_layer_call_fn_14289?<?9
2?/
%?"
input_3?????????
p 

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_4_layer_call_fn_14659?;?8
1?.
$?!
inputs?????????
p

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_4_layer_call_fn_14670?;?8
1?.
$?!
inputs?????????
p 

 
? "E?B
?
0?????????
?
1??????????
#__inference_signature_wrapper_14326???<
? 
5?2
0
input_3%?"
input_3?????????"s?p
0
dense_4%?"
dense_4?????????
<
linear_update+?(
linear_update?????????