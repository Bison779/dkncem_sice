??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:P*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:P*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:PP*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:P*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:P*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
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
?'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
R
:regularization_losses
;	variables
<trainable_variables
=	keras_api
h

>kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
_
Dkernels
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
 
F
0
1
 2
!3
*4
+5
46
57
>8
?9
F
0
1
 2
!3
*4
+5
46
57
>8
?9
?
regularization_losses
	variables
Imetrics
trainable_variables
Jlayer_regularization_losses
Klayer_metrics

Llayers
Mnon_trainable_variables
 
 
 
 
?
regularization_losses
	variables
Nmetrics
trainable_variables
Olayer_regularization_losses
Player_metrics

Qlayers
Rnon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables
Smetrics
trainable_variables
Tlayer_regularization_losses
Ulayer_metrics

Vlayers
Wnon_trainable_variables
 
 
 
?
regularization_losses
	variables
Xmetrics
trainable_variables
Ylayer_regularization_losses
Zlayer_metrics

[layers
\non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
?
"regularization_losses
#	variables
]metrics
$trainable_variables
^layer_regularization_losses
_layer_metrics

`layers
anon_trainable_variables
 
 
 
?
&regularization_losses
'	variables
bmetrics
(trainable_variables
clayer_regularization_losses
dlayer_metrics

elayers
fnon_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
?
,regularization_losses
-	variables
gmetrics
.trainable_variables
hlayer_regularization_losses
ilayer_metrics

jlayers
knon_trainable_variables
 
 
 
?
0regularization_losses
1	variables
lmetrics
2trainable_variables
mlayer_regularization_losses
nlayer_metrics

olayers
pnon_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?
6regularization_losses
7	variables
qmetrics
8trainable_variables
rlayer_regularization_losses
slayer_metrics

tlayers
unon_trainable_variables
 
 
 
?
:regularization_losses
;	variables
vmetrics
<trainable_variables
wlayer_regularization_losses
xlayer_metrics

ylayers
znon_trainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

>0
?1
?
@regularization_losses
A	variables
{metrics
Btrainable_variables
|layer_regularization_losses
}layer_metrics

~layers
non_trainable_variables
 
 
 
 
?
Eregularization_losses
F	variables
?metrics
Gtrainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?non_trainable_variables
 
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
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
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_12278
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_13593
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
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
!__inference__traced_restore_13633??
?
?
'__inference_dense_3_layer_call_fn_13283

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
B__inference_dense_3_layer_call_and_return_conditional_losses_115252
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
?3
?
B__inference_dense_1_layer_call_and_return_conditional_losses_11323

inputs+
'tensordot_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_1_kernel*
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
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
H
.__inference_dense_1_activity_regularizer_11349
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
B__inference_dense_3_layer_call_and_return_conditional_losses_11525

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
?2
?
@__inference_dense_layer_call_and_return_conditional_losses_11226

inputs)
%tensordot_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp%tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
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
:?????????2
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
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_13115

inputs
dense_1_kernel
dense_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_113232
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
?6
j
H__inference_linear_update_layer_call_and_return_conditional_losses_13423
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
?
F
,__inference_dense_activity_regularizer_11107
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
F
,__inference_dense_activity_regularizer_11252
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
?
'__inference_dense_2_layer_call_fn_13195

inputs
dense_2_kernel
dense_2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_114202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
H
.__inference_dense_3_activity_regularizer_11551
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
?
'__inference_model_1_layer_call_fn_12056
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_120412
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

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
F
*__inference_activation_layer_call_fn_13054

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
GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_112732
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
?6
h
H__inference_linear_update_layer_call_and_return_conditional_losses_11715
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
ğ
?
 __inference__wrapped_model_11094
input_17
3model_1_dense_tensordot_readvariableop_dense_kernel3
/model_1_dense_biasadd_readvariableop_dense_bias;
7model_1_dense_1_tensordot_readvariableop_dense_1_kernel7
3model_1_dense_1_biasadd_readvariableop_dense_1_bias;
7model_1_dense_2_tensordot_readvariableop_dense_2_kernel7
3model_1_dense_2_biasadd_readvariableop_dense_2_bias;
7model_1_dense_3_tensordot_readvariableop_dense_3_kernel7
3model_1_dense_3_biasadd_readvariableop_dense_3_bias;
7model_1_dense_4_tensordot_readvariableop_dense_4_kernel7
3model_1_dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??$model_1/dense/BiasAdd/ReadVariableOp?&model_1/dense/Tensordot/ReadVariableOp?&model_1/dense_1/BiasAdd/ReadVariableOp?(model_1/dense_1/Tensordot/ReadVariableOp?&model_1/dense_2/BiasAdd/ReadVariableOp?(model_1/dense_2/Tensordot/ReadVariableOp?&model_1/dense_3/BiasAdd/ReadVariableOp?(model_1/dense_3/Tensordot/ReadVariableOp?&model_1/dense_4/BiasAdd/ReadVariableOp?(model_1/dense_4/Tensordot/ReadVariableOpe
model_1/reshape/ShapeShapeinput_1*
T0*
_output_shapes
:2
model_1/reshape/Shape?
#model_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_1/reshape/strided_slice/stack?
%model_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_1/reshape/strided_slice/stack_1?
%model_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_1/reshape/strided_slice/stack_2?
model_1/reshape/strided_sliceStridedSlicemodel_1/reshape/Shape:output:0,model_1/reshape/strided_slice/stack:output:0.model_1/reshape/strided_slice/stack_1:output:0.model_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_1/reshape/strided_slice?
model_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
model_1/reshape/Reshape/shape/1?
model_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
model_1/reshape/Reshape/shape/2?
model_1/reshape/Reshape/shapePack&model_1/reshape/strided_slice:output:0(model_1/reshape/Reshape/shape/1:output:0(model_1/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
model_1/reshape/Reshape/shape?
model_1/reshape/ReshapeReshapeinput_1&model_1/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
model_1/reshape/Reshape?
&model_1/dense/Tensordot/ReadVariableOpReadVariableOp3model_1_dense_tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
dtype02(
&model_1/dense/Tensordot/ReadVariableOp?
model_1/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
model_1/dense/Tensordot/axes?
model_1/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
model_1/dense/Tensordot/free?
model_1/dense/Tensordot/ShapeShape model_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2
model_1/dense/Tensordot/Shape?
%model_1/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_1/dense/Tensordot/GatherV2/axis?
 model_1/dense/Tensordot/GatherV2GatherV2&model_1/dense/Tensordot/Shape:output:0%model_1/dense/Tensordot/free:output:0.model_1/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 model_1/dense/Tensordot/GatherV2?
'model_1/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense/Tensordot/GatherV2_1/axis?
"model_1/dense/Tensordot/GatherV2_1GatherV2&model_1/dense/Tensordot/Shape:output:0%model_1/dense/Tensordot/axes:output:00model_1/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_1/dense/Tensordot/GatherV2_1?
model_1/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
model_1/dense/Tensordot/Const?
model_1/dense/Tensordot/ProdProd)model_1/dense/Tensordot/GatherV2:output:0&model_1/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
model_1/dense/Tensordot/Prod?
model_1/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
model_1/dense/Tensordot/Const_1?
model_1/dense/Tensordot/Prod_1Prod+model_1/dense/Tensordot/GatherV2_1:output:0(model_1/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
model_1/dense/Tensordot/Prod_1?
#model_1/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model_1/dense/Tensordot/concat/axis?
model_1/dense/Tensordot/concatConcatV2%model_1/dense/Tensordot/free:output:0%model_1/dense/Tensordot/axes:output:0,model_1/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
model_1/dense/Tensordot/concat?
model_1/dense/Tensordot/stackPack%model_1/dense/Tensordot/Prod:output:0'model_1/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
model_1/dense/Tensordot/stack?
!model_1/dense/Tensordot/transpose	Transpose model_1/reshape/Reshape:output:0'model_1/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2#
!model_1/dense/Tensordot/transpose?
model_1/dense/Tensordot/ReshapeReshape%model_1/dense/Tensordot/transpose:y:0&model_1/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
model_1/dense/Tensordot/Reshape?
model_1/dense/Tensordot/MatMulMatMul(model_1/dense/Tensordot/Reshape:output:0.model_1/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2 
model_1/dense/Tensordot/MatMul?
model_1/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2!
model_1/dense/Tensordot/Const_2?
%model_1/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_1/dense/Tensordot/concat_1/axis?
 model_1/dense/Tensordot/concat_1ConcatV2)model_1/dense/Tensordot/GatherV2:output:0(model_1/dense/Tensordot/Const_2:output:0.model_1/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_1/dense/Tensordot/concat_1?
model_1/dense/TensordotReshape(model_1/dense/Tensordot/MatMul:product:0)model_1/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
model_1/dense/Tensordot?
$model_1/dense/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02&
$model_1/dense/BiasAdd/ReadVariableOp?
model_1/dense/BiasAddBiasAdd model_1/dense/Tensordot:output:0,model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
model_1/dense/BiasAdd?
(model_1/dense/ActivityRegularizer/SquareSquaremodel_1/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2*
(model_1/dense/ActivityRegularizer/Square?
'model_1/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'model_1/dense/ActivityRegularizer/Const?
%model_1/dense/ActivityRegularizer/SumSum,model_1/dense/ActivityRegularizer/Square:y:00model_1/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2'
%model_1/dense/ActivityRegularizer/Sum?
'model_1/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2)
'model_1/dense/ActivityRegularizer/mul/x?
%model_1/dense/ActivityRegularizer/mulMul0model_1/dense/ActivityRegularizer/mul/x:output:0.model_1/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%model_1/dense/ActivityRegularizer/mul?
'model_1/dense/ActivityRegularizer/ShapeShapemodel_1/dense/BiasAdd:output:0*
T0*
_output_shapes
:2)
'model_1/dense/ActivityRegularizer/Shape?
5model_1/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5model_1/dense/ActivityRegularizer/strided_slice/stack?
7model_1/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_1/dense/ActivityRegularizer/strided_slice/stack_1?
7model_1/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_1/dense/ActivityRegularizer/strided_slice/stack_2?
/model_1/dense/ActivityRegularizer/strided_sliceStridedSlice0model_1/dense/ActivityRegularizer/Shape:output:0>model_1/dense/ActivityRegularizer/strided_slice/stack:output:0@model_1/dense/ActivityRegularizer/strided_slice/stack_1:output:0@model_1/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/model_1/dense/ActivityRegularizer/strided_slice?
&model_1/dense/ActivityRegularizer/CastCast8model_1/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&model_1/dense/ActivityRegularizer/Cast?
)model_1/dense/ActivityRegularizer/truedivRealDiv)model_1/dense/ActivityRegularizer/mul:z:0*model_1/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2+
)model_1/dense/ActivityRegularizer/truediv?
model_1/activation/ReluRelumodel_1/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
model_1/activation/Relu?
(model_1/dense_1/Tensordot/ReadVariableOpReadVariableOp7model_1_dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype02*
(model_1/dense_1/Tensordot/ReadVariableOp?
model_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_1/dense_1/Tensordot/axes?
model_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_1/dense_1/Tensordot/free?
model_1/dense_1/Tensordot/ShapeShape%model_1/activation/Relu:activations:0*
T0*
_output_shapes
:2!
model_1/dense_1/Tensordot/Shape?
'model_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_1/Tensordot/GatherV2/axis?
"model_1/dense_1/Tensordot/GatherV2GatherV2(model_1/dense_1/Tensordot/Shape:output:0'model_1/dense_1/Tensordot/free:output:00model_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_1/dense_1/Tensordot/GatherV2?
)model_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_1/dense_1/Tensordot/GatherV2_1/axis?
$model_1/dense_1/Tensordot/GatherV2_1GatherV2(model_1/dense_1/Tensordot/Shape:output:0'model_1/dense_1/Tensordot/axes:output:02model_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_1/dense_1/Tensordot/GatherV2_1?
model_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_1/dense_1/Tensordot/Const?
model_1/dense_1/Tensordot/ProdProd+model_1/dense_1/Tensordot/GatherV2:output:0(model_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_1/dense_1/Tensordot/Prod?
!model_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_1/dense_1/Tensordot/Const_1?
 model_1/dense_1/Tensordot/Prod_1Prod-model_1/dense_1/Tensordot/GatherV2_1:output:0*model_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_1/dense_1/Tensordot/Prod_1?
%model_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_1/dense_1/Tensordot/concat/axis?
 model_1/dense_1/Tensordot/concatConcatV2'model_1/dense_1/Tensordot/free:output:0'model_1/dense_1/Tensordot/axes:output:0.model_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_1/dense_1/Tensordot/concat?
model_1/dense_1/Tensordot/stackPack'model_1/dense_1/Tensordot/Prod:output:0)model_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_1/dense_1/Tensordot/stack?
#model_1/dense_1/Tensordot/transpose	Transpose%model_1/activation/Relu:activations:0)model_1/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2%
#model_1/dense_1/Tensordot/transpose?
!model_1/dense_1/Tensordot/ReshapeReshape'model_1/dense_1/Tensordot/transpose:y:0(model_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_1/dense_1/Tensordot/Reshape?
 model_1/dense_1/Tensordot/MatMulMatMul*model_1/dense_1/Tensordot/Reshape:output:00model_1/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2"
 model_1/dense_1/Tensordot/MatMul?
!model_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2#
!model_1/dense_1/Tensordot/Const_2?
'model_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_1/Tensordot/concat_1/axis?
"model_1/dense_1/Tensordot/concat_1ConcatV2+model_1/dense_1/Tensordot/GatherV2:output:0*model_1/dense_1/Tensordot/Const_2:output:00model_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_1/dense_1/Tensordot/concat_1?
model_1/dense_1/TensordotReshape*model_1/dense_1/Tensordot/MatMul:product:0+model_1/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
model_1/dense_1/Tensordot?
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp3model_1_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype02(
&model_1/dense_1/BiasAdd/ReadVariableOp?
model_1/dense_1/BiasAddBiasAdd"model_1/dense_1/Tensordot:output:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
model_1/dense_1/BiasAdd?
*model_1/dense_1/ActivityRegularizer/SquareSquare model_1/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2,
*model_1/dense_1/ActivityRegularizer/Square?
)model_1/dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_1/dense_1/ActivityRegularizer/Const?
'model_1/dense_1/ActivityRegularizer/SumSum.model_1/dense_1/ActivityRegularizer/Square:y:02model_1/dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_1/ActivityRegularizer/Sum?
)model_1/dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_1/dense_1/ActivityRegularizer/mul/x?
'model_1/dense_1/ActivityRegularizer/mulMul2model_1/dense_1/ActivityRegularizer/mul/x:output:00model_1/dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_1/ActivityRegularizer/mul?
)model_1/dense_1/ActivityRegularizer/ShapeShape model_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_1/dense_1/ActivityRegularizer/Shape?
7model_1/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_1/dense_1/ActivityRegularizer/strided_slice/stack?
9model_1/dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_1/ActivityRegularizer/strided_slice/stack_1?
9model_1/dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_1/ActivityRegularizer/strided_slice/stack_2?
1model_1/dense_1/ActivityRegularizer/strided_sliceStridedSlice2model_1/dense_1/ActivityRegularizer/Shape:output:0@model_1/dense_1/ActivityRegularizer/strided_slice/stack:output:0Bmodel_1/dense_1/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_1/dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_1/dense_1/ActivityRegularizer/strided_slice?
(model_1/dense_1/ActivityRegularizer/CastCast:model_1/dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_1/dense_1/ActivityRegularizer/Cast?
+model_1/dense_1/ActivityRegularizer/truedivRealDiv+model_1/dense_1/ActivityRegularizer/mul:z:0,model_1/dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_1/dense_1/ActivityRegularizer/truediv?
model_1/activation_1/ReluRelu model_1/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
model_1/activation_1/Relu?
(model_1/dense_2/Tensordot/ReadVariableOpReadVariableOp7model_1_dense_2_tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype02*
(model_1/dense_2/Tensordot/ReadVariableOp?
model_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_1/dense_2/Tensordot/axes?
model_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_1/dense_2/Tensordot/free?
model_1/dense_2/Tensordot/ShapeShape'model_1/activation_1/Relu:activations:0*
T0*
_output_shapes
:2!
model_1/dense_2/Tensordot/Shape?
'model_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_2/Tensordot/GatherV2/axis?
"model_1/dense_2/Tensordot/GatherV2GatherV2(model_1/dense_2/Tensordot/Shape:output:0'model_1/dense_2/Tensordot/free:output:00model_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_1/dense_2/Tensordot/GatherV2?
)model_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_1/dense_2/Tensordot/GatherV2_1/axis?
$model_1/dense_2/Tensordot/GatherV2_1GatherV2(model_1/dense_2/Tensordot/Shape:output:0'model_1/dense_2/Tensordot/axes:output:02model_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_1/dense_2/Tensordot/GatherV2_1?
model_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_1/dense_2/Tensordot/Const?
model_1/dense_2/Tensordot/ProdProd+model_1/dense_2/Tensordot/GatherV2:output:0(model_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_1/dense_2/Tensordot/Prod?
!model_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_1/dense_2/Tensordot/Const_1?
 model_1/dense_2/Tensordot/Prod_1Prod-model_1/dense_2/Tensordot/GatherV2_1:output:0*model_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_1/dense_2/Tensordot/Prod_1?
%model_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_1/dense_2/Tensordot/concat/axis?
 model_1/dense_2/Tensordot/concatConcatV2'model_1/dense_2/Tensordot/free:output:0'model_1/dense_2/Tensordot/axes:output:0.model_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_1/dense_2/Tensordot/concat?
model_1/dense_2/Tensordot/stackPack'model_1/dense_2/Tensordot/Prod:output:0)model_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_1/dense_2/Tensordot/stack?
#model_1/dense_2/Tensordot/transpose	Transpose'model_1/activation_1/Relu:activations:0)model_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2%
#model_1/dense_2/Tensordot/transpose?
!model_1/dense_2/Tensordot/ReshapeReshape'model_1/dense_2/Tensordot/transpose:y:0(model_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_1/dense_2/Tensordot/Reshape?
 model_1/dense_2/Tensordot/MatMulMatMul*model_1/dense_2/Tensordot/Reshape:output:00model_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 model_1/dense_2/Tensordot/MatMul?
!model_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model_1/dense_2/Tensordot/Const_2?
'model_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_2/Tensordot/concat_1/axis?
"model_1/dense_2/Tensordot/concat_1ConcatV2+model_1/dense_2/Tensordot/GatherV2:output:0*model_1/dense_2/Tensordot/Const_2:output:00model_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_1/dense_2/Tensordot/concat_1?
model_1/dense_2/TensordotReshape*model_1/dense_2/Tensordot/MatMul:product:0+model_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
model_1/dense_2/Tensordot?
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp3model_1_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02(
&model_1/dense_2/BiasAdd/ReadVariableOp?
model_1/dense_2/BiasAddBiasAdd"model_1/dense_2/Tensordot:output:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model_1/dense_2/BiasAdd?
*model_1/dense_2/ActivityRegularizer/SquareSquare model_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2,
*model_1/dense_2/ActivityRegularizer/Square?
)model_1/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_1/dense_2/ActivityRegularizer/Const?
'model_1/dense_2/ActivityRegularizer/SumSum.model_1/dense_2/ActivityRegularizer/Square:y:02model_1/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_2/ActivityRegularizer/Sum?
)model_1/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_1/dense_2/ActivityRegularizer/mul/x?
'model_1/dense_2/ActivityRegularizer/mulMul2model_1/dense_2/ActivityRegularizer/mul/x:output:00model_1/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_2/ActivityRegularizer/mul?
)model_1/dense_2/ActivityRegularizer/ShapeShape model_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_1/dense_2/ActivityRegularizer/Shape?
7model_1/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_1/dense_2/ActivityRegularizer/strided_slice/stack?
9model_1/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_2/ActivityRegularizer/strided_slice/stack_1?
9model_1/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_2/ActivityRegularizer/strided_slice/stack_2?
1model_1/dense_2/ActivityRegularizer/strided_sliceStridedSlice2model_1/dense_2/ActivityRegularizer/Shape:output:0@model_1/dense_2/ActivityRegularizer/strided_slice/stack:output:0Bmodel_1/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_1/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_1/dense_2/ActivityRegularizer/strided_slice?
(model_1/dense_2/ActivityRegularizer/CastCast:model_1/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_1/dense_2/ActivityRegularizer/Cast?
+model_1/dense_2/ActivityRegularizer/truedivRealDiv+model_1/dense_2/ActivityRegularizer/mul:z:0,model_1/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_1/dense_2/ActivityRegularizer/truediv?
.model_1/compute_aux_inputs/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model_1/compute_aux_inputs/strided_slice/stack?
0model_1/compute_aux_inputs/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model_1/compute_aux_inputs/strided_slice/stack_1?
0model_1/compute_aux_inputs/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model_1/compute_aux_inputs/strided_slice/stack_2?
(model_1/compute_aux_inputs/strided_sliceStridedSlice model_1/dense_2/BiasAdd:output:07model_1/compute_aux_inputs/strided_slice/stack:output:09model_1/compute_aux_inputs/strided_slice/stack_1:output:09model_1/compute_aux_inputs/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2*
(model_1/compute_aux_inputs/strided_slice?
!model_1/compute_aux_inputs/SquareSquare1model_1/compute_aux_inputs/strided_slice:output:0*
T0*+
_output_shapes
:?????????2#
!model_1/compute_aux_inputs/Square?
(model_1/compute_aux_inputs/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         2*
(model_1/compute_aux_inputs/Reshape/shape?
"model_1/compute_aux_inputs/ReshapeReshape%model_1/compute_aux_inputs/Square:y:01model_1/compute_aux_inputs/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2$
"model_1/compute_aux_inputs/Reshape?
0model_1/compute_aux_inputs/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0model_1/compute_aux_inputs/Sum/reduction_indices?
model_1/compute_aux_inputs/SumSum+model_1/compute_aux_inputs/Reshape:output:09model_1/compute_aux_inputs/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????2 
model_1/compute_aux_inputs/Sum?
(model_1/dense_3/Tensordot/ReadVariableOpReadVariableOp7model_1_dense_3_tensordot_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02*
(model_1/dense_3/Tensordot/ReadVariableOp?
model_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_1/dense_3/Tensordot/axes?
model_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_1/dense_3/Tensordot/free?
model_1/dense_3/Tensordot/ShapeShape'model_1/compute_aux_inputs/Sum:output:0*
T0*
_output_shapes
:2!
model_1/dense_3/Tensordot/Shape?
'model_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_3/Tensordot/GatherV2/axis?
"model_1/dense_3/Tensordot/GatherV2GatherV2(model_1/dense_3/Tensordot/Shape:output:0'model_1/dense_3/Tensordot/free:output:00model_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_1/dense_3/Tensordot/GatherV2?
)model_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_1/dense_3/Tensordot/GatherV2_1/axis?
$model_1/dense_3/Tensordot/GatherV2_1GatherV2(model_1/dense_3/Tensordot/Shape:output:0'model_1/dense_3/Tensordot/axes:output:02model_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_1/dense_3/Tensordot/GatherV2_1?
model_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_1/dense_3/Tensordot/Const?
model_1/dense_3/Tensordot/ProdProd+model_1/dense_3/Tensordot/GatherV2:output:0(model_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_1/dense_3/Tensordot/Prod?
!model_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_1/dense_3/Tensordot/Const_1?
 model_1/dense_3/Tensordot/Prod_1Prod-model_1/dense_3/Tensordot/GatherV2_1:output:0*model_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_1/dense_3/Tensordot/Prod_1?
%model_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_1/dense_3/Tensordot/concat/axis?
 model_1/dense_3/Tensordot/concatConcatV2'model_1/dense_3/Tensordot/free:output:0'model_1/dense_3/Tensordot/axes:output:0.model_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_1/dense_3/Tensordot/concat?
model_1/dense_3/Tensordot/stackPack'model_1/dense_3/Tensordot/Prod:output:0)model_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_1/dense_3/Tensordot/stack?
#model_1/dense_3/Tensordot/transpose	Transpose'model_1/compute_aux_inputs/Sum:output:0)model_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2%
#model_1/dense_3/Tensordot/transpose?
!model_1/dense_3/Tensordot/ReshapeReshape'model_1/dense_3/Tensordot/transpose:y:0(model_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_1/dense_3/Tensordot/Reshape?
 model_1/dense_3/Tensordot/MatMulMatMul*model_1/dense_3/Tensordot/Reshape:output:00model_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model_1/dense_3/Tensordot/MatMul?
!model_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2#
!model_1/dense_3/Tensordot/Const_2?
'model_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_3/Tensordot/concat_1/axis?
"model_1/dense_3/Tensordot/concat_1ConcatV2+model_1/dense_3/Tensordot/GatherV2:output:0*model_1/dense_3/Tensordot/Const_2:output:00model_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_1/dense_3/Tensordot/concat_1?
model_1/dense_3/TensordotReshape*model_1/dense_3/Tensordot/MatMul:product:0+model_1/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
model_1/dense_3/Tensordot?
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp3model_1_dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes	
:?*
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp?
model_1/dense_3/BiasAddBiasAdd"model_1/dense_3/Tensordot:output:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model_1/dense_3/BiasAdd?
*model_1/dense_3/ActivityRegularizer/SquareSquare model_1/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2,
*model_1/dense_3/ActivityRegularizer/Square?
)model_1/dense_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_1/dense_3/ActivityRegularizer/Const?
'model_1/dense_3/ActivityRegularizer/SumSum.model_1/dense_3/ActivityRegularizer/Square:y:02model_1/dense_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_3/ActivityRegularizer/Sum?
)model_1/dense_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_1/dense_3/ActivityRegularizer/mul/x?
'model_1/dense_3/ActivityRegularizer/mulMul2model_1/dense_3/ActivityRegularizer/mul/x:output:00model_1/dense_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_3/ActivityRegularizer/mul?
)model_1/dense_3/ActivityRegularizer/ShapeShape model_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_1/dense_3/ActivityRegularizer/Shape?
7model_1/dense_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_1/dense_3/ActivityRegularizer/strided_slice/stack?
9model_1/dense_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_3/ActivityRegularizer/strided_slice/stack_1?
9model_1/dense_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_3/ActivityRegularizer/strided_slice/stack_2?
1model_1/dense_3/ActivityRegularizer/strided_sliceStridedSlice2model_1/dense_3/ActivityRegularizer/Shape:output:0@model_1/dense_3/ActivityRegularizer/strided_slice/stack:output:0Bmodel_1/dense_3/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_1/dense_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_1/dense_3/ActivityRegularizer/strided_slice?
(model_1/dense_3/ActivityRegularizer/CastCast:model_1/dense_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_1/dense_3/ActivityRegularizer/Cast?
+model_1/dense_3/ActivityRegularizer/truedivRealDiv+model_1/dense_3/ActivityRegularizer/mul:z:0,model_1/dense_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_1/dense_3/ActivityRegularizer/truediv?
model_1/activation_2/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model_1/activation_2/Relu?
(model_1/dense_4/Tensordot/ReadVariableOpReadVariableOp7model_1_dense_4_tensordot_readvariableop_dense_4_kernel*
_output_shapes
:	?*
dtype02*
(model_1/dense_4/Tensordot/ReadVariableOp?
model_1/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_1/dense_4/Tensordot/axes?
model_1/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_1/dense_4/Tensordot/free?
model_1/dense_4/Tensordot/ShapeShape'model_1/activation_2/Relu:activations:0*
T0*
_output_shapes
:2!
model_1/dense_4/Tensordot/Shape?
'model_1/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_4/Tensordot/GatherV2/axis?
"model_1/dense_4/Tensordot/GatherV2GatherV2(model_1/dense_4/Tensordot/Shape:output:0'model_1/dense_4/Tensordot/free:output:00model_1/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_1/dense_4/Tensordot/GatherV2?
)model_1/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_1/dense_4/Tensordot/GatherV2_1/axis?
$model_1/dense_4/Tensordot/GatherV2_1GatherV2(model_1/dense_4/Tensordot/Shape:output:0'model_1/dense_4/Tensordot/axes:output:02model_1/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_1/dense_4/Tensordot/GatherV2_1?
model_1/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_1/dense_4/Tensordot/Const?
model_1/dense_4/Tensordot/ProdProd+model_1/dense_4/Tensordot/GatherV2:output:0(model_1/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_1/dense_4/Tensordot/Prod?
!model_1/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_1/dense_4/Tensordot/Const_1?
 model_1/dense_4/Tensordot/Prod_1Prod-model_1/dense_4/Tensordot/GatherV2_1:output:0*model_1/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_1/dense_4/Tensordot/Prod_1?
%model_1/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_1/dense_4/Tensordot/concat/axis?
 model_1/dense_4/Tensordot/concatConcatV2'model_1/dense_4/Tensordot/free:output:0'model_1/dense_4/Tensordot/axes:output:0.model_1/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_1/dense_4/Tensordot/concat?
model_1/dense_4/Tensordot/stackPack'model_1/dense_4/Tensordot/Prod:output:0)model_1/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_1/dense_4/Tensordot/stack?
#model_1/dense_4/Tensordot/transpose	Transpose'model_1/activation_2/Relu:activations:0)model_1/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2%
#model_1/dense_4/Tensordot/transpose?
!model_1/dense_4/Tensordot/ReshapeReshape'model_1/dense_4/Tensordot/transpose:y:0(model_1/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_1/dense_4/Tensordot/Reshape?
 model_1/dense_4/Tensordot/MatMulMatMul*model_1/dense_4/Tensordot/Reshape:output:00model_1/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 model_1/dense_4/Tensordot/MatMul?
!model_1/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model_1/dense_4/Tensordot/Const_2?
'model_1/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_4/Tensordot/concat_1/axis?
"model_1/dense_4/Tensordot/concat_1ConcatV2+model_1/dense_4/Tensordot/GatherV2:output:0*model_1/dense_4/Tensordot/Const_2:output:00model_1/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_1/dense_4/Tensordot/concat_1?
model_1/dense_4/TensordotReshape*model_1/dense_4/Tensordot/MatMul:product:0+model_1/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
model_1/dense_4/Tensordot?
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp3model_1_dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:*
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOp?
model_1/dense_4/BiasAddBiasAdd"model_1/dense_4/Tensordot:output:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model_1/dense_4/BiasAdd?
*model_1/dense_4/ActivityRegularizer/SquareSquare model_1/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2,
*model_1/dense_4/ActivityRegularizer/Square?
)model_1/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)model_1/dense_4/ActivityRegularizer/Const?
'model_1/dense_4/ActivityRegularizer/SumSum.model_1/dense_4/ActivityRegularizer/Square:y:02model_1/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_4/ActivityRegularizer/Sum?
)model_1/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2+
)model_1/dense_4/ActivityRegularizer/mul/x?
'model_1/dense_4/ActivityRegularizer/mulMul2model_1/dense_4/ActivityRegularizer/mul/x:output:00model_1/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_4/ActivityRegularizer/mul?
)model_1/dense_4/ActivityRegularizer/ShapeShape model_1/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:2+
)model_1/dense_4/ActivityRegularizer/Shape?
7model_1/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_1/dense_4/ActivityRegularizer/strided_slice/stack?
9model_1/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_4/ActivityRegularizer/strided_slice/stack_1?
9model_1/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_4/ActivityRegularizer/strided_slice/stack_2?
1model_1/dense_4/ActivityRegularizer/strided_sliceStridedSlice2model_1/dense_4/ActivityRegularizer/Shape:output:0@model_1/dense_4/ActivityRegularizer/strided_slice/stack:output:0Bmodel_1/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_1/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_1/dense_4/ActivityRegularizer/strided_slice?
(model_1/dense_4/ActivityRegularizer/CastCast:model_1/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_1/dense_4/ActivityRegularizer/Cast?
+model_1/dense_4/ActivityRegularizer/truedivRealDiv+model_1/dense_4/ActivityRegularizer/mul:z:0,model_1/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_1/dense_4/ActivityRegularizer/truediv?
)model_1/linear_update/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)model_1/linear_update/strided_slice/stack?
+model_1/linear_update/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+model_1/linear_update/strided_slice/stack_1?
+model_1/linear_update/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+model_1/linear_update/strided_slice/stack_2?
#model_1/linear_update/strided_sliceStridedSlice model_1/dense_2/BiasAdd:output:02model_1/linear_update/strided_slice/stack:output:04model_1/linear_update/strided_slice/stack_1:output:04model_1/linear_update/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#model_1/linear_update/strided_slice?
+model_1/linear_update/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+model_1/linear_update/strided_slice_1/stack?
-model_1/linear_update/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_1/linear_update/strided_slice_1/stack_1?
-model_1/linear_update/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_1/linear_update/strided_slice_1/stack_2?
%model_1/linear_update/strided_slice_1StridedSlice model_1/dense_4/BiasAdd:output:04model_1/linear_update/strided_slice_1/stack:output:06model_1/linear_update/strided_slice_1/stack_1:output:06model_1/linear_update/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_1/linear_update/strided_slice_1?
+model_1/linear_update/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+model_1/linear_update/strided_slice_2/stack?
-model_1/linear_update/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_1/linear_update/strided_slice_2/stack_1?
-model_1/linear_update/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_1/linear_update/strided_slice_2/stack_2?
%model_1/linear_update/strided_slice_2StridedSlice.model_1/linear_update/strided_slice_1:output:04model_1/linear_update/strided_slice_2/stack:output:06model_1/linear_update/strided_slice_2/stack_1:output:06model_1/linear_update/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_1/linear_update/strided_slice_2
model_1/linear_update/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_1/linear_update/mul/y?
model_1/linear_update/mulMul.model_1/linear_update/strided_slice_2:output:0$model_1/linear_update/mul/y:output:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/mul?
model_1/linear_update/ExpExpmodel_1/linear_update/mul:z:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/Exp?
+model_1/linear_update/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+model_1/linear_update/strided_slice_3/stack?
-model_1/linear_update/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_1/linear_update/strided_slice_3/stack_1?
-model_1/linear_update/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_1/linear_update/strided_slice_3/stack_2?
%model_1/linear_update/strided_slice_3StridedSlice.model_1/linear_update/strided_slice_1:output:04model_1/linear_update/strided_slice_3/stack:output:06model_1/linear_update/strided_slice_3/stack_1:output:06model_1/linear_update/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_1/linear_update/strided_slice_3?
model_1/linear_update/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_1/linear_update/mul_1/y?
model_1/linear_update/mul_1Mul.model_1/linear_update/strided_slice_3:output:0&model_1/linear_update/mul_1/y:output:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/mul_1?
model_1/linear_update/CosCosmodel_1/linear_update/mul_1:z:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/Cos?
+model_1/linear_update/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+model_1/linear_update/strided_slice_4/stack?
-model_1/linear_update/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_1/linear_update/strided_slice_4/stack_1?
-model_1/linear_update/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_1/linear_update/strided_slice_4/stack_2?
%model_1/linear_update/strided_slice_4StridedSlice.model_1/linear_update/strided_slice_1:output:04model_1/linear_update/strided_slice_4/stack:output:06model_1/linear_update/strided_slice_4/stack_1:output:06model_1/linear_update/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2'
%model_1/linear_update/strided_slice_4?
model_1/linear_update/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
model_1/linear_update/mul_2/y?
model_1/linear_update/mul_2Mul.model_1/linear_update/strided_slice_4:output:0&model_1/linear_update/mul_2/y:output:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/mul_2?
model_1/linear_update/SinSinmodel_1/linear_update/mul_2:z:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/Sin?
model_1/linear_update/Mul_3Mulmodel_1/linear_update/Exp:y:0model_1/linear_update/Cos:y:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/Mul_3?
model_1/linear_update/Mul_4Mulmodel_1/linear_update/Exp:y:0model_1/linear_update/Sin:y:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/Mul_4?
model_1/linear_update/NegNegmodel_1/linear_update/Mul_4:z:0*
T0*#
_output_shapes
:?????????2
model_1/linear_update/Neg?
model_1/linear_update/stackPackmodel_1/linear_update/Mul_3:z:0model_1/linear_update/Neg:y:0model_1/linear_update/Mul_4:z:0model_1/linear_update/Mul_3:z:0*
N*
T0*'
_output_shapes
:?????????*

axis2
model_1/linear_update/stack?
#model_1/linear_update/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2%
#model_1/linear_update/Reshape/shape?
model_1/linear_update/ReshapeReshape$model_1/linear_update/stack:output:0,model_1/linear_update/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
model_1/linear_update/Reshape?
+model_1/linear_update/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+model_1/linear_update/strided_slice_5/stack?
-model_1/linear_update/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_1/linear_update/strided_slice_5/stack_1?
-model_1/linear_update/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model_1/linear_update/strided_slice_5/stack_2?
%model_1/linear_update/strided_slice_5StridedSlice,model_1/linear_update/strided_slice:output:04model_1/linear_update/strided_slice_5/stack:output:06model_1/linear_update/strided_slice_5/stack_1:output:06model_1/linear_update/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2'
%model_1/linear_update/strided_slice_5?
#model_1/linear_update/einsum/EinsumEinsum.model_1/linear_update/strided_slice_5:output:0&model_1/linear_update/Reshape:output:0*
N*
T0*'
_output_shapes
:?????????*
equation
ik,ikj->ij2%
#model_1/linear_update/einsum/Einsum?
model_1/linear_update/stack_1Pack,model_1/linear_update/einsum/Einsum:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
model_1/linear_update/stack_1?
%model_1/linear_update/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2'
%model_1/linear_update/Reshape_1/shape?
model_1/linear_update/Reshape_1Reshape&model_1/linear_update/stack_1:output:0.model_1/linear_update/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2!
model_1/linear_update/Reshape_1?
model_1/linear_update/stack_2Pack(model_1/linear_update/Reshape_1:output:0*
N*
T0*+
_output_shapes
:?????????*

axis2
model_1/linear_update/stack_2?
%model_1/linear_update/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2'
%model_1/linear_update/Reshape_2/shape?
model_1/linear_update/Reshape_2Reshape&model_1/linear_update/stack_2:output:0.model_1/linear_update/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????2!
model_1/linear_update/Reshape_2?
IdentityIdentity model_1/dense_4/BiasAdd:output:0%^model_1/dense/BiasAdd/ReadVariableOp'^model_1/dense/Tensordot/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp)^model_1/dense_1/Tensordot/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp)^model_1/dense_2/Tensordot/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp)^model_1/dense_3/Tensordot/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp)^model_1/dense_4/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity(model_1/linear_update/Reshape_2:output:0%^model_1/dense/BiasAdd/ReadVariableOp'^model_1/dense/Tensordot/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp)^model_1/dense_1/Tensordot/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp)^model_1/dense_2/Tensordot/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp)^model_1/dense_3/Tensordot/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp)^model_1/dense_4/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::2L
$model_1/dense/BiasAdd/ReadVariableOp$model_1/dense/BiasAdd/ReadVariableOp2P
&model_1/dense/Tensordot/ReadVariableOp&model_1/dense/Tensordot/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2T
(model_1/dense_1/Tensordot/ReadVariableOp(model_1/dense_1/Tensordot/ReadVariableOp2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2T
(model_1/dense_2/Tensordot/ReadVariableOp(model_1/dense_2/Tensordot/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2T
(model_1/dense_3/Tensordot/ReadVariableOp(model_1/dense_3/Tensordot/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2T
(model_1/dense_4/Tensordot/ReadVariableOp(model_1/dense_4/Tensordot/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
a
E__inference_activation_layer_call_and_return_conditional_losses_13049

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
?3
?
B__inference_dense_2_layer_call_and_return_conditional_losses_11420

inputs+
'tensordot_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
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
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
H
.__inference_dense_2_activity_regularizer_11133
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
G__inference_activation_1_layer_call_and_return_conditional_losses_13129

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
__inference_loss_fn_8_13528C
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
?2
?
@__inference_dense_layer_call_and_return_conditional_losses_13028

inputs)
%tensordot_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp%tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
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
:?????????2
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
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_11176

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
?
H
,__inference_activation_2_layer_call_fn_13302

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
G__inference_activation_2_layer_call_and_return_conditional_losses_115722
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
?!
?
__inference__traced_save_13593
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*j
_input_shapesY
W: :P:P:PP:P:P::	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::

_output_shapes
: 
?
?
F__inference_dense_2_layer_call_and_return_all_conditional_losses_13204

inputs
dense_2_kernel
dense_2_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_114202
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
.__inference_dense_2_activity_regularizer_114462
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
__inference_loss_fn_9_13539?
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
?-
?
!__inference__traced_restore_13633
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias%
!assignvariableop_8_dense_4_kernel#
assignvariableop_9_dense_4_bias
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?3
?
B__inference_dense_2_layer_call_and_return_conditional_losses_13188

inputs+
'tensordot_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
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
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_11273

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
H
.__inference_dense_4_activity_regularizer_11648
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
.__inference_dense_4_activity_regularizer_11159
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
%__inference_dense_layer_call_fn_13035

inputs
dense_kernel

dense_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_112262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
B__inference_dense_4_layer_call_and_return_conditional_losses_13356

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
?
?
__inference_loss_fn_7_13517?
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
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_12041

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
reshape/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_111762
reshape/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_112262
dense/StatefulPartitionedCall?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *5
f0R.
,__inference_dense_activity_regularizer_112522+
)dense/ActivityRegularizer/PartitionedCall?
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape?
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack?
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1?
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_112732
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_113232!
dense_1/StatefulPartitionedCall?
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
.__inference_dense_1_activity_regularizer_113492-
+dense_1/ActivityRegularizer/PartitionedCall?
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shape?
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack?
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1?
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truediv?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_113702
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_114202!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
.__inference_dense_2_activity_regularizer_114462-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
"compute_aux_inputs/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_114752$
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
B__inference_dense_3_layer_call_and_return_conditional_losses_115252!
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
.__inference_dense_3_activity_regularizer_115512-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_115722
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
B__inference_dense_4_layer_call_and_return_conditional_losses_116222!
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
.__inference_dense_4_activity_regularizer_116482-
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
linear_update/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
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
H__inference_linear_update_layer_call_and_return_conditional_losses_117152
linear_update/PartitionedCall?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
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
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
-__inference_linear_update_layer_call_fn_13429
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
H__inference_linear_update_layer_call_and_return_conditional_losses_117152
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
?
H
.__inference_dense_3_activity_regularizer_11146
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
__inference_loss_fn_4_13484C
?dense_2_kernel_regularizer_square_readvariableop_dense_2_kernel
identity??0dense_2/kernel/Regularizer/Square/ReadVariableOp?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_2_kernel_regularizer_square_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentity"dense_2/kernel/Regularizer/mul:z:01^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_2_13462C
?dense_1_kernel_regularizer_square_readvariableop_dense_1_kernel
identity??0dense_1/kernel/Regularizer/Square/ReadVariableOp?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_1_kernel_regularizer_square_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:01^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp
?
?
F__inference_dense_1_layer_call_and_return_all_conditional_losses_13124

inputs
dense_1_kernel
dense_1_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_113232
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
.__inference_dense_1_activity_regularizer_113492
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
?
'__inference_model_1_layer_call_fn_12956

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_121842
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

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_13473?
;dense_1_bias_regularizer_square_readvariableop_dense_1_bias
identity??.dense_1/bias/Regularizer/Square/ReadVariableOp?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_1_bias_regularizer_square_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
IdentityIdentity dense_1/bias/Regularizer/mul:z:0/^dense_1/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_0_13440?
;dense_kernel_regularizer_square_readvariableop_dense_kernel
identity??.dense/kernel/Regularizer/Square/ReadVariableOp?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_kernel_regularizer_square_readvariableop_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
IdentityIdentity dense/kernel/Regularizer/mul:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
?
?
'__inference_dense_4_layer_call_fn_13363

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
B__inference_dense_4_layer_call_and_return_conditional_losses_116222
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
?
?
'__inference_model_1_layer_call_fn_12939

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_120412
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

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_13495?
;dense_2_bias_regularizer_square_readvariableop_dense_2_bias
identity??.dense_2/bias/Regularizer/Square/ReadVariableOp?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_2_bias_regularizer_square_readvariableop_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
IdentityIdentity dense_2/bias/Regularizer/mul:z:0/^dense_2/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_12600

inputs/
+dense_tensordot_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias3
/dense_1_tensordot_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias3
/dense_2_tensordot_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias3
/dense_3_tensordot_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias3
/dense_4_tensordot_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape/Reshape?
dense/Tensordot/ReadVariableOpReadVariableOp+dense_tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freev
dense/Tensordot/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposereshape/Reshape:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense/BiasAdd?
 dense/ActivityRegularizer/SquareSquaredense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2"
 dense/ActivityRegularizer/Square?
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
dense/ActivityRegularizer/Const?
dense/ActivityRegularizer/SumSum$dense/ActivityRegularizer/Square:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/Sum?
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2!
dense/ActivityRegularizer/mul/x?
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/mul?
dense/ActivityRegularizer/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape?
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack?
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1?
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast?
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivx
activation/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp/dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes?
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/free
dense_1/Tensordot/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape?
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2?
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod?
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1?
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack?
dense_1/Tensordot/transpose	Transposeactivation/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot/transpose?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense_1/Tensordot/Const_2?
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_1/BiasAdd?
"dense_1/ActivityRegularizer/SquareSquaredense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2$
"dense_1/ActivityRegularizer/Square?
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_1/ActivityRegularizer/Const?
dense_1/ActivityRegularizer/SumSum&dense_1/ActivityRegularizer/Square:y:0*dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/Sum?
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_1/ActivityRegularizer/mul/x?
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/mul?
!dense_1/ActivityRegularizer/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shape?
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack?
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1?
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast?
#dense_1/ActivityRegularizer/truedivRealDiv#dense_1/ActivityRegularizer/mul:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truediv~
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation_1/Relu?
 dense_2/Tensordot/ReadVariableOpReadVariableOp/dense_2_tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposeactivation_1/Relu:activations:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_2/BiasAdd?
"dense_2/ActivityRegularizer/SquareSquaredense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
"dense_2/ActivityRegularizer/Square?
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_2/ActivityRegularizer/Const?
dense_2/ActivityRegularizer/SumSum&dense_2/ActivityRegularizer/Square:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/Sum?
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_2/ActivityRegularizer/mul/x?
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/mul?
!dense_2/ActivityRegularizer/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/mul:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
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
 compute_aux_inputs/strided_sliceStridedSlicedense_2/BiasAdd:output:0/compute_aux_inputs/strided_slice/stack:output:01compute_aux_inputs/strided_slice/stack_1:output:01compute_aux_inputs/strided_slice/stack_2:output:0*
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
linear_update/strided_sliceStridedSlicedense_2/BiasAdd:output:0*linear_update/strided_slice/stack:output:0,linear_update/strided_slice/stack_1:output:0,linear_update/strided_slice/stack_2:output:0*
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
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_2_tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
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
dense_4/bias/Regularizer/mul?
IdentityIdentitydense_4/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity linear_update/Reshape_2:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
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
:?????????
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_11572

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

?
#__inference_signature_wrapper_12278
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_110942
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

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_11786
input_1
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
reshape/PartitionedCallPartitionedCallinput_1*
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
GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_111762
reshape/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_112262
dense/StatefulPartitionedCall?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *5
f0R.
,__inference_dense_activity_regularizer_112522+
)dense/ActivityRegularizer/PartitionedCall?
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape?
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack?
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1?
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_112732
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_113232!
dense_1/StatefulPartitionedCall?
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
.__inference_dense_1_activity_regularizer_113492-
+dense_1/ActivityRegularizer/PartitionedCall?
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shape?
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack?
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1?
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truediv?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_113702
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_114202!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
.__inference_dense_2_activity_regularizer_114462-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
"compute_aux_inputs/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_114752$
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
B__inference_dense_3_layer_call_and_return_conditional_losses_115252!
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
.__inference_dense_3_activity_regularizer_115512-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_115722
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
B__inference_dense_4_layer_call_and_return_conditional_losses_116222!
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
.__inference_dense_4_activity_regularizer_116482-
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
linear_update/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
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
H__inference_linear_update_layer_call_and_return_conditional_losses_117152
linear_update/PartitionedCall?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
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
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
C
'__inference_reshape_layer_call_fn_12974

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
GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_111762
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
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_12922

inputs/
+dense_tensordot_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias3
/dense_1_tensordot_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias3
/dense_2_tensordot_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias3
/dense_3_tensordot_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias3
/dense_4_tensordot_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias
identity

identity_1??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape/Reshape?
dense/Tensordot/ReadVariableOpReadVariableOp+dense_tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freev
dense/Tensordot/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposereshape/Reshape:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense/BiasAdd?
 dense/ActivityRegularizer/SquareSquaredense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2"
 dense/ActivityRegularizer/Square?
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
dense/ActivityRegularizer/Const?
dense/ActivityRegularizer/SumSum$dense/ActivityRegularizer/Square:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/Sum?
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2!
dense/ActivityRegularizer/mul/x?
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/mul?
dense/ActivityRegularizer/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape?
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack?
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1?
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast?
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivx
activation/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp/dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes?
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/free
dense_1/Tensordot/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape?
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2?
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod?
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1?
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack?
dense_1/Tensordot/transpose	Transposeactivation/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot/transpose?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense_1/Tensordot/Const_2?
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_1/BiasAdd?
"dense_1/ActivityRegularizer/SquareSquaredense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2$
"dense_1/ActivityRegularizer/Square?
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_1/ActivityRegularizer/Const?
dense_1/ActivityRegularizer/SumSum&dense_1/ActivityRegularizer/Square:y:0*dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/Sum?
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_1/ActivityRegularizer/mul/x?
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/mul?
!dense_1/ActivityRegularizer/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shape?
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack?
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1?
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast?
#dense_1/ActivityRegularizer/truedivRealDiv#dense_1/ActivityRegularizer/mul:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truediv~
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
activation_1/Relu?
 dense_2/Tensordot/ReadVariableOpReadVariableOp/dense_2_tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposeactivation_1/Relu:activations:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_2/BiasAdd?
"dense_2/ActivityRegularizer/SquareSquaredense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$
"dense_2/ActivityRegularizer/Square?
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!dense_2/ActivityRegularizer/Const?
dense_2/ActivityRegularizer/SumSum&dense_2/ActivityRegularizer/Square:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/Sum?
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2#
!dense_2/ActivityRegularizer/mul/x?
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/mul?
!dense_2/ActivityRegularizer/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/mul:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
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
 compute_aux_inputs/strided_sliceStridedSlicedense_2/BiasAdd:output:0/compute_aux_inputs/strided_slice/stack:output:01compute_aux_inputs/strided_slice/stack_1:output:01compute_aux_inputs/strided_slice/stack_2:output:0*
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
linear_update/strided_sliceStridedSlicedense_2/BiasAdd:output:0*linear_update/strided_slice/stack:output:0,linear_update/strided_slice/stack_1:output:0,linear_update/strided_slice/stack_2:output:0*
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
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_tensordot_readvariableop_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_1_tensordot_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/dense_2_tensordot_readvariableop_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
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
dense_4/bias/Regularizer/mul?
IdentityIdentitydense_4/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity linear_update/Reshape_2:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
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
:?????????
 
_user_specified_nameinputs
?
d
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_13217
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
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_11912
input_1
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
reshape/PartitionedCallPartitionedCallinput_1*
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
GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_111762
reshape/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_112262
dense/StatefulPartitionedCall?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *5
f0R.
,__inference_dense_activity_regularizer_112522+
)dense/ActivityRegularizer/PartitionedCall?
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape?
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack?
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1?
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_112732
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_113232!
dense_1/StatefulPartitionedCall?
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
.__inference_dense_1_activity_regularizer_113492-
+dense_1/ActivityRegularizer/PartitionedCall?
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shape?
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack?
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1?
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truediv?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_113702
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_114202!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
.__inference_dense_2_activity_regularizer_114462-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
"compute_aux_inputs/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_114752$
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
B__inference_dense_3_layer_call_and_return_conditional_losses_115252!
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
.__inference_dense_3_activity_regularizer_115512-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_115722
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
B__inference_dense_4_layer_call_and_return_conditional_losses_116222!
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
.__inference_dense_4_activity_regularizer_116482-
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
linear_update/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
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
H__inference_linear_update_layer_call_and_return_conditional_losses_117152
linear_update/PartitionedCall?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
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
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
H
.__inference_dense_1_activity_regularizer_11120
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
B__inference_dense_3_layer_call_and_return_conditional_losses_13276

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
^
B__inference_reshape_layer_call_and_return_conditional_losses_12969

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
?
?
F__inference_dense_3_layer_call_and_return_all_conditional_losses_13292

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
B__inference_dense_3_layer_call_and_return_conditional_losses_115252
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
.__inference_dense_3_activity_regularizer_115512
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
?

?
D__inference_dense_layer_call_and_return_all_conditional_losses_13044

inputs
dense_kernel

dense_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_112262
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
GPU2*0J 8? *5
f0R.
,__inference_dense_activity_regularizer_112522
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
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
.__inference_dense_2_activity_regularizer_11446
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
G__inference_activation_1_layer_call_and_return_conditional_losses_11370

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
?3
?
B__inference_dense_4_layer_call_and_return_conditional_losses_11622

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
?
I
2__inference_compute_aux_inputs_layer_call_fn_13222
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
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_114752
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
?
__inference_loss_fn_1_13451;
7dense_bias_regularizer_square_readvariableop_dense_bias
identity??,dense/bias/Regularizer/Square/ReadVariableOp?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_bias_regularizer_square_readvariableop_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
IdentityIdentitydense/bias/Regularizer/mul:z:0-^dense/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_13297

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
d
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_11475
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
F__inference_dense_4_layer_call_and_return_all_conditional_losses_13372

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
B__inference_dense_4_layer_call_and_return_conditional_losses_116222
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
.__inference_dense_4_activity_regularizer_116482
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
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_12184

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
dense_4_dense_4_kernel
dense_4_dense_4_bias
identity

identity_1??dense/StatefulPartitionedCall?,dense/bias/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?.dense_2/bias/Regularizer/Square/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?.dense_3/bias/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?.dense_4/bias/Regularizer/Square/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
reshape/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_111762
reshape/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_112262
dense/StatefulPartitionedCall?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *5
f0R.
,__inference_dense_activity_regularizer_112522+
)dense/ActivityRegularizer/PartitionedCall?
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape?
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack?
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1?
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_112732
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_113232!
dense_1/StatefulPartitionedCall?
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
.__inference_dense_1_activity_regularizer_113492-
+dense_1/ActivityRegularizer/PartitionedCall?
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/Shape?
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stack?
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1?
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_slice?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Cast?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truediv?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_113702
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
B__inference_dense_2_layer_call_and_return_conditional_losses_114202!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
.__inference_dense_2_activity_regularizer_114462-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
"compute_aux_inputs/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_114752$
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
B__inference_dense_3_layer_call_and_return_conditional_losses_115252!
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
.__inference_dense_3_activity_regularizer_115512-
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
G__inference_activation_2_layer_call_and_return_conditional_losses_115722
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
B__inference_dense_4_layer_call_and_return_conditional_losses_116222!
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
.__inference_dense_4_activity_regularizer_116482-
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
linear_update/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
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
H__inference_linear_update_layer_call_and_return_conditional_losses_117152
linear_update/PartitionedCall?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:P*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_bias*
_output_shapes
:P*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp?
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
dense/bias/Regularizer/Square?
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const?
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum?
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2
dense/bias/Regularizer/mul/x?
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:P*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
.dense_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_bias*
_output_shapes
:*
dtype020
.dense_2/bias/Regularizer/Square/ReadVariableOp?
dense_2/bias/Regularizer/SquareSquare6dense_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_2/bias/Regularizer/Square?
dense_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_2/bias/Regularizer/Const?
dense_2/bias/Regularizer/SumSum#dense_2/bias/Regularizer/Square:y:0'dense_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/Sum?
dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_2/bias/Regularizer/mul/x?
dense_2/bias/Regularizer/mulMul'dense_2/bias/Regularizer/mul/x:output:0%dense_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_2/bias/Regularizer/mul?
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
dense_4/bias/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity?

Identity_1Identity&linear_update/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall/^dense_2/bias/Regularizer/Square/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall/^dense_3/bias/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall/^dense_4/bias/Regularizer/Square/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2`
.dense_2/bias/Regularizer/Square/ReadVariableOp.dense_2/bias/Regularizer/Square/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2`
.dense_3/bias/Regularizer/Square/ReadVariableOp.dense_3/bias/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2`
.dense_4/bias/Regularizer/Square/ReadVariableOp.dense_4/bias/Regularizer/Square/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_12199
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
dense_4_kernel
dense_4_bias
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_121842
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

identity_1Identity_1:output:0*R
_input_shapesA
?:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
H
,__inference_activation_1_layer_call_fn_13134

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
GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_113702
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
__inference_loss_fn_6_13506C
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
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp
?3
?
B__inference_dense_1_layer_call_and_return_conditional_losses_13108

inputs+
'tensordot_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_1/bias/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_1_kernel*
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
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAdd?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'tensordot_readvariableop_dense_1_kernel*
_output_shapes

:PP*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:P*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp?
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:P2!
dense_1/bias/Regularizer/Square?
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const?
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum?
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *.?)2 
dense_1/bias/Regularizer/mul/x?
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
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
input_14
serving_default_input_1:0??????????
dense_44
StatefulPartitionedCall:0?????????E
linear_update4
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?Z
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?V
_tf_keras_network?V{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 4]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "compute_aux_inputs", "config": {"name": "compute_aux_inputs", "trainable": true, "dtype": "float32", "num_complex": 1, "num_real": 0}, "name": "compute_aux_inputs", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["compute_aux_inputs", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "linear_update", "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 2]}, "kernels": [], "num_complex": 1, "num_real": 0, "dt": 0.05}, "name": "linear_update", "inbound_nodes": [[["dense_2", 0, 0, {}], ["dense_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_4", 0, 0], ["linear_update", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 4]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "compute_aux_inputs", "config": {"name": "compute_aux_inputs", "trainable": true, "dtype": "float32", "num_complex": 1, "num_real": 0}, "name": "compute_aux_inputs", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["compute_aux_inputs", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "linear_update", "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 2]}, "kernels": [], "num_complex": 1, "num_real": 0, "dt": 0.05}, "name": "linear_update", "inbound_nodes": [[["dense_2", 0, 0, {}], ["dense_4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_4", 0, 0], ["linear_update", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 4]}}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 4]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 80]}}
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.11180339887498948, "maxval": 0.11180339887498948, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 80]}}
?
0regularization_losses
1	variables
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "compute_aux_inputs", "name": "compute_aux_inputs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "compute_aux_inputs", "trainable": true, "dtype": "float32", "num_complex": 1, "num_real": 0}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 2]}}
?


4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 170, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -1.0, "maxval": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1]}}
?
:regularization_losses
;	variables
<trainable_variables
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


>kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.07669649888473704, "maxval": 0.07669649888473704, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 170}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.9999998245167e-14}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 170]}}
?
Dkernels
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "linear_update", "name": "linear_update", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "linear_update", "trainable": true, "dtype": "float32", "output_dim": {"class_name": "TensorShape", "items": [1, 2]}, "kernels": [], "num_complex": 1, "num_real": 0, "dt": 0.05}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 2]}, {"class_name": "TensorShape", "items": [null, 1, 2]}]}
p
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9"
trackable_list_wrapper
f
0
1
 2
!3
*4
+5
46
57
>8
?9"
trackable_list_wrapper
f
0
1
 2
!3
*4
+5
46
57
>8
?9"
trackable_list_wrapper
?
regularization_losses
	variables
Imetrics
trainable_variables
Jlayer_regularization_losses
Klayer_metrics

Llayers
Mnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables
Nmetrics
trainable_variables
Olayer_regularization_losses
Player_metrics

Qlayers
Rnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:P2dense/kernel
:P2
dense/bias
0
?0
?1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables
Smetrics
trainable_variables
Tlayer_regularization_losses
Ulayer_metrics

Vlayers
Wnon_trainable_variables
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables
Xmetrics
trainable_variables
Ylayer_regularization_losses
Zlayer_metrics

[layers
\non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :PP2dense_1/kernel
:P2dense_1/bias
0
?0
?1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
"regularization_losses
#	variables
]metrics
$trainable_variables
^layer_regularization_losses
_layer_metrics

`layers
anon_trainable_variables
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
&regularization_losses
'	variables
bmetrics
(trainable_variables
clayer_regularization_losses
dlayer_metrics

elayers
fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :P2dense_2/kernel
:2dense_2/bias
0
?0
?1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
,regularization_losses
-	variables
gmetrics
.trainable_variables
hlayer_regularization_losses
ilayer_metrics

jlayers
knon_trainable_variables
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0regularization_losses
1	variables
lmetrics
2trainable_variables
mlayer_regularization_losses
nlayer_metrics

olayers
pnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_3/kernel
:?2dense_3/bias
0
?0
?1"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
6regularization_losses
7	variables
qmetrics
8trainable_variables
rlayer_regularization_losses
slayer_metrics

tlayers
unon_trainable_variables
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
:regularization_losses
;	variables
vmetrics
<trainable_variables
wlayer_regularization_losses
xlayer_metrics

ylayers
znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_4/kernel
:2dense_4/bias
0
?0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
@regularization_losses
A	variables
{metrics
Btrainable_variables
|layer_regularization_losses
}layer_metrics

~layers
non_trainable_variables
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
Eregularization_losses
F	variables
?metrics
Gtrainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
'__inference_model_1_layer_call_fn_12056
'__inference_model_1_layer_call_fn_12956
'__inference_model_1_layer_call_fn_12939
'__inference_model_1_layer_call_fn_12199?
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
 __inference__wrapped_model_11094?
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
input_1?????????
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_11786
B__inference_model_1_layer_call_and_return_conditional_losses_12922
B__inference_model_1_layer_call_and_return_conditional_losses_11912
B__inference_model_1_layer_call_and_return_conditional_losses_12600?
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
'__inference_reshape_layer_call_fn_12974?
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
B__inference_reshape_layer_call_and_return_conditional_losses_12969?
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
%__inference_dense_layer_call_fn_13035?
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
D__inference_dense_layer_call_and_return_all_conditional_losses_13044?
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
*__inference_activation_layer_call_fn_13054?
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
E__inference_activation_layer_call_and_return_conditional_losses_13049?
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
'__inference_dense_1_layer_call_fn_13115?
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
F__inference_dense_1_layer_call_and_return_all_conditional_losses_13124?
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
,__inference_activation_1_layer_call_fn_13134?
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
G__inference_activation_1_layer_call_and_return_conditional_losses_13129?
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
'__inference_dense_2_layer_call_fn_13195?
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
F__inference_dense_2_layer_call_and_return_all_conditional_losses_13204?
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
2__inference_compute_aux_inputs_layer_call_fn_13222?
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
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_13217?
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
'__inference_dense_3_layer_call_fn_13283?
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
F__inference_dense_3_layer_call_and_return_all_conditional_losses_13292?
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
,__inference_activation_2_layer_call_fn_13302?
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
G__inference_activation_2_layer_call_and_return_conditional_losses_13297?
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
'__inference_dense_4_layer_call_fn_13363?
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
F__inference_dense_4_layer_call_and_return_all_conditional_losses_13372?
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
-__inference_linear_update_layer_call_fn_13429?
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
H__inference_linear_update_layer_call_and_return_conditional_losses_13423?
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
__inference_loss_fn_0_13440?
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
__inference_loss_fn_1_13451?
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
__inference_loss_fn_2_13462?
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
__inference_loss_fn_3_13473?
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
__inference_loss_fn_4_13484?
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
__inference_loss_fn_5_13495?
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
__inference_loss_fn_6_13506?
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
__inference_loss_fn_7_13517?
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
__inference_loss_fn_8_13528?
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
__inference_loss_fn_9_13539?
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
#__inference_signature_wrapper_12278input_1"?
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
,__inference_dense_activity_regularizer_11107?
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
@__inference_dense_layer_call_and_return_conditional_losses_13028?
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
.__inference_dense_1_activity_regularizer_11120?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_13108?
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
.__inference_dense_2_activity_regularizer_11133?
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
B__inference_dense_2_layer_call_and_return_conditional_losses_13188?
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
.__inference_dense_3_activity_regularizer_11146?
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
B__inference_dense_3_layer_call_and_return_conditional_losses_13276?
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
.__inference_dense_4_activity_regularizer_11159?
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
B__inference_dense_4_layer_call_and_return_conditional_losses_13356?
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
 __inference__wrapped_model_11094?
 !*+45>?4?1
*?'
%?"
input_1?????????
? "s?p
0
dense_4%?"
dense_4?????????
<
linear_update+?(
linear_update??????????
G__inference_activation_1_layer_call_and_return_conditional_losses_13129`3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????P
? ?
,__inference_activation_1_layer_call_fn_13134S3?0
)?&
$?!
inputs?????????P
? "??????????P?
G__inference_activation_2_layer_call_and_return_conditional_losses_13297b4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
,__inference_activation_2_layer_call_fn_13302U4?1
*?'
%?"
inputs??????????
? "????????????
E__inference_activation_layer_call_and_return_conditional_losses_13049`3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????P
? ?
*__inference_activation_layer_call_fn_13054S3?0
)?&
$?!
inputs?????????P
? "??????????P?
M__inference_compute_aux_inputs_layer_call_and_return_conditional_losses_13217[.?+
$?!
?
x?????????
? ")?&
?
0?????????
? ?
2__inference_compute_aux_inputs_layer_call_fn_13222N.?+
$?!
?
x?????????
? "??????????[
.__inference_dense_1_activity_regularizer_11120)?
?
?
self
? "? ?
F__inference_dense_1_layer_call_and_return_all_conditional_losses_13124r !3?0
)?&
$?!
inputs?????????P
? "7?4
?
0?????????P
?
?	
1/0 ?
B__inference_dense_1_layer_call_and_return_conditional_losses_13108d !3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????P
? ?
'__inference_dense_1_layer_call_fn_13115W !3?0
)?&
$?!
inputs?????????P
? "??????????P[
.__inference_dense_2_activity_regularizer_11133)?
?
?
self
? "? ?
F__inference_dense_2_layer_call_and_return_all_conditional_losses_13204r*+3?0
)?&
$?!
inputs?????????P
? "7?4
?
0?????????
?
?	
1/0 ?
B__inference_dense_2_layer_call_and_return_conditional_losses_13188d*+3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????
? ?
'__inference_dense_2_layer_call_fn_13195W*+3?0
)?&
$?!
inputs?????????P
? "??????????[
.__inference_dense_3_activity_regularizer_11146)?
?
?
self
? "? ?
F__inference_dense_3_layer_call_and_return_all_conditional_losses_13292s453?0
)?&
$?!
inputs?????????
? "8?5
 ?
0??????????
?
?	
1/0 ?
B__inference_dense_3_layer_call_and_return_conditional_losses_13276e453?0
)?&
$?!
inputs?????????
? "*?'
 ?
0??????????
? ?
'__inference_dense_3_layer_call_fn_13283X453?0
)?&
$?!
inputs?????????
? "???????????[
.__inference_dense_4_activity_regularizer_11159)?
?
?
self
? "? ?
F__inference_dense_4_layer_call_and_return_all_conditional_losses_13372s>?4?1
*?'
%?"
inputs??????????
? "7?4
?
0?????????
?
?	
1/0 ?
B__inference_dense_4_layer_call_and_return_conditional_losses_13356e>?4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
'__inference_dense_4_layer_call_fn_13363X>?4?1
*?'
%?"
inputs??????????
? "??????????Y
,__inference_dense_activity_regularizer_11107)?
?
?
self
? "? ?
D__inference_dense_layer_call_and_return_all_conditional_losses_13044r3?0
)?&
$?!
inputs?????????
? "7?4
?
0?????????P
?
?	
1/0 ?
@__inference_dense_layer_call_and_return_conditional_losses_13028d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????P
? ?
%__inference_dense_layer_call_fn_13035W3?0
)?&
$?!
inputs?????????
? "??????????P?
H__inference_linear_update_layer_call_and_return_conditional_losses_13423?X?U
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
-__inference_linear_update_layer_call_fn_13429xX?U
N?K
I?F
!?
x/0?????????
!?
x/1?????????
? "??????????:
__inference_loss_fn_0_13440?

? 
? "? :
__inference_loss_fn_1_13451?

? 
? "? :
__inference_loss_fn_2_13462 ?

? 
? "? :
__inference_loss_fn_3_13473!?

? 
? "? :
__inference_loss_fn_4_13484*?

? 
? "? :
__inference_loss_fn_5_13495+?

? 
? "? :
__inference_loss_fn_6_135064?

? 
? "? :
__inference_loss_fn_7_135175?

? 
? "? :
__inference_loss_fn_8_13528>?

? 
? "? :
__inference_loss_fn_9_13539??

? 
? "? ?
B__inference_model_1_layer_call_and_return_conditional_losses_11786?
 !*+45>?<?9
2?/
%?"
input_1?????????
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
B__inference_model_1_layer_call_and_return_conditional_losses_11912?
 !*+45>?<?9
2?/
%?"
input_1?????????
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
B__inference_model_1_layer_call_and_return_conditional_losses_12600?
 !*+45>?;?8
1?.
$?!
inputs?????????
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
B__inference_model_1_layer_call_and_return_conditional_losses_12922?
 !*+45>?;?8
1?.
$?!
inputs?????????
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
'__inference_model_1_layer_call_fn_12056?
 !*+45>?<?9
2?/
%?"
input_1?????????
p

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_1_layer_call_fn_12199?
 !*+45>?<?9
2?/
%?"
input_1?????????
p 

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_1_layer_call_fn_12939?
 !*+45>?;?8
1?.
$?!
inputs?????????
p

 
? "E?B
?
0?????????
?
1??????????
'__inference_model_1_layer_call_fn_12956?
 !*+45>?;?8
1?.
$?!
inputs?????????
p 

 
? "E?B
?
0?????????
?
1??????????
B__inference_reshape_layer_call_and_return_conditional_losses_12969`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ~
'__inference_reshape_layer_call_fn_12974S3?0
)?&
$?!
inputs?????????
? "???????????
#__inference_signature_wrapper_12278?
 !*+45>???<
? 
5?2
0
input_1%?"
input_1?????????"s?p
0
dense_4%?"
dense_4?????????
<
linear_update+?(
linear_update?????????