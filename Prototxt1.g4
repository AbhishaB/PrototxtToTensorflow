grammar Prototxt1;	
	
start:	progtxt 
	 |
	 ;

progtxt: name input inpshape layers;
name: 'name' COLON STRING ;
input: 'input' COLON STRING ;
inpshape: 'input_shape' LEFT_BRACE  dimension RIGHT_BRACE ;
dimension: ('dim' COLON NUMBER )+;
layers: ('layer' LEFT_BRACE  properties RIGHT_BRACE )* ;
properties: name type (bottom)* top parameters;
type: 'type' COLON types ;
types: '"Convolution"'
	 | '"BatchNorm"'
	 | '"Scale"'
	 | '"ReLU"'
	 | '"Pooling"'
	 | '"Concat"'
	 | '"Dropout"'
	 | '"Reshape"'
	 | '"Softmax"'
	 ;
bottom: 'bottom' COLON STRING ;
top: 'top' COLON STRING ;
parameters: (param1)* param2
		  | param3
		  | param4
		  | param5
		  | param6
		  | param7
		  |
		  ;

param1: 'param' LEFT_BRACE  'lr_mult' COLON NUMBER  'decay_mult' COLON NUMBER  RIGHT_BRACE ;

param2: 'convolution_param' LEFT_BRACE  (bias)* output (pad)* kersize (stride)* (weight)* RIGHT_BRACE ;
bias: 'bias_term' COLON bool ;
output: 'num_output' COLON NUMBER ;
pad: 'pad' COLON NUMBER ;
kersize: 'kernel_size' COLON NUMBER ;
stride: 'stride' COLON NUMBER ;
weight: 'weight_filler' LEFT_BRACE  type1 'std' COLON NUMBER  RIGHT_BRACE ;
type1: 'type' COLON typeval;

typeval: '"gaussian"'
	   | '"xavier"'
	   ;

param3: 'batch_norm_param' LEFT_BRACE  'use_global_stats' COLON bool eps RIGHT_BRACE ;
eps: 'eps' COLON NUMBER
   |
   ;

param4: 'scale_param' LEFT_BRACE  'bias_term' COLON bool  RIGHT_BRACE ;

param5: 'pooling_param' LEFT_BRACE  pool param5_1 RIGHT_BRACE ;
param5_1: kersize stride (pad)* 
		| gpool
		;
gpool: 'global_pooling' COLON bool ;
pool: 'pool' COLON size ;
size: 'MAX'
	| 'AVE'
	| 'STOCHASTIC'
	;
param6: 'dropout_param' LEFT_BRACE  dropoutratio RIGHT_BRACE ;
dropoutratio: 'dropout_ratio' COLON NUMBER ; 

param7: 'reshape_param' LEFT_BRACE  'shape' LEFT_BRACE  dimension RIGHT_BRACE  RIGHT_BRACE ;
bool: 'true'
	| 'false';

STRING: ["]([a-zA-Z0-9_/])+["];
NUMBER: ([0-9]+[.])?[0-9]+ ;
//FLOAT: ([0-9]+([.][0-9]*)?|[.][0-9]+);
LEFT_BRACE: '{' ;
RIGHT_BRACE: '}' ;
COLON: ':' ;
WS: [ \n\t\r]+ -> skip;