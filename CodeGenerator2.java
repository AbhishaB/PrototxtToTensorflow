import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.antlr.v4.runtime.tree.ParseTree;

public class CodeGenerator2 {
	ParseTree tree = null;
	FileOutputStream fos = null;
	
	public CodeGenerator2(ParseTree tree, FileOutputStream fos) {
		super();
		this.tree = tree;
		this.fos = fos;
	}
	
	public void generate(boolean multiplexing) {
		
		//Adding header content
		String header_content = "from __future__ import absolute_import\r\n" + 
    			"from __future__ import division\r\n" + 
    			"from __future__ import print_function\r\n" + 
    			"\r\n" + 
    			"import tensorflow as tf\r\n" + 
    			"\r\n" + 
    			"slim = tf.contrib.slim\r\n";
		
		//Getting the roots of the 4 main subtrees
        ParseTree nametree = tree.getChild(0).getChild(0);
        ParseTree inputtree = tree.getChild(0).getChild(1);
        ParseTree shapetree = tree.getChild(0).getChild(2);
        ParseTree layertree = tree.getChild(0).getChild(3);
            
        //Getting the function name in the output python file
        String func = nametree.getChild(2).getText();
        String func_name = func.substring(1, func.length()-1);
            
        //Getting input tensor to the model
        String inp = inputtree.getChild(2).getText();
        String inp_name = inp.substring(1, inp.length()-1);
        
        //Getting the default number of predicted classes
        String numclass = getNumClass(layertree);
            
        //Adding initial content (name of function and its arguments) 
        String initial_content = header_content +
        		"def "+func_name+"("+inp_name+",\r\n" + 
            	"\t\t\t\tnum_classes="+numclass+",\r\n" + 
            	"\t\t\t\tis_training=True,\r\n" + 
            	"\t\t\t\treuse=None,\r\n" + 
            	"\t\t\t\tscope='"+func_name+"'";
        
        if(multiplexing) { //Adding the extra template code for multiplexing option
        	initial_content+= ",\r\n\t\t\t\tconfig=None):\r\n\r\n"
        			+ "  selectdepth = lambda k,v: int(config[k]['ratio']*v) "
        			+ "if config and k in config and 'ratio' in config[k] else v\r\n\r\n"
        			+ "  selectinput = lambda k, v: config[k]['input'] "
        			+ "if config and k in config and 'input' in config[k] else v\r\n\r\n";
        }
        else {
        	initial_content+= "):\r\n\r\n";
        }
        
        initial_content+="  with tf.variable_scope(scope, \"Model\", reuse=reuse):\r\n" + 
    			"\twith slim.arg_scope(default_arg_scope(is_training)):\r\n"+
    			"\t  end_points = {}\r\n\r\n";
            	
        writeToFile(initial_content);
        
        Map<String, List<String>> endPts= createEndPoints(layertree, inp_name);
        
        //Generating the main code
        if(multiplexing) {
        	generateCodeMultiplexing(layertree, inp_name, endPts);
        }
        else {
        	generateCodeSimple(layertree, inp_name, endPts);    
        }
        
        String dim = shapetree.getChild(2).getChild(11).getText();
        
        String end_content = " return logits, end_points\r\n\r\n"+func_name+".default_image_size ="+dim+"\r\n\r\n";
   
        end_content+= "def default_arg_scope(is_training=True, \r\n" + 
        		"\t\t\t\t\t   weight_decay=0.00004,\r\n" + 
        		"\t\t\t\t\t   use_batch_norm=True,\r\n" + 
        		"\t\t\t\t\t   batch_norm_decay=0.9997,\r\n" + 
        		"\t\t\t\t\t   batch_norm_epsilon=0.001,\r\n" + 
        		"\t\t\t\t\t   batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):\r\n" + 
        		"\r\n" + 
        		"  batch_norm_params = {\r\n" + 
        		"\t  # Decay for the moving averages.\r\n" + 
        		"\t  'decay': batch_norm_decay,\r\n" + 
        		"\t  # epsilon to prevent 0s in variance.\r\n" + 
        		"\t  'epsilon': batch_norm_epsilon,\r\n" + 
        		"\t  # collection containing update_ops.\r\n" + 
        		"\t  'updates_collections': batch_norm_updates_collections,\r\n" + 
        		"\t  # use fused batch norm if possible.\r\n" + 
        		"\t  'fused': None,\r\n" + 
        		"  }\r\n" + 
        		"  if use_batch_norm:\r\n" + 
        		"\tnormalizer_fn = slim.batch_norm\r\n" + 
        		"\tnormalizer_params = batch_norm_params\r\n" + 
        		"  else:\r\n" + 
        		"\tnormalizer_fn = None\r\n" + 
        		"\tnormalizer_params = {}\r\n" + 
        		"\r\n" + 
        		"  # Set training state \r\n" + 
        		"  with slim.arg_scope([slim.batch_norm, slim.dropout],\r\n" + 
        		"                        is_training=is_training):\r\n" + 
        		"    # Set weight_decay for weights in Conv and FC layers.\r\n" + 
        		"    with slim.arg_scope([slim.conv2d, slim.fully_connected],\r\n" + 
        		"                        weights_regularizer=slim.l2_regularizer(weight_decay)):\r\n" + 
        		"      # Set batch norm \r\n" + 
        		"      with slim.arg_scope(\r\n" + 
        		"          [slim.conv2d],\r\n" + 
        		"          normalizer_fn=normalizer_fn,\r\n" + 
        		"          normalizer_params=normalizer_params):\r\n" + 
        		"          # Set default padding and stride\r\n" + 
        		"            with slim.arg_scope([slim.conv2d, slim.max_pool2d],\r\n" + 
        		"                      stride=1, padding='SAME') as sc:\r\n" + 
        		"              return sc";
        
        writeToFile(end_content);
        
        System.out.println("Content written to file successfully!");
	}
	
    public String getNumClass(ParseTree node){  //Finding the default number of predicted classes
    	int totChildren = node.getChildCount() - 2;
    	
    	while(totChildren>2){
    		ParseTree child = node.getChild(totChildren);
    		String type = child.getChild(1).getChild(2).getChild(0).getText();
    		
    		if(type.contains(Types.Convolution.toString())) {
    			if(child.getChild(4).getChildCount() == 1) { //only param2 is present
    				String param = child.getChild(4).getChild(0).getText();
    				if(param.contains("bias")) {
    					return child.getChild(4).getChild(0).getChild(3).getChild(2).getText();
    				}
    				else {
    					return child.getChild(4).getChild(0).getChild(2).getChild(2).getText();
    				}
    			}
    			else if(child.getChild(4).getChildCount() == 2) { //both param1 and param2 are present
    				String param = child.getChild(4).getChild(1).getText();
    				
    				if(param.contains("bias")) {
    					return child.getChild(4).getChild(1).getChild(3).getChild(2).getText();
    				}
    				else {
    					return child.getChild(4).getChild(1).getChild(2).getChild(2).getText();
    				}   				
    			}
    			//check if any of the children can be empty then the structure of the tree might be different    			 
    		}
    		
    		totChildren = totChildren - 4;
    	}
    	return null;
	}
    
    public Map<String, List<String>> createEndPoints(ParseTree layerRoot, String inp_name) {
    	int i=2;
    	Map<String, List<String>> layerMap = new HashMap<String, List<String>>();
    	Map<String, List<String>> layerMapTemp = new HashMap<String, List<String>>();

    	layerMap.put(inp_name, new ArrayList<String>());
     	
    	List<String> types = new ArrayList<String>();
    	types.add(Types.BatchNorm.toString());
    	types.add(Types.Scale.toString());
    	types.add(Types.ReLU.toString());
    	    	
    	while(layerRoot.getChild(i)!=null){
    		
    		ParseTree properties = layerRoot.getChild(i);
    		String type = properties.getChild(1).getChild(2).getChild(0).getText();
    		type=type.substring(1, type.length()-1);
    		String top = new String();
    		
    		//Finding the value of the "top" field
    		int k=3; 		
    		if(type.contains(Types.Concat.toString())) {
    			while(!(properties.getChild(k).getText()).contains("top")) {
    				k++;
    			}
    		}
    		top = properties.getChild(k).getChild(2).getText();
    		top = top.substring(1, top.length()-1);
    		
    		if(!types.contains(type)) {
    			String name = properties.getChild(0).getChild(2).getText();
    			name = name.substring(1, name.length()-1);   
    			
        		int j=2;
        		String bottom = new String();
				do {
					List<String> nextNames = new ArrayList<String>();
					
					bottom = properties.getChild(j).getChild(2).getText();
					j++;
					bottom = bottom.substring(1, bottom.length()-1);
					layerMap.put(name, new ArrayList<String>());
					
					if(!top.equals(name)) {
						layerMap.put(top, new ArrayList<String>());
						
						List<String> layrNameList = layerMapTemp.get(top);
						if(layrNameList==null) {
							layrNameList = new ArrayList<String>();
						}
						layrNameList.add(name);
						layerMapTemp.put(top, layrNameList);
					}
					
					nextNames = layerMap.get(bottom);
					
					nextNames.add(name);
					layerMap.put(bottom, nextNames);
					
				} while(properties.getChild(j).getText().contains("bottom"));
    		}
    		
    		i = i+4;
    	}
    	return layerMap;
    }
    
    public void generateCodeSimple(ParseTree layer, String inp_name, Map<String, List<String>> layerMap) {
    	int i=2;
    	String branchContent = new String();
    	Map<String, String> layerNameMap = new HashMap<String, String>();
    	List<String> endPtList = new ArrayList<String>();
    	int branchNum=0;
    	
    	layerNameMap.put(inp_name, inp_name);
    	int branching = -1;
    	
    	//Code to find the beginning of the logits layer
    	int layerCount = layer.getChildCount() - 2;
    	String layerBottom = new String();
    	
    	while(layerCount>2){
    		ParseTree child = layer.getChild(layerCount);
    		String type = child.getChild(1).getChild(2).getChild(0).getText();
    		layerCount = layerCount - 4;
    		
    		if(type.contains(Types.Softmax.toString())) { //if current type is Softmax (this should be the last layer)
    			layerBottom = child.getChild(2).getChild(2).getText();
    			layerBottom = layerBottom.substring(1, layerBottom.length()-1); //Find the name of the logits layer
    			continue;
    		}
    		
    		String top = child.getChild(3).getChild(2).getText();
			top = top.substring(1, top.length()-1);
    				
    		if(!layerBottom.isEmpty() && !top.equals(layerBottom)) {  //Found the starting point index(layerCount)	
    			layerCount = layerCount + 8;
   				break;
    		}
    	}
         	
    	while(layer.getChild(i)!=null){
    		String content = new String();
    		ParseTree properties = layer.getChild(i);
    		String type = properties.getChild(1).getChild(2).getChild(0).getText();
    		    		
    		String name = properties.getChild(0).getChild(2).getText();
    		name = name.substring(1, name.length()-1);
    		    		
    		String bottom = properties.getChild(2).getChild(2).getText();
			bottom = bottom.substring(1, bottom.length()-1);
			
			String top = properties.getChild(3).getChild(2).getText();
			top = top.substring(1,top.length()-1);
    		
    		if(branching < 2) { //no branching
    			if(type.contains(Types.Convolution.toString()) && !top.equals(bottom) && i<layerCount) {  //For Convolution type
    				
    				int k = i+4;
    				List<String> types = new ArrayList<String>();
    		    	types.add(Types.BatchNorm.toString());
    		    	types.add(Types.Scale.toString());
    		    	types.add(Types.ReLU.toString());
    				String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
    				nextType = nextType.substring(1, nextType.length()-1);
    				
    				if(!types.contains(nextType)) {
    					content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 0, false);
    				}
    				else {
    					List<String> typesPresent = new ArrayList<String>();
    					typesPresent.add(nextType);
    					
    					while(types.contains(nextType)) {
    						k = k+4;
        					nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
        					nextType = nextType.substring(1, nextType.length()-1);
        					typesPresent.add(nextType);
    					}
    					
    					if(typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 3, false);
    					}
    					else if(typesPresent.contains(Types.BatchNorm.toString()) && !typesPresent.contains(Types.ReLU.toString())) {
    						content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 1, false);
    					}
    					else if(!typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 2, false);
    					}
    				}

    				writeToFile(content);  				
    			}
    			else if(type.contains(Types.Pooling.toString()) && i<layerCount){  //For Pooling type  	
    				
    				content = Pooling(properties, layerNameMap, name, bottom, false, endPtList, branchNum);
    				
    				
    				int k = i+4;
					String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
					String nextName = new String();
					
					String nextTop = layer.getChild(k).getChild(3).getChild(2).getText();
					nextTop = nextTop.substring(1, nextTop.length()-1);

					if(nextType.contains(Types.Dropout.toString()) && nextTop.equals(name)) {
						nextName = layer.getChild(k).getChild(0).getChild(2).getText();
						nextName = nextName.substring(1, nextName.length()-1);
						String layerName = "net";
						layerNameMap.put(nextName, layerName);
						String dropout = layer.getChild(k).getChild(4).getChild(0).getChild(2).getChild(2).getText();
						
						String nextBottom = layer.getChild(k).getChild(2).getChild(2).getText();
						nextBottom = nextBottom.substring(1, nextBottom.length()-1);

						float dropout_float = Float.parseFloat(dropout);
						content+= "\t  "+layerName+" = slim.dropout("+layerNameMap.get(nextBottom)+", "+(1-dropout_float)+", scope='"+nextName+"')\r\n";
					
						if(!nextTop.equals(nextName)) {
							layerNameMap.put(nextTop, layerName);
						}
						i = i+4;
					}
    				content+="\t  end_points[end_point] = net\r\n\r\n";
    				
    				writeToFile(content);
    			}
    			else if(type.contains(Types.Softmax.toString())) {  //For Softmax type				
    				
       				content = generateLogitsCode(layer, layerNameMap);
 				
    				String layerName = "net";
    				layerNameMap.put(name, layerName);
    				content+= "\t  end_points['"+name+"'] = slim.softmax("+layerNameMap.get(bottom)+", scope='"+name+"')\r\n";
    				writeToFile(content);
    			}
    			else if(type.contains(Types.Dropout.toString())) {
    				String layerName = "net";
    				layerNameMap.put(name, layerName);
    				String dropout = properties.getChild(4).getChild(0).getChild(2).getChild(2).getText();
    	
    				float dropout_float = Float.parseFloat(dropout);
    				content+= "\t\t"+layerName+" = slim.dropout("+layerNameMap.get(bottom)+", "+(1-dropout_float)+", scope='"+name+"')\r\n"+
    						"\t\tend_points['"+name+"'] = "+layerName+"\r\n\r\n";
    				   				
    				if(!top.equals(name)) {
    					layerNameMap.put(top, layerName);
    				}
    			}
    			
    			endPtList = layerMap.get(name);
    			
    			if(endPtList!=null) {
    				branching = endPtList.size();
    			}
    		}
    		else { //branched layers
    			if(type.contains(Types.Softmax.toString())) {  //For Softmax type

       				content = generateLogitsCode(layer, layerNameMap);
 				
    				String layerName = "net";
    				layerNameMap.put(name, layerName);
    				content+= "\t  end_points['"+name+"'] = slim.softmax("+layerNameMap.get(bottom)+", scope='"+name+"')\r\n";
    				writeToFile(content);
    			}  			
    			else if(type.contains(Types.Convolution.toString())) {//For Convolution type
    				
    				int k = i+4;
    				List<String> types = new ArrayList<String>();
    		    	types.add(Types.BatchNorm.toString());
    		    	types.add(Types.Scale.toString());
    		    	types.add(Types.ReLU.toString());
    				String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
    				nextType = nextType.substring(1, nextType.length()-1);
    				
    				if(!types.contains(nextType)) {
    					branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 0, false);
    				}
    				else {
    					List<String> typesPresent = new ArrayList<String>();
    					typesPresent.add(nextType);
    					
    					while(types.contains(nextType)) {
    						k = k+4;
        					nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
        					nextType = nextType.substring(1, nextType.length()-1);
        					typesPresent.add(nextType);
    					}
    					
    					if(typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 3, false);
    					}
    					else if(typesPresent.contains(Types.BatchNorm.toString()) && !typesPresent.contains(Types.ReLU.toString())) {
    						branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 1, false);
    					}
    					else if(!typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 2, false);
    					}
    				}
    			
    				if(endPtList.contains(name)) {
    					branchNum++;
    				}
    				
       			}
    			else if(type.contains(Types.Pooling.toString())) {
   				
    				branchContent+=Pooling(properties, layerNameMap, name, bottom, true, endPtList, branchNum);
    				if(endPtList.contains(name)) {
    					branchNum++;
    				}	
    			}
    			else if(type.contains(Types.Concat.toString())) {   				
    				content= "\t  end_point = '"+name+"'\r\n"+
    						"\t  with tf.variable_scope(end_point):\r\n"+branchContent;
    				
					String layerName = "net";
    				layerNameMap.put(name, layerName);
    				List<String> bottomList = new ArrayList<String>();
    				int j = 2;
    				String multiBottom = new String();
    				do {
    					multiBottom = properties.getChild(j).getChild(2).getText();
    					multiBottom = multiBottom.substring(1, multiBottom.length()-1);
    					bottomList.add(multiBottom);
    					j++;
    				} while(properties.getChild(j).getText().contains("bottom"));
    				
    				String branches = new String();
    				for(String layr : bottomList) {
    					branches+= layerNameMap.get(layr)+" ,";
    				}
    				    				
					content+= "\t\t"+layerName+" = tf.concat(" + 
							"axis=3, values=["+branches.substring(0, branches.length()-2)+"])\r\n";
							
					int k = i+4;
					String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
					String nextName = new String();
					
					String nextTop = layer.getChild(k).getChild(3).getChild(2).getText();
					nextTop = nextTop.substring(1, nextTop.length()-1);
					
					if(nextType.contains(Types.Dropout.toString()) && nextTop.equals(name)) {
						nextName = layer.getChild(k).getChild(0).getChild(2).getText();
						nextName = nextName.substring(1, nextName.length()-1);
						layerName = "net";
						layerNameMap.put(nextName, layerName);
						String dropout = layer.getChild(k).getChild(4).getChild(0).getChild(2).getChild(2).getText();
						String nextBottom = layer.getChild(k).getChild(2).getChild(2).getText();
						nextBottom = nextBottom.substring(1, nextBottom.length()-1);
						
						float dropout_float = Float.parseFloat(dropout);
						content+= "\t\t"+layerName+" = slim.dropout("+layerNameMap.get(nextBottom)+", "+(1-dropout_float)+", scope='"+nextName+"')\r\n";

						if(!nextTop.equals(nextName)) {
							layerNameMap.put(nextTop, layerName);
						}
						i = i+4;
					}
					
					content+= "\t  end_points[end_point] = net\r\n\r\n";
					
    				writeToFile(content);
    				branchNum = 0;
    				if(!nextName.isEmpty()) {
    					endPtList = layerMap.get(nextName);
    				}
    				else {
    					endPtList = layerMap.get(name);
    				}
    				branching = endPtList.size();
    				branchContent = new String();
    			}
    		}
    		i=i+4;
    	}
	}
    
    public void generateCodeMultiplexing(ParseTree layer, String inp_name, Map<String, List<String>> layerMap) {
    	int i=2;
    	String branchContent = new String();
    	Map<String, String> layerNameMap = new HashMap<String, String>();
    	List<String> endPtList = new ArrayList<String>();
    	int branchNum=0;
    	
    	layerNameMap.put(inp_name, inp_name);
    	int branching = -1;
    	
    	//Code to find the beginning of the logits layer
    	int layerCount = layer.getChildCount() - 2;
    	String layerBottom = new String();
    	
    	while(layerCount>2){
    		ParseTree child = layer.getChild(layerCount);
    		String type = child.getChild(1).getChild(2).getChild(0).getText();
    		layerCount = layerCount - 4;
    		
    		if(type.contains(Types.Softmax.toString())) { //if current type is Softmax (this should be the last layer)
    			layerBottom = child.getChild(2).getChild(2).getText();
    			layerBottom = layerBottom.substring(1, layerBottom.length()-1); //Find the name of the logits layer
    			continue;
    		}
    		
    		String top = child.getChild(3).getChild(2).getText();
			top = top.substring(1, top.length()-1);
    				
    		if(!layerBottom.isEmpty() && !top.equals(layerBottom)) {  //Found the starting point index(layerCount)	
    			layerCount = layerCount + 8;
   				break;
    		}
    	}
         	
    	while(layer.getChild(i)!=null){
    		String content = new String();
    		ParseTree properties = layer.getChild(i);
    		String type = properties.getChild(1).getChild(2).getChild(0).getText();
    		
    		//Finding name of the current layer
    		String name = properties.getChild(0).getChild(2).getText();
    		name = name.substring(1, name.length()-1);
    		
    		String bottom = properties.getChild(2).getChild(2).getText();
			bottom = bottom.substring(1, bottom.length()-1);
			
			String top = properties.getChild(3).getChild(2).getText();
			top = top.substring(1,top.length()-1);
			
			if(branching < 2) { //no branching
    			if(type.contains(Types.Convolution.toString()) && !top.equals(bottom) && i<layerCount) {  //For Convolution type    				
    				
    				int k = i+4;
    				List<String> types = new ArrayList<String>();
    		    	types.add(Types.BatchNorm.toString());
    		    	types.add(Types.Scale.toString());
    		    	types.add(Types.ReLU.toString());
    				String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
    				nextType = nextType.substring(1, nextType.length()-1);
    				
    				if(!types.contains(nextType)) {
    					content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 0, true);
    				}
    				else {
    					List<String> typesPresent = new ArrayList<String>();
    					typesPresent.add(nextType);
    					
    					while(types.contains(nextType)) {
    						k = k+4;
        					nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
        					nextType = nextType.substring(1, nextType.length()-1);
        					typesPresent.add(nextType);
    					}
    					
    					if(typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 3, true);
    					}
    					else if(typesPresent.contains(Types.BatchNorm.toString()) && !typesPresent.contains(Types.ReLU.toString())) {
    						content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 1, true);
    					}
    					else if(!typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						content = Convolution(properties, layerNameMap, name, bottom, false, endPtList, branchNum, 2, true);
    					}
    				}
    				writeToFile(content);
    			}
    			else if(type.contains(Types.Pooling.toString()) && i<layerCount){  //For Pooling type
    				
    				content = Pooling(properties, layerNameMap, name, bottom, false, endPtList, branchNum);
    				
    				int k = i+4;
					String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
					String nextName = new String();
					
					String nextTop = layer.getChild(k).getChild(3).getChild(2).getText();
					nextTop = nextTop.substring(1, nextTop.length()-1);
    				
					if(nextType.contains(Types.Dropout.toString()) && nextTop.equals(name)) {
						nextName = layer.getChild(k).getChild(0).getChild(2).getText();
						nextName = nextName.substring(1, nextName.length()-1);
						String layerName = "net";
						layerNameMap.put(nextName, layerName);
						String dropout = layer.getChild(k).getChild(4).getChild(0).getChild(2).getChild(2).getText();
						
						String nextBottom = layer.getChild(k).getChild(2).getChild(2).getText();
						nextBottom = nextBottom.substring(1, nextBottom.length()-1);

						float dropout_float = Float.parseFloat(dropout);
						content+= "\t  "+layerName+" = slim.dropout("+layerNameMap.get(nextBottom)+", "+(1-dropout_float)+", scope='"+nextName+"')\r\n";
					
						if(!nextTop.equals(nextName)) {
							layerNameMap.put(nextTop, layerName);
						}
						i = i+4;
					}
    				content+="\t  end_points[end_point] = net\r\n\r\n";
		
    				writeToFile(content);
    			}
    			else if(type.contains(Types.Softmax.toString())) {  //For Softmax type    				
    				content = generateLogitsCode(layer, layerNameMap);

    				String layerName = "net";
    				layerNameMap.put(name, layerName);
    				content+= "\t  end_points['"+name+"'] = slim.softmax("+layerNameMap.get(bottom)+", scope='"+name+"')\r\n";
    				writeToFile(content);
    			}
    			else if(type.contains(Types.Dropout.toString())) {
    				String layerName = "net";
    				layerNameMap.put(name, layerName);
    				String dropout = properties.getChild(4).getChild(0).getChild(2).getChild(2).getText();
    	
    				float dropout_float = Float.parseFloat(dropout);
    				content+= "\t\t"+layerName+" = slim.dropout("+layerNameMap.get(bottom)+", "+(1-dropout_float)+", scope='"+name+"')\r\n"+
    						"\t\tend_points['"+name+"'] = "+layerName+"\r\n\r\n";
    				   				
    				if(!top.equals(name)) {
    					layerNameMap.put(top, layerName);
    				}
    			}
    			
    			endPtList = layerMap.get(name);
    			
    			if(endPtList!=null) {
    				branching = endPtList.size();
    			}
    		}
    		else { //branched layers
    			if(type.contains(Types.Softmax.toString())) {  //For Softmax type

       				content = generateLogitsCode(layer, layerNameMap);
 				
    				String layerName = "net";
    				layerNameMap.put(name, layerName);
    				content+= "\t  end_points['"+name+"'] = slim.softmax("+layerNameMap.get(bottom)+", scope='"+name+"')\r\n";
    				writeToFile(content);
    			}
    			else if(type.contains(Types.Convolution.toString())) {
    				
    				int k = i+4;
    				List<String> types = new ArrayList<String>();
    		    	types.add(Types.BatchNorm.toString());
    		    	types.add(Types.Scale.toString());
    		    	types.add(Types.ReLU.toString());
    				String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
    				nextType = nextType.substring(1, nextType.length()-1);
    				
    				if(!types.contains(nextType)) {
    					branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 0, true);
    				}
    				else {
    					List<String> typesPresent = new ArrayList<String>();
    					typesPresent.add(nextType);
    					
    					while(types.contains(nextType)) {
    						k = k+4;
        					nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
        					nextType = nextType.substring(1, nextType.length()-1);
        					typesPresent.add(nextType);
    					}
    					
    					if(typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 3, true);
    					}
    					else if(typesPresent.contains(Types.BatchNorm.toString()) && !typesPresent.contains(Types.ReLU.toString())) {
    						branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 1, true);
    					}
    					else if(!typesPresent.contains(Types.BatchNorm.toString()) && typesPresent.contains(Types.ReLU.toString())) {
    						branchContent+=Convolution(properties, layerNameMap, name, bottom, true, endPtList, branchNum, 2, true);
    					}
    				}
    			
    				if(endPtList.contains(name)) {
    					branchNum++;
    				}
    			}
    			else if(type.contains(Types.Pooling.toString())) {
    				
    				branchContent+=Pooling(properties, layerNameMap, name, bottom, true, endPtList, branchNum);
    				if(endPtList.contains(name)) {
    					branchNum++;
    				}					
    			}
    			else if(type.contains(Types.Concat.toString())) {
    				content= "\t  end_point = '"+name+"'\r\n\r\n"+
    						"\t  net = selectinput(end_point, net)\r\n\r\n"+
    						"\t  with tf.variable_scope(end_point):\r\n"+branchContent;
    				
					String layerName = "net";
    				layerNameMap.put(name, layerName);
    				List<String> bottomList = new ArrayList<String>();
    				int j = 2;
    				String multiBottom = new String();
    				do {
    					multiBottom = properties.getChild(j).getChild(2).getText();
    					multiBottom = multiBottom.substring(1, multiBottom.length()-1);
    					bottomList.add(multiBottom);
    					j++;
    				} while(properties.getChild(j).getText().contains("bottom"));
    				
    				String branches = new String();
    				for(String layr : bottomList) {
    					branches+= layerNameMap.get(layr)+" ,";
    				}
    				
					content+= "\t\t"+layerName+" = tf.concat(" + 
							"axis=3, values=["+branches.substring(0, branches.length()-2)+"])\r\n";
	    				
					int k = i+4;
					String nextType = layer.getChild(k).getChild(1).getChild(2).getChild(0).getText();
					String nextName = new String();
					
					String nextTop = layer.getChild(k).getChild(3).getChild(2).getText();
					nextTop = nextTop.substring(1, nextTop.length()-1);
					
					if(nextType.contains(Types.Dropout.toString()) && nextTop.equals(name)) {
						nextName = layer.getChild(k).getChild(0).getChild(2).getText();
						nextName = nextName.substring(1, nextName.length()-1);
						layerName = "net";
						layerNameMap.put(nextName, layerName);
						String dropout = layer.getChild(k).getChild(4).getChild(0).getChild(2).getChild(2).getText();
						String nextBottom = layer.getChild(k).getChild(2).getChild(2).getText();
						nextBottom = nextBottom.substring(1, nextBottom.length()-1);
						
						float dropout_float = Float.parseFloat(dropout);
						content+= "\t\t"+layerName+" = slim.dropout("+layerNameMap.get(nextBottom)+", "+(1-dropout_float)+", scope='"+nextName+"')\r\n";

						if(!nextTop.equals(nextName)) {
							layerNameMap.put(nextTop, layerName);
						}
						i = i+4;
					}
					
					content+= "\t  end_points[end_point] = net\r\n\r\n";
					
    				writeToFile(content);
    				branchNum = 0;
    				if(!nextName.isEmpty()) {
    					endPtList = layerMap.get(nextName);
    				}
    				else {
    					endPtList = layerMap.get(name);
    				}
    				branching = endPtList.size();
    				branchContent = new String();
    			}
    		}
    		i=i+4;
    	}
	}
       
    public String generateLogitsCode(ParseTree layer, Map<String, String> layerNameMap) {
    	int layerCount = layer.getChildCount() - 2;
    	String bottom = new String();

    	//Finding the starting point of the logits layer
    	while(layerCount>2){
    		ParseTree child = layer.getChild(layerCount);
    		String type = child.getChild(1).getChild(2).getChild(0).getText();
    		layerCount = layerCount - 4;
    		
    		if(type.contains(Types.Softmax.toString())) { //if current type is Softmax (this should be the last layer)
    			bottom = child.getChild(2).getChild(2).getText();
    			bottom = bottom.substring(1, bottom.length()-1); //Find the name of the logits layer
    			continue;
    		}
    		
    		String top = child.getChild(3).getChild(2).getText();
			top = top.substring(1, top.length()-1);
			
    		if(!bottom.isEmpty() && !top.equals(bottom)) {  //Found the starting point index(layerCount)	
    			layerCount = layerCount + 8;
   				break;
    		}
    	}
    	
    	//Adding initial content of logits layer
    	String content = "\t  end_point = 'Logits'\r\n" + 
    			"      with tf.variable_scope(end_point):\r\n";
    	
    	String convContent = new String();
    	
    	List<String> types = new ArrayList<String>();
    	types.add(Types.BatchNorm.toString());
    	types.add(Types.Scale.toString());
    	types.add(Types.ReLU.toString());
    	
    	String type = new String();
    	boolean flag = false;
    	do {
    		ParseTree properties = layer.getChild(layerCount);
    		
    		type = properties.getChild(1).getChild(2).getChild(0).getText();
    		String name = properties.getChild(0).getChild(2).getText();
    		name = name.substring(1, name.length()-1);
    		
    		bottom = properties.getChild(2).getChild(2).getText();
			bottom = bottom.substring(1, bottom.length()-1);
    		
    		if(flag && !types.contains(type)) { 
    			//We have found a convolution layer before (flag=true) but the next layer is not BN, Scale or ReLU
				content+= convContent;
				flag = false;
    		}

    		if(type.contains(Types.Dropout.toString())) {
    			String layerName = "net";
				layerNameMap.put(name, layerName);
				String dropout = properties.getChild(4).getChild(0).getChild(2).getChild(2).getText();
	
				float dropout_float = Float.parseFloat(dropout);
				content+= "\t\t"+layerName+" = slim.dropout("+layerNameMap.get(bottom)+", "+(1-dropout_float)+", scope='"+name+"')\r\n";
				
				String top = properties.getChild(3).getChild(2).getText();
				top = top.substring(1, top.length()-1);
				
				if(!top.equals(name)) {
					layerNameMap.put(top, layerName);
				}
    		}
    		else if(type.contains(Types.Convolution.toString())) {
    			String layerName = "logits";
				layerNameMap.put(name, layerName);
				
				int count = properties.getChild(4).getChildCount();
				ParseTree parameters = properties.getChild(4).getChild(count-1);
		
				String kersize = parameters.getChild(5).getChild(2).getText();
				convContent = "\t\t"+layerName+" = slim.conv2d("+layerNameMap.get(bottom)+", num_classes, ["+kersize+", "+kersize+"], activation_fn=None,\r\n" + 
						"                             normalizer_fn=None, scope='"+name+"')\r\n\r\n";
    			flag = true;
    			
    			String top = properties.getChild(3).getChild(2).getText();
				top = top.substring(1, top.length()-1);
				
				if(!top.equals(name)) {
					layerNameMap.put(top, layerName);
				}
      		}
    		else if(type.contains(Types.Pooling.toString())) {
    			ParseTree param = properties.getChild(4).getChild(0).getChild(3);
				String pool = param.getParent().getChild(2).getChild(2).getText();
				String layerName = "logits";
				layerNameMap.put(name, layerName);

				if(pool.equals("MAX")) { //For max pooling
					String kersize = param.getChild(0).getChild(2).getText();
					String stride = param.getChild(1).getChild(2).getText();
					content+= "\t\t"+layerName+" = slim.max_pool2d("+layerNameMap.get(bottom)+", ["+kersize+", "+kersize+"], "+
							"stride="+stride+", scope=end_point)\r\n";
				}
				else if(pool.equals("AVE")) { //For average pooling
					String kersize = param.getChild(0).getChild(2).getText();
					String stride = param.getChild(1).getChild(2).getText();
					content+= "\t\t"+layerName+" = slim.avg_pool2d("+layerNameMap.get(bottom)+", ["+kersize+", "+kersize+"], "+
							"stride="+stride+", scope=end_point)\r\n";
				} 			
    		}
    		else if(type.contains(Types.Reshape.toString())) {
    			String layerName = "logits";
     			layerNameMap.put(name, layerName);
     			
     			List<String> dimList = new ArrayList<String>();
     			ParseTree dim = properties.getChild(4).getChild(0).getChild(4);
     			
     			int j = 2;
     			while(dim.getChild(j-2)!=null && dim.getChild(j-2).getText().contains("dim")) {
     				dimList.add(dim.getChild(j).getText());
     				j = j+3;
     			}
     		     			
     			String squeeze = new String();
     			
     			switch (dimList.size()) {
     			case 1:
     				squeeze = "[1, 2, 3]";
     				break;
     			
     			case 2:
     				squeeze = "[1, 2]";
     				break;
     				
     			case 3:
     				squeeze = "[2]";
     				break;
     				
     			default:
     				squeeze = "[]";
     			}
     			
     			//Need to Change the content of this layer
     			content+= "\t\t"+layerName+" = tf.squeeze("+layerNameMap.get(bottom)+", "+squeeze+", name='"+name+"')\r\n";
     					
     			String top = properties.getChild(3).getChild(2).getText();
				top = top.substring(1, top.length()-1);
				
				if(!top.equals(name)) {
					layerNameMap.put(top, layerName);
				}
     			
    		}
    		layerCount = layerCount + 4;
    	} while(!type.contains(Types.Softmax.toString()));   	
 
    	content+="\t\tend_points[end_point] = logits\r\n\r\n";
    	return content;   	    	
    }

    public String Convolution(ParseTree properties, Map<String, String> layerNameMap, String name, String bottom,
    		boolean branching, List<String> endPtList, int branchNum, int next, boolean multiplexing) {
		
		String op = new String();
		String kersize = new String();
		String stride = new String();
		String content = new String();
		
		ParseTree parameters = null;
		
		int count = properties.getChild(4).getChildCount();
		parameters = properties.getChild(4).getChild(count-1);
			
		String param = parameters.getText();
		if(param.contains("bias")) {
			op = parameters.getChild(3).getChild(2).getText();
				
			if(param.contains("pad")) {      					
    			kersize = parameters.getChild(5).getChild(2).getText();
    				
    			if(parameters.getChild(6).getChild(2)!=null) {
    				stride = parameters.getChild(6).getChild(2).getText();
    			}
    			else {
    				stride = "1";
    			}
			}
			else {
				kersize = parameters.getChild(4).getChild(2).getText();
				
				if(parameters.getChild(5).getChild(2)!=null) {
    				stride = parameters.getChild(5).getChild(2).getText();
    			}
    			else {
    				stride = "1";
    			}
			}
		}
		else {
			op = parameters.getChild(2).getChild(2).getText();
			        						
			if(param.contains("pad")) {      					
				kersize = parameters.getChild(4).getChild(2).getText();
				
				if(parameters.getChild(5).getChild(2)!=null) {
    				stride = parameters.getChild(5).getChild(2).getText();
   				}
   				else {
   					stride = "1";
   				}
			}
			else {
				kersize = parameters.getChild(3).getChild(2).getText();
			
				if(parameters.getChild(4).getChild(2)!=null) {
   					stride = parameters.getChild(4).getChild(2).getText();
   				}
   				else {
   					stride = "1";
   				}
			}
		}	

		if(branching) {
			String layerName = new String();
			
			if(endPtList.contains(name)) {
				layerName = "branch_"+branchNum;
				content+= "\t\twith tf.variable_scope('Branch_"+branchNum+"'):\r\n"+
						"\t\t  "+layerName+" = slim.conv2d("+layerNameMap.get(bottom)+", ";
				
				if(multiplexing) {
					content+= "selectdepth(end_point,"+op+"), ";
				}
				else {
					content+= op+", ";
				}
			}
			else {
				layerName = "branch_"+(branchNum-1);
				content+= "\t\t  "+layerName+" = slim.conv2d("+layerNameMap.get(bottom)+", "+op+", ";
			}
				
			layerNameMap.put(name, layerName);
			
			
			switch (next) {
			case 0:
				content+= "activation_fn=None, normalizer_fn=None, ";
				break;
			
			case 1:
				content+= "activation_fn=None, ";
				break;
			
			case 2:
				content+= "normalizer_fn=None, ";
				break;
				
			case 3:
				break;
			}
			
			content+= "["+kersize+", "+kersize+"], scope='"+name+"')\r\n";
		}
		else {
			String layerName = "net";
			layerNameMap.put(name, layerName);
			
			content = "\t  end_point = '"+ name +"'\r\n" + 
					"\t  "+layerName+" = slim.conv2d("+layerNameMap.get(bottom)+", "+ op +", ["+kersize+", "+kersize+"], ";
			
			switch (next) {		
			case 0:	
				content+= "activation_fn=None, normalizer_fn=None, ";
				break;
				
			case 1:
				content+= "activation_fn=None, ";
				break;

			case 2:
				content+= "normalizer_fn=None, ";
				break;
				
			case 3:
				break;
			}
			
			if(stride.equals("1")) {
				content+="scope=end_point)\r\n" + 
						"\t  end_points[end_point] = net\r\n\r\n";
			}
			else {
				content+="stride="+stride+", scope=end_point)\r\n" + 
						"\t  end_points[end_point] = net\r\n\r\n";
			}
		}
		return content;
    }
    
    public String Pooling(ParseTree properties, Map<String, String> layerNameMap, String name, String bottom,
    		boolean branching, List<String> endPtList, int branchNum) {
		ParseTree param = properties.getChild(4).getChild(0).getChild(3);
		String content = new String();
		
		if(branching) {
			String kersize = param.getChild(0).getChild(2).getText();
			
			if(endPtList.contains(name)) {
				String layerName = "branch_"+branchNum;
				layerNameMap.put(name, layerName);
				    				
				content+= "\t\twith tf.variable_scope('Branch_"+branchNum+"'):\r\n"+ 
							"\t\t  "+layerName+" = slim.max_pool2d("+layerNameMap.get(bottom)+
							", ["+kersize+", "+kersize+"], scope='"+name+"')\r\n";
			}
			else {
				String layerName = "branch_"+(branchNum-1);
				layerNameMap.put(name, layerName);
			    				
				content+= "\t\t  "+layerName+" = slim.max_pool2d("+layerNameMap.get(bottom)+
							", ["+kersize+", "+kersize+"], scope='"+name+"')\r\n";
			}
		}
		else {		
			String pool = param.getParent().getChild(2).getChild(2).getText();
			String layerName = "net";
			layerNameMap.put(name, layerName);

			if(pool.equals("MAX")) { //For max pooling
				String kersize = param.getChild(0).getChild(2).getText();
				String stride = param.getChild(1).getChild(2).getText();
				content = "\r\n\t  end_point = '"+ name +"'\r\n" + 
					"\t  "+layerName+" = slim.max_pool2d("+layerNameMap.get(bottom)+", ["+kersize+", "+kersize+"], "+
					"stride="+stride+", scope=end_point)\r\n";
			}
			else if(pool.equals("AVE")) { //For average pooling
				String kersize = param.getChild(0).getChild(2).getText();
				String stride = param.getChild(1).getChild(2).getText();
				content = "\r\n\t  end_point = '"+ name +"'\r\n" + 
					"\t  "+layerName+" = slim.avg_pool2d("+layerNameMap.get(bottom)+", ["+kersize+", "+kersize+"], "+
					"stride="+stride+", scope=end_point)\r\n";
			}
		}
		return content;
    }
    
	public void writeToFile(String content) {
		try {
			byte[] bytesArray = content.getBytes();
           	  	
			fos.write(bytesArray);
			fos.flush();
		} catch (IOException e) {
			e.printStackTrace();
        }
	}	
}
