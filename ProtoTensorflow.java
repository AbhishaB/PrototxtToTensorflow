import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
public class ProtoTensorflow 
{
  	public static void main( String[] args) throws Exception 
   	{
  		//Checking if proper number of arguments have been passed
  		if(args.length!=2) {
  			System.err.println("Not enough arguments");
  			System.exit(1);
  		}
  		
  		//Initiating input file stream
  		String fileName = args[0];
        File file = new File(fileName);
        FileInputStream fis = null;
        
        // Creating output filename
        int pos = fileName.lastIndexOf(".");
        String newFileName = fileName.substring(0, pos) + "_"+args[1]+".py";
        File OutputFile = new File(newFileName);
        FileOutputStream fos = null;
        
        try {
            // Open the input file stream
            fis = new FileInputStream(file);
            
            //Generate parse tree using Antlr
            @SuppressWarnings("deprecation")
			ANTLRInputStream input = new ANTLRInputStream(fis);
            Prototxt1Lexer lexer = new Prototxt1Lexer(input);
            CommonTokenStream tokens = new CommonTokenStream(lexer);
            Prototxt1Parser parser = new Prototxt1Parser(tokens);
            ParseTree tree = parser.start(); // begin parsing at rule 'start'
   		
            //System.out.println(tree.toStringTree(parser)); // print LISP-style tree
            
            //Open the output file stream
            fos = new FileOutputStream(OutputFile);
            
        	//Create output file if it doesn't exist
  	  		if (!OutputFile.exists()) {
  	  			OutputFile.createNewFile();
  	  		}
  	  		
  	  	//	CodeGenerator cgen = new CodeGenerator(tree, fos);
  	  		
  	  	//	CodeGenerator1 cgen = new CodeGenerator1(tree, fos);
  	  		
  	  		CodeGenerator2 cgen = new CodeGenerator2(tree, fos);
  	  		
  	  		switch (args[1]) {
  	  		case "simple":
  	  			cgen.generate(false);
  	  			break;
  	  		
  	  		case "multiplexing":
  	  			cgen.generate(true);
  	  			break;
  	  			
  	  		default:
  	  			System.err.println("Second argument incorrect!");
  	  			System.exit(1);
  	  		}
  	  		
                        
        } catch (IOException e) {
            e.printStackTrace();
        }
        finally {
        	try {
        		if (fos != null) {
        			fos.close();
        		}
        	} catch (IOException e) {
        		e.printStackTrace();
        	}
        }
   	}
}