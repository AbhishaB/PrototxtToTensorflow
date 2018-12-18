Prerequisites: The machine must have java 8, 9 or 10 and jdk installed.

This project uses Antlr parser generator to generate scanner and parser. The Antlr jar is included in the repository
and along with instructions to run the Antlr parser.

Some test files are present in the folder called Test_files.

Instructions to run the code (assuming the repository has been cloned and downloaded):
From the root folder which has the java files run the following commands

export CLASSPATH=".:./antlr-4.7.1-complete.jar:$CLASSPATH"

./init.sh

The first command sets the classpath for the Antlr jar and the script in the second command 
will perform the initial setup including compiling the grammar and running the Antlr jar to 
create the necessary java files and also compile all java files.

Now run the below commands to generate simple Tensorflow code for the three test files

./run_simple.sh ./Test_files/inception_v2.prototxt
./run_simple.sh ./Test_files/xception-dw.prototxt
./run_simple.sh ./Test_files/SqueezeNet.prototxt

Run the below commands to generate multiplexing Tensorflow code for the three test files

./run_multiplexing.sh ./Test_files/inception_v2.prototxt
./run_multiplexing.sh ./Test_files/xception-dw.prototxt
./run_multiplexing.sh ./Test_files/SqueezeNet.prototxt

The output python files will get created in the Test_files folder.

For running any other file either place it directly in the root folder and run

./run_simple.sh <filename>
./run_multiplexing.sh <filename>

Or provide the full path+filename in place of the <filename> in the above commands

Output files will always be created in the same folder as the corresponding input files.

Once all testing is done run the below command to clear up the generated class files and 
the Antlr java files.

./clear.sh

Please note that once clear.sh has been run all the class files and extra java files(created by Antlr) 
will be removed. So the export CLASSPATH command and init.sh must be run again before running 
run_simple.sh or run_multiplexing.sh.
 
