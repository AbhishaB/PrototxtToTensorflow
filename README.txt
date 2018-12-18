Prerequisites: The machine must have java 8, 9 or 10 and jdk installed.

I have used Antlr parser generator to generate scanner and parser. I have included 
the Antlr jar in the package and also instructions to run the Antlr parser 

To run the Wootz compiler untar the abhatt22.tar . This will have the following contents:

1. Folder - Code 
2. Report - Final_submission_abhatt22.pdf

The folder Code has all the code for the compiler as well as the test files.
The test files are inside a folder called Test_files inside the Code folder

From the Code folder run the following commands

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

For running any other file either place it directly in the Code folder and run

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


I had initially included the export CLASSPATH command inside the init.sh script. 
However, sometimes I am getting the below error when I run run_simple.sh after running init.sh

Error: A JNI error has occurred, please check your installation and try again
Exception in thread "main" java.lang.NoClassDefFoundError: org/antlr/v4/runtime/CharStream
	at java.lang.Class.getDeclaredMethods0(Native Method)
	at java.lang.Class.privateGetDeclaredMethods(Class.java:2701)
	at java.lang.Class.privateGetMethodRecursive(Class.java:3048)
	at java.lang.Class.getMethod0(Class.java:3018)
	at java.lang.Class.getMethod(Class.java:1784)
	at sun.launcher.LauncherHelper.validateMainClass(LauncherHelper.java:544)
	at sun.launcher.LauncherHelper.checkAndLoadMain(LauncherHelper.java:526)
Caused by: java.lang.ClassNotFoundException: org.antlr.v4.runtime.CharStream
	at java.net.URLClassLoader.findClass(URLClassLoader.java:382)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
	at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:349)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
	... 7 more

This gets resolved if I run the export CLASSPATH command separately instead of as part of the init.sh
and then run ./init.sh . After this ./run_simple.sh and ./run_multiplexing.sh run normally.
I am not sure why this is happening. Since I found a workaround I am not trying to fix the issue.
 
