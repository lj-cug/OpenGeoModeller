Steps to add Julia to Jupyter Notebook

Step 1: Download and Install Julia

Step 2: Open the Julia Command-Line

Next, open the Julia command-line, also known as the REPL (read-eval-print-loop):
julia>

Step 3: Add Julia to Jupyter Notebook

In order to add Julia to Jupyter Notebook, you’ll need to type the following command and then press ENTER:

using Pkg

Then, type this command and press ENTER:

Pkg.add("IJulia")

或者

] add IJulia    # 别忘了 ]

Step 4: Download and Install Anaconda
用anaconda和miniconda都行，图方便省事还是用前者了，切记Python版本需要3.8以上。

#升级jupyter

pip install --upgrade jupyter

Step 5: Create a new Notebook

To create a new Notebook for Julia:

Click on New which is located on the top-right of your screen. Then, select Julia from the drop-down list

Step 6: Write your Code

For the final step, write your code. For example, here is a simple code to print “Hello World” using Julia:

println("Hello World")
Click on Run to execute the code, and the phrase, Hello World, would be printed as follows:

Hello World

