# language-ai


## Reproducibility
Notes:
* latest version of keras does not include keras.preprocessing.text
  * pip install Keras-Preprocessing
 

**This is the ideal README file that is wanted from us in this assignment, try to cover all topics here:**

- A header with vital information. People using your code (don’t worry, barely anyone will) should know the what/who/where/why answers fairly quickly. Hence, for academic code, I generally always include the following:
    - A link to the paper.
    - Where it was/will be published (gives more credibility to the source).
    - The distinct software license(s) for the code AND the data (if provided).
    - The `.bib` file to cite the work (important for those h-index gains).
 
 - A section with some generally useful points for reproduction:
    - A tl;dr which highlights some points why someone who found your research code should care about this repository.
    - Instructions on how to reproduce the results in the paper (and how to get the data to do so), and what system it was built on (I generally provide Python version and OS, could be better, but it’s something).
    - Dependencies and their versions. This is generally best formatted in a `requirements.txt` but I like to put it here too.
    - Resources required. What kind of CPU/GPU it was ran on, and how long that took. Bonus points if you calculate CO2 emissions (see e.g. [here](https://mlco2.github.io/impact/#compute)).
- A section dedicated to experimental manipulation. What elements can be changed to change the experiment? Where do we change those? As you can see I even have specific line numbers in these (it’d probably be better if they were linked, but anyway).
- Ideally: a section on how to add to the research code. Which components are modular and can be swapped out? How does one do that?
