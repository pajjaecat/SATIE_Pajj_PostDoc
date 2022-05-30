## What it is 
### A repository that contains my post-Doc Work
All the code presented here is written in [python](https://www.python.org/), making use of the powerful [Jupyter](https://jupyter.org/). A list of the library and packages needed to run the notebooks and modules is available in [package-list](package-list.txt). 

Create new environment
-
To create a new environment containing all the prerequisites packages, follow these steps:
1. Open up an Anaconda Prompt and navigate to the repository's root directory (where [environmentFile.yml](environmentFile.yml) is located in the host machine);
2. Create the environment named `PajjPostDoc` by running: 
    ```python
      conda env create -f environmentFile.yml
    ```
    See [here](https://medium.com/swlh/setting-up-a-conda-environment-in-less-than-5-minutes-e64d8fc338e4) if you encounter ResolvePackageNotFound error.
3. Activate the new environment by running 
    ```python
      conda activate PajjPostDoc
    ```
4. Open an Anaconda navigator, and select the newly created environment

    ![Anaconda_nav (4)](https://user-images.githubusercontent.com/37332216/170979169-14c1203e-e257-4088-8e9e-5010fb09dce3.JPG)
5. Continue as usual

___
## Architecture
The repository is subdivided into four subfolders as follows:  
- [Notebooks](Notebooks/): Contains all the notebooks and examples of usage of the function defined in the modules. See [NotebooksReadMe](Notebooks/README.md) for a brief description of each notebook ;
- [Modules](Modules/): Contains different modules(functions and variables). See [ModulesReadMe](Modules/README.md) for more details ;
[DataFiles](DataFiles/): Contains the files/dataset associated with the notebooks or simulation results that can be recalled for easy use. See [DataReadME](DataFiles/README.md) for more details.
- [PdfFiles](PdfFiles/): Contains the PDF files. See [PdfReadME](PdfFiles/README.md) for more details.


