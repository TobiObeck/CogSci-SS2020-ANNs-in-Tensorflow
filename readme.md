
How to setup new conda environment for this repository:
```powershell
conda create -n iannwtf python=3.7
```
-n or -name is the name option

How to activate environment to work with it:

```powershell
conda activate iannwtf
```

command line starts with `(iannwtf)` then

`conda install` does work to install packages like TensorFlow, but we use `pipinstall` because typically `pip` is better kept up to date than conda for package updates.

Install pip inside of virtual conda environment
```powershell
conda install pip`
```

Get a list of all installed packages in our virtual environment:
```powershell
conda list
```

> Before you advance, please confirm that this list (should still be rather empty) contains pip now.
> We need to make sure of this, because when subsequently we use pip to install further packages,
> if we do not have pip in our virtual conda env,
> this will use your system’s pip to install python packages into your user,
> instead of the virtual env’s pip to install packages into your virtual env

```powershell
pip install -upgrade pip
```

Installs packages and sub-dependencies

```powershell
pip install tensorflow
```

```powershell
pip install matplotlib
```

run test python script to verify tensorflow and matplotlib have been installed properly (assuming you are in the homework01 folder):

```powershell
python .\test_tensorflow.py
```