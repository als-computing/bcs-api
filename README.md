# bcs-api

## Overview: Combining LabView BCS with Bluesky

### BCS Single Motor Scan

This notebook demonstrates how the LabView Beamline Control System API can be used to implement a "Single Motor Scan" as an `ophyd` device. Furthermore, this "Single Motor Scan" device can be initiated by the `bluesky` Run Engine as a `fly` plan with data avaliable in a bluesky run document.

Open `BCS-API_06_BlueskyFlying_M201Roll_BcszSync_nb.ipynb` or `BCS-API_06_BlueskyFlying_M201Roll_BcszSync_nb.html` for details.

### BCS Single Motor _Flying_ Scan

This notebook demonstrates how the LabView Beamline Control System API can be used to implement a "Single Motor Flying Scan" as an `ophyd` device. Furthermore, this "Single Motor Flying Scan" device can be initiated by the `bluesky` Run Engine as a `fly` plan with data avaliable in a bluesky run document.

Open `BCS-API_07_BlueskyFlying_FlyingBeamlineEnergy_BcszSync_nb.ipynb` or `BCS-API_07_BlueskyFlying_FlyingBeamlineEnergy_BcszSync_nb.html` for details.

---

---

## Getting Started
1. Install *conda*
2. Configure conda environment
3. Install *git*
4. Configure git *repository*  
    ...in a new/empty folder
5. *Pull*: Download repository to local folder  
6. *Push*: Upload committed changes to server

---

## Install *conda*

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

---

## Configure conda environment

Run the following commands on the command line; 
press 'Y'(es) when requested

```bash
conda create --name bluesky python=3.8
conda activate bluesky
conda install future
conda install -c conda-forge jupyterlab
conda install intake
conda install -c conda-forge caproto ophyd nodejs
conda update -c conda-forge pip setuptools numpy
conda install git scikit-image
```

Then install the following bluesky requirements, 
per [`bluesky-tutorial` instructions](https://github.com/bluesky/tutorials#local-installation)

```bash
# Download tutorial notebooks, requirements, utilities
git clone https://github.com/bluesky/tutorials
cd tutorials

# Install required packages
python -m pip install -r binder/requirements.txt
python -m pip install -e ./bluesky-tutorial-utils  # MUST use -e here

# Install extension that supports '%matplotlib widget'.
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
```

If you get errors regarding "cannot import name 'Blosc' from 'numcodecs'",
you might need to (re)install `zarr`

```bash
conda install git zarr
```

---

## Install *git*
>### on Windows
>[Installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)  
>[Download *git* for windows](http://git-scm.com/download/win)  
>### on Mac OS X
>[Installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)  
>Git is probably already installed on your computer. 
>Try typing the following command in a Terminal window.  
>```bash
>git --version
>```
>### on Linux
>[Installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)  

---

## Configure a git repository on your computer

### In a folder with existing files
The following commands will:  

+ initialize a local git repository,  
+ create a branch of that repository that is dedicated to your local folder contents,  
+ add all files/subfolders to the repository,  
+ upload a copy of your files to a remote repository  


```bash
cd <LOCAL-PROJECT-FOLDER>
git init
git config user.name "<YOUR-NAME>"
git config user.email <YOUR-EMAIL>
git remote add origin https://<GITHUB-USER-NAME>@github.com/als-computing/bcs-api.git
git checkout <EXISTING-BRANCH-NAME>
```

_Example:_

```bash  
cd "/C/experiments/Beamline Commissioning/BCS-API/" 
git init
git config user.name "Ernest O. Lawrence"
git config user.email Ernest.O.Lawrence@lbl.gov
git remote add origin https://eolawrence@github.com/als-computing/bcs-api.git
git checkout feature0002/configure-devices
```

---

## Download remote repository to your local folder

```bash
cd <LOCAL-PROJECT-FOLDER>
git pull
```

_Example:_

```bash
cd "/C/experiments/Beamline Commissioning/BCS-API/" 
git pull
```

---

## *Push*: Upload committed changes to server

### Make changes on a NEW branch; push to remote repository

```bash
git checkout -b <BRANCH-NAME>
# Edit and save files
git add .
git commit -m "<COMMIT-MESSAGE>"
git push -u origin <BRANCH-NAME>
```

_Example:_

```bash
git checkout -b feature0003/backlash-control
# Edit and save files
git add .
git commit -m "Added implicit backlash control to motors"
git push -u feature0003/backlash-control
```
