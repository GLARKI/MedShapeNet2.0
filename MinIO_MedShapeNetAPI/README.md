# MedShapeNet API
The [**MedShapeNet Package (MSN-API)**](https://github.com/GLARKI/MedShapeNet2.0/tree/main/MedShapeNetAPI) is an API that enables direct connectivity to the MedShapeNet database, comprising over 100,000 3D shapes for the medical domain.<br> 

The included datasets comprise collections of anatomical shapes (e.g., bones, organs, vessels), 3D models of surgical instruments, and even molecular structures.<br>

MedShapeNet has been established with the objective of facilitating the translation of data-driven vision algorithms for medical machine learning applications, extended reality, 3D printing, benchmarking and other related fields.<br>

Further information on MedShapeNet can be found in the [first MedShapeNet Paper](https://arxiv.org/abs/2308.16139), which describes the initial contributions. A subsequent paper will describe the new contributions and usage of the [MedShapeNet API](https://github.com/GLARKI/MedShapeNet2.0/tree/main/MedShapeNetAPI).<br>

The API enables users to search the Database, retrieve author information, download data, visualise shapes, and transform shapes into file formats that are more suitable for machine learning (e.g. as numpy arrays in .npz format).<br>

[Samples](https://github.com/GLARKI/MedShapeNet2.0/tree/main/Samples) on MSN-API usage and using it for machine learning applications will made available on the [MedShapeNet 2.0 GitHub Page](https://github.com/GLARKI/MedShapeNet2.0) in the near future.<br>

The initial version will be demonstrated during the [MICCAI 2024 tutorial](https://medshapenet-miccai-tutorial.ikim.nrw/). Following the event, further functionality will be added, for example adding labels to the shapes and including more datasets. Additionally, a [Streamlit websit](https://medshapenet-ikim.streamlit.app/) has been created as a result of the first paper. Further information can be found on the [Project Page of the Institute for Artificial Intelligence in Medicine (IKIM)](https://medshapenet.ikim.nrw/).<br>

***Functionality under constructions, functionality and all datasets will be added soon.***<br>
***Want to contribute, checkout the [MedShapeNet 2.0 GitHub Page](https://github.com/GLARKI/MedShapeNet2.0).***<br>

#### Content on this readme:
[Installation](#installation)
[Help function](#help-function)
[MSN Usage](#usage)
[Cite us](#referencecite)
[Licence information](#licence)
****

## Installation
You can install the package using pip:
```bash
pip install MedShapeNet
```

## Help function
In the command line interface after installation with pip:

```bash
msn help
```
Or in Python:
```Python
# Import MedShapeNet class for MedShapeNet package
from MedShapeNet import MedShapeNet as msn

# Call the help function
msn.msn_help()
# or
msn_instance = msn()
msn_instance.msn_help()

# Checkout the docstring - print(msn.{method}.__doc__) or print(msn.__doc__), e.g.:
print(msn.msn_help.__doc__)
print(msn.__doc__)
```
****

## Usage
The MedShapeNet object will be imported into the Python environment as MedShapeNet, and the methods can be invoked directly either via *MedShapeNet.method(args)* syntax or by creating an instance (,see section about help function,) first.<br><br>

You can use the [GettingStarted.ipynb](https://github.com/GLARKI/MedShapeNet2.0/blob/main/Samples/GettingStarted/) to get to know MedShapeNet's functionality.
****

## reference/Cite
If you use MedShapeNet in your (research) project(s), we kindly request you to cite MedShapeNet as:
```bash
@article{li_medshapenet_2023,
	title = {MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
    journal={arXiv preprint arXiv:2308.16139},
	doi = {10.48550/arXiv.2308.16139},
	author = {Li, Jianning and Zhou, Zongwei and Yang, Jiancheng and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and others},
	year = {2023},
}
```
****

## Licence
[MedShapeNet 2.0](https://github.com/GLARKI/MedShapeNet2.0) © 2024 by [Gijs Luijten](http://www.linkedin.com/in/gijsl) is licensed under:<br>[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International - CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1).
