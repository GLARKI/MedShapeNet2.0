# MedShapeNET/__init__.py

#Message to display on importing of MedShapeNet for the first time in the python project
# Define a global variable to track if the message has been displayed
_message_displayed = False
def display_message():
    global _message_displayed
    if not _message_displayed:
        message = """
        This message only displays once when importing MedShapeNet for the first time.

        MedShapeNet API is under construction, more functionality will come soon!

        For information use MedShapeNet.msn_help().
        Alternatively, check the GitHub Page: https://github.com/GLARKI/MedShapeNet2.0

        PLEASE CITE US If you used MedShapeNet API for your (research) project:
        
        @article{li2023medshapenet,
        title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
        author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada, and others},
        journal={arXiv preprint arXiv:2308.16139},
        year={2023}
        }

        PLEASE USE the def dataset_info(self, bucket_name: str) to find the proper citation alongside MedShapeNet when utilizing a dataset for your resarch project.
        """
        print(message)
        _message_displayed = True

# Display the message on import
display_message()

#Import classes and methods from .main, import cli
from .main import MedShapeNet
from .main import print_dynamic_line, download_file, download_file_from_url
from . import cli
