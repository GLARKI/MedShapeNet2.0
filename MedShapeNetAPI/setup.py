# Imports to setup (and find) PYPI package
from setuptools import setup, find_packages

# Imports to print a message after installation
from setuptools.command.install import install
# import sys

# Call help function after install: Doesn't work in most cases using PIP, therefor we have a message in the __init__.py when first importing MedShapeNet
class PostInstallCommand(install):
        '''Displays post-installation message relating to MedShapeNet.'''
        def run(self) -> None:
        # Post pip installation message
            print(
                'MedShapeNet API is under construction, more functionality will come soon!\n\n'
                'For method descriptions, please use msn.msn_help() in Python or msn_help in the command line interface.\n'
                'Alternatively, check the readme on PYPI or the GitHub Page: https://github.com/GLARKI/MedShapeNet2.0\n\n'
                'If you used MedShapeNet API for your research, please cite us:\n'
                '@article{li2023medshapenet,\n'
                'title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},\n'
                'author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada, and others},\n'
                'journal={arXiv preprint arXiv:2308.16139},\n'
                'year={2023}\n'
                '}'
            )
            # Call the base class's run method to ensure normal installation
            super().run()


# read README.md file to variable desciption (for automatic uploading readme onto PYPI)
with open('README.md', 'r') as f:
    description = f.read()

# Setup information
setup(
    name='MedShapeNet',
    version='0.1.3',
    description='Python API to connect and work with the MedShapeNet Medical Shapes Database (https://medshapenet.ikim.nrw/)',
    author='Gijs Luijten',
    packages=find_packages(),
    license='CC BY-NC-SA 4.0',

    # Requirements and dependancies -> you can use this for the requirements.txt as well
    install_requires=[
        # Add dependencies here.
        # e.g. 'numpy>=1.11.1'
    ],

    # CLI mapping 'name' to 'method from main.py'
    entry_points={
        'console_scripts':[
            # map 'msn_help' cli cmmand to 'main.py' inside the 'MedShapeNet' package, specifically the function msn_help
            # can be achieved trough wrapper functions or static/class-method decorators
            'msn_help = MedShapeNet.main:MedShapeNet.msn_help',
        ],
    },

    # Add README.md file to desciption page using variable description
    long_description=description,
    long_description_content_type='text/markdown',

    # Prints the PostInstallCommand at the end of the setup.
    cmdclass={
        'install': PostInstallCommand,
    },
)