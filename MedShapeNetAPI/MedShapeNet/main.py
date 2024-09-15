# main.py

# Imports
# For dynamic line/access the OS and to copy files
import os
import shutil
# Imports minio
from minio import Minio
from minio.error import S3Error
# To create HTTPConnectionPool/PoolManager and check the socket to handle with timeouts
import urllib3
import socket
# Handle paths system agnostic
from pathlib import Path
# Imports (thread save) multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed
# Progress bar
from tqdm import tqdm
# To convert stl to numpy and vice versa
import numpy as np
from stl import mesh
# To work with files in temporary memory
import tempfile
# handle http requests and parse filenames from url
import requests
from urllib.parse import urlparse, parse_qs

# # to time method duration
# from time import time

# # to plot
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# # to get random selection
# import random

# Helper function(s)
# print a line in the terminal for more seperation between commands
def print_dynamic_line():
    # Get the terminal width
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        # Default width if the terminal size cannot be determined
        terminal_width = 80

    # Print a line of underscores that spans the terminal width
    print('_' * terminal_width)

# Download a single file: helps for download dataset and is faster then MedShapeNet.download_file().
def download_file(minio_client: Minio, bucket_name: str, object_name: str, file_path: Path) -> None:
    """
    Downloads a file from a specified bucket in MinIO.

    :param minio_client: Minio client object.
    :param bucket_name: Name of the bucket where the file is located.
    :param object_name: Name of the object in MinIO.
    :param file_path: Path to save the downloaded file.
    """
    try:
        minio_client.fget_object(bucket_name, object_name, str(file_path))
    except S3Error as e:
        print(f"Error occurred: {e}")

def download_file_from_url(url: str, save_folder: str, filename: str = None, extension: str = '.stl', print_output=False) -> None:
    """Helper function to download a single file from an url."""
    
    # Extract filenam from url
    if filename is None:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        if 'files' in query_params:
            filename = query_params['files'][0]
        else:
            filename = 'downloade_file' + extension



    # create file_path to save
    save_folder = Path(save_folder)
    filename_path = Path(filename)
    file_path = save_folder / filename_path

    #Create the save folder if it doesn't already exists
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Handle file name conflicts by appending a number if the file already exists
    base_name, file_extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(file_path):
        # Create a new filename by appending the counter
        file_path = save_folder / f'{base_name}_{counter}{file_extension}'
        counter += 1

    # download file from url
    try:
        response = requests.get(url.strip(), stream=True)
        response.raise_for_status()

        # Save the file in the given folder with the correct filename
        with open(file_path, 'wb') as file:
            file.write(response.content)

        if print_output:
            print(f"Download from url: {url}")
            print(f"Downloaded {filename}, to directory {save_folder}")

    except Exception as e:
        print(f"Error downloading {filename}")
        print(f"Error downloading from url {url}")
        print(f'Error message: {e}')

# Main functionality to interact with the data-base/sets, their shapes/information, and labels.
class MedShapeNet:
    '''
    This class holds the main methods to:
     - Access the database
     - Download datasets/medical shapes
     - Visualize shapes
     - Convert shapes file format for Machine Learning applications
     - Gain dataset's author and paper information
     - *Under construction, more to come*

    If this API was found useful within your research please cite MedShapeNet:
     @article{li2023medshapenet,
     title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
     author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
     journal={arXiv preprint arXiv:2308.16139},
     year={2023}
     }

    METHODS:
    --------
    msn_help() -> None
        Prints a help message about the package's current status.
    --------
    *Under construction*
    '''
    # Initialize the class (minio)
    def __init__(self,
                minio_endpoint: str = "127.0.0.1:9000", # Local host
                # minio_endpoint: str = "10.49.131.44:9000", # Wireless LAN adaptor wifi 04/09/2024 -> open access will come soon.
                access_key: str = "msn_user_readwrite", 
                secret_key: str = "ikim1234",
                secure: bool = False,
                timeout: int = 5
            ) -> None:
        """
        Initializes the MedShapeNet instance with a MinIO client and sets up a download directory.

        :param minio_endpoint: MinIO server endpoint (e.g., 'localhost:9000').
        :param access_key: Access key for MinIO.
        :param secret_key: Secret key for MinIO.
        :param secure: Whether to use HTTPS (default is False).
        """
        try:
            # Custom HTTP client with timeout settings
            http_client = urllib3.PoolManager(
                timeout=urllib3.util.Timeout(connect=timeout, read=timeout),
                retries=False,
            )

            # Create the MinIO client with the custom http client
            self.minio_client = Minio(
                minio_endpoint, 
                access_key=access_key, 
                secret_key=secret_key, 
                secure=secure, 
                http_client=http_client
            )

            # Test the connection by listing buckets (lightweight operation)
            self.minio_client.list_buckets()
            print("Connection to MinIO server successful.\n")

        except urllib3.exceptions.TimeoutError:
            print(f"Connection to MinIO server timed out after {timeout} seconds.")
        except urllib3.exceptions.HTTPError as e:
            print(f"An HTTP error occurred: {e}")
        except S3Error as e:
            print(f"Failed to connect to MinIO server: {e}")
        except socket.timeout:
            print(f"Socket timed out after {timeout} seconds.")
        except Exception as e:
            print(f"An error occurred while connecting to MinIO: {e}")

         # Define the download directory
        self.download_dir = Path("msn_downloads")
        
        # Create the directory if it does not exist
        if not self.download_dir.exists():
            self.download_dir.mkdir(parents=True, exist_ok=True)
            print(f"Download directory created at: {self.download_dir.resolve()}")
        else:
            print(f"Download directory already exists at: {self.download_dir.resolve()}")


    # Help method explaining all functions and a few examples.
    @staticmethod     # Make this function act as a normal and class function, independant of the class
    def help() -> None:
        '''
        Prints a help message regarding current functionality of MedShapeNet API.
        
        Returns:
        --------
        None
        '''
        print_dynamic_line()
        print("""
                This package is currently under heavy construction, functionality will come soon!
                Current S3 access is for development only and reachable via https://xrlab.ikim.nrw wifi, full access will come soon!
              
                CURRENT FUNCTIONS:
                - msn_help (in the CLI or in Python as a method of the MedShapeNet Class.)
                
                CALL DOCSTRING of MedShapeNet class or method (in Python) using:
                print(MedShapeNet.__doc__) OR print(MedShapeNet.'{'method_name'}'.__doc__)
              
                Datasets within the s3 bucket:
                *under construction*

                If you used MedShapeNet within your research please CITE:
                @article{li2023medshapenet,
                title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
                author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
                journal={arXiv preprint arXiv:2308.16139},
                year={2023}
                }
                """)
        print_dynamic_line()


    # Create a list of datasets within the S3 storage
    def datasets(self, print_output: bool = False) -> list:
        """
        Lists all top-level datasets (buckets and top-level folders) in the MinIO server.
        It avoids listing folders that are nested within other datasets.

        :param print_output: Whether to print the dataset names.
        :return: A list of dataset names (buckets and top-level folders).
        """
        list_of_datasets = []  # List to store dataset names
        top_level_folders = set()  # To store top-level folders only

        try:
            # List all buckets
            buckets = self.minio_client.list_buckets()

            if not buckets:
                print('No buckets found, please contact the owner of the database.')
                return list_of_datasets

            # Add bucket names to the list_of_datasets
            for bucket in buckets:
                bucket_name = bucket.name
                list_of_datasets.append(bucket_name)

                # List objects in the bucket to find folders
                objects = self.minio_client.list_objects(bucket_name, recursive=False)
                database_list = []

                for obj in objects:
                    # Check if object is a folder (i.e., it ends with '/')
                    if obj.object_name.endswith('/'):
                        # Extract folder name (removing the trailing slash)
                        folder_name = obj.object_name.rstrip('/')
                        database_list.append(obj.object_name)
                        if '/' not in folder_name:  # Only consider top-level folders
                            top_level_folders.add(f"{bucket_name}/{folder_name}")

                # Add top-level folder names to list_of_datasets
                list_of_datasets.extend(top_level_folders)

            # Iterate through the list and filter out names that are part of others
            filtered_datasets = [name for name in list_of_datasets
                     if not any(other.startswith(name + '/') for other in list_of_datasets if name != other)]

            # Print results if requested
            if print_output:
                # Print each dataset with its corresponding number
                print_dynamic_line()
                for i, dataset in enumerate(filtered_datasets, 1):  # Start numbering from 1
                    print(f"{i}. {dataset}")
                print_dynamic_line()

        except S3Error as e:
            print(f"Error occurred: {e}")

        return filtered_datasets
    

    # Give (licence, citation, # of files) info on the dataset
    # Adding to MedShapeNet class
    def dataset_info(self, bucket_name: str) -> None:
        """
        Provides detailed information on the dataset:
        1. Prints the contents of the 'cite.txt' and 'licence.txt' files if they exist.
        2. Prints the total number of '.txt', '.json', and '.stl' files in the dataset.
        
        :param bucket_name: Name of the bucket to extract dataset information from.
        """
        try:

            # Determine if the bucket_name is a bucket or a bucket/folder name -> extract bucket name
            if '/' in bucket_name:
                bucket_name, folder_path  = bucket_name.split('/', 1)
            else:
                folder_path = ""


            # Use dataset_files to get a list of all files in the bucked for file counts and details
            files = self.dataset_files(bucket_name)

            # Initialize counters for different file types
            txt_count = 0
            json_count = 0
            stl_count = 0

            # Read and print the contents of cite.txt and licence.txt if available
            print_dynamic_line()
            if len(folder_path) > 0: print(f'\nDATASET: {bucket_name}/{folder_path}')
            else: print(f'\nDATASET: {bucket_name}')
            for obj in files:

                # Check if file is inside the specified folder path
                if folder_path and not obj.startswith(folder_path):
                    continue

                if obj.endswith("licence.txt"):
                    licence_content = self.minio_client.get_object(bucket_name, obj)
                    print("LICENCE INFO:")
                    print(licence_content.read().decode('utf-8'))
                elif obj.endswith("cite.txt"):
                    cite_content = self.minio_client.get_object(bucket_name, obj)
                    print("\n\nCITATION INFO:")
                    print(cite_content.read().decode('utf-8'))
                                
                # Count file types
                if obj.endswith('.txt'):
                    txt_count += 1
                elif obj.endswith('.json'):
                    json_count += 1
                elif obj.endswith('.stl'):
                    stl_count += 1

            # Print total count of specific file types with aligned formatting
            print(f"\n{'Total .txt files (how to cite, license file):':<60} {txt_count:>5}")
            print(f"{'Total .json files (labels, if in dataset as json):':<60} {json_count:>5}")
            print(f"{'Total .stl files (shapes):':<60} {stl_count:>5}")

            # Print statement about citing and providing feedback for MedShapeNet
            print("""\n
            THANK YOU FOR USING MedShapeNet  2.0 API
            If you used MedShapeNet within your research please CITE:
                @article{li2023medshapenet,
                title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
                author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and 
                Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
                journal={arXiv preprint arXiv:2308.16139},
                year={2023}}
            THANK YOU AGAIN, ALL FEEDBACK IS WELCOME AT https://github.com/GLARKI/MedShapeNet2.0 or email me https://ait.ikim.nrw/authors/gijs-luijten/ !!!
            """)
            print_dynamic_line()

        except S3Error as e:
            print(f"Error occurred: {e}")


    # List all files for a bucket/dataset (optional determine .stl,.json,.json file typ)
    def dataset_files(self, bucket_name: str, file_extension: str = None, print_output: bool = False) -> list:
        """
        Lists all files in a specified bucket, optionally filtering by file extension.
        Additionally, it counts and prints the total number of .txt, .json, and .stl files.

        :param bucket_name: Name of the bucket to list files from.
        :param file_extension: (Optional) File extension to filter by (e.g., '.stl', '.txt', '.json').
        :return: A list of file names in the bucket, filtered by the specified file extension (if any).
        """
        files_list = []  # List to store file names
        txt_count = 0
        json_count = 0
        stl_count = 0

        # Handle upper case people
        if file_extension:
            file_extension = file_extension.lower()

        try:
            # Split the input path into bucket and possible folder prefix
            if '/' in bucket_name:
                bucket_name, prefix = bucket_name.split('/', 1)
            else:
                bucket_name = bucket_name
                prefix = None

            # List all objects in the bucket
            objects = self.minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)

            if print_output:
                print_dynamic_line()
                if prefix != None:
                    print(f'Files and overview of dataset: {bucket_name + '/' + prefix}\n ')
                else:
                    print(f'Files and overview of dataset: {bucket_name}\n ')


            # Filter and list objects based on file extension
            for obj in objects:
                # Store the file name
                file_name = obj.object_name

                # Increment counters for specific file types
                if file_name.endswith('.txt'):
                    txt_count += 1
                elif file_name.endswith('.json'):
                    json_count += 1
                elif file_name.endswith('.stl'):
                    stl_count += 1

                # If a file extension is provided, filter by that
                if file_extension:
                    if file_name.endswith(file_extension):
                        files_list.append(file_name)
                        if print_output:
                            print(f"File: {file_name}")
                else:
                    # If no extension filter is provided, list all files
                    files_list.append(file_name)
                    if print_output:
                        print(f"File: {file_name}")

            # Print total count of specific file types
            if print_output:
                print('\n')
                # Corrected print statements with consistent alignment
                print(f"{'Total .txt files (how to cite, license file):':<50} {txt_count:>5}")
                print(f"{'Total .json files (labels, if in dataset as json):':<50} {json_count:>5}")
                print(f"{'Total .stl files (shapes):':<50} {stl_count:>5}")
                print_dynamic_line()

            return files_list

        except S3Error as e:
            print(f"Error occurred: {e}")
            return []


    # download a specific file from the S3 (MinIO) storage
    # The bucket is currently hosted locally and thus not available for others until I'm granted the storage solution from work.
    def download_file(self, bucket_name: str, object_name: str, file_path: Path = None, print_output: bool = True) -> None:
        """
        Downloads a file from a specified bucket in MinIO.

        :param bucket_name: Name of the bucket where the file is located.
        :param object_name: Name of the object in MinIO.
        :param file_path: Path to save the downloaded file. If None, it creates a directory named after the bucket.
        """
        # To later print the dataset name which it is downloaded from.
        dataset = bucket_name
        if file_path is None:
            # Handle bucket with or without folder paths. 
            # In other words handle both case (bucket per dataset) or one bucket multiple datasets (folders) the same. Flexible for future minio implementations.
            if '/' in bucket_name:
                # Handle case where bucket_name includes folder path
                bucket_dir = self.download_dir / bucket_name.split('/')[-1]

                # Check if dataset directory exists, if not create it
                if not bucket_dir.exists():
                    bucket_dir.mkdir(parents=True, exist_ok=True)

                # create file path and bucket name. Case insensitive
                file_path = object_name.split('/')[-1]
                bucket_name = bucket_name.split('/')[0]


            else:
                # Handle case where bucket_name does not include folder path
                bucket_dir = self.download_dir / bucket_name
                bucket_dir.mkdir(parents=True, exist_ok=True)
                file_path = bucket_dir / object_name
        
        try:
            self.minio_client.fget_object(bucket_name, object_name, str(file_path))
            if print_output:
                print(f"'{object_name}', from dataset '{dataset}', successfully downloaded to '{file_path}'")
        except S3Error as e:
            print(f"Error occurred: {e}")


    # Download a dataset (multithreathed -> increase download speed with factor 2)
    def download_dataset(self, dataset_name: str, download_dir: Path = None, num_threads: int = 4, print_output: bool = True) -> None:
        """
        Downloads all files from a specified dataset to a local directory.

        :param dataset_name: Name of the dataset. It can be a bucket name or a folder within a bucket.
        :param print_output: Whether to print download progress messages.
        """
        try:
            # Determine if dataset_name includes folder path
            if '/' in dataset_name:
                bucket_name, folder_path = dataset_name.split('/', 1)
            else:
                bucket_name = dataset_name
                folder_path = None
            
            # Create a local download directory based on whether there's a folder path or not
            if download_dir is None:
                download_dir = self.download_dir / (folder_path if folder_path else bucket_name)
                download_dir.mkdir(parents=True, exist_ok=True)

            # Get a list of all files in the dataset
            files = self.dataset_files(dataset_name)  # Assuming this function returns all files including paths
            
            # num of files
            num_of_files = len(files)

            # list to store futures (multithreading)
            futures = []

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                with tqdm(total=num_of_files, desc="Downloading files") as pbar:
                    futures = []
                    for file in files:
                        # object_name = Path(file).name
                        file_path = download_dir / Path(file).name
                        future = executor.submit(download_file, self.minio_client, bucket_name, file, file_path)
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            future.result()  # Retrieve the result to raise any exceptions
                        except Exception as e:
                            print(f"Error occurred: {e}")
                        pbar.update(1)
            
            # Print dataset info to make sure the researchers sees the citations and licence info.
            self.dataset_info(dataset_name)

        
        except S3Error as e:
            print(f"Error occurred: {e}")


    # Convert .stl (STL format) to .npz (NumPy compressed)
    def stl_to_npz(self, stl_file: str, npz_file: str, print_output = True) -> None:
        """
        Converts an STL file to a NumPy .npz file.
        
        :param stl_file: Path to the .stl file containing 3D shape data.
        :param npz_file: Path to save the converted .npz file.
        """
        try:
            # Load the STL file
            stl_mesh = mesh.Mesh.from_file(stl_file)

            # Extract vertices and faces
            vertices = stl_mesh.vectors.reshape(-1, 3)
            faces = np.arange(len(vertices)).reshape(-1, 3)

            # Save vertices and faces into the .npz file
            np.savez_compressed(npz_file, vertices=vertices, faces=faces)
            if print_output:
                print(f"Successfully converted {stl_file} to {npz_file}")

        except Exception as e:
            print(f"An error occurred while converting stl to npz: {e}")


    # convert an entire already downloaded dataset to masks
    def dataset_stl_to_npz(self, dataset: str, num_threads: int = 4, output_dir: Path = None, print_output: bool = False) -> None:
        '''
        Converts all .stl files in the root of the dataset directory to NPZ format and saves them in the masks_numpy folder.
        Copies .txt and .json files to the masks_numpy folder.

        :param dataset: Name of the dataset (can be bucket or folder name).
        :param num_threads: Number of threads for parallel processing.
        :param output_dir: Directory to save the NPZ files. Defaults to 'masks_numpy' within the dataset directory.
        :param print_output: Whether to print progress messages.
        '''
        # ensure output dir exists and make it if it doesn't
        if output_dir is None:
            if '/' in dataset:
                _, dataset_dir = dataset.split('/', 1)
                output_dir = self.download_dir / dataset_dir / 'masks_numpy'
                dataset_dir = self.download_dir / dataset_dir
            else:
                output_dir = self.download_dir / dataset / 'masks_numpy'
                dataset_dir = self.download_dir / dataset
        
        # Create the mask_numpy folder
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Use glob to get all stl files and licence/reference(.txt) and labels(.json) in the root directory (no subdirectories)
        root_files = [file for file in dataset_dir.glob('*') if file.is_file()]
        stl_files = [file_path for file_path in root_files if file_path.suffix.lower() == '.stl']
        other_files = [file_path for file_path in root_files if file_path.suffix.lower() in ['.txt', '.json']]

        # handle datasets without stl
        if not stl_files:
            if print_output:
                print(f"No STL files found in {dataset_dir}.")
            return

        # function for multithreathing (scope only to this function)
        def process_stl(file_path):
            # Convert each STL to NPZ with a correlating name
            npz_file = output_dir / (file_path.stem + '.npz')  # Create the output .npz file path
            self.stl_to_npz(str(file_path), str(npz_file),  False)

        # Use a thread pool to convert STL files in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            with tqdm(total=len(stl_files), desc="Converting STL to NPZ", unit="file") as pbar:
                for _ in executor.map(process_stl, stl_files):
                    pbar.update(1)

        # Copy .txt and .json files to the masks_numpy folder
        for file_path in other_files:
            destination = output_dir / file_path.name  # Keep the same file name in the destination
            shutil.copy(file_path, destination)

        if print_output:
            print(f'Converted STLs from {dataset} to Numpy Mask and saved to folder {output_dir}.\n Also coppied the licence and labels to this folder.')
            pass


    # Download a single stl in memory and convert it to numpy and save it
    def download_stl_as_numpy(self, bucket_name: str, stl_file: str, output_dir: Path, print_output: True) -> None:
        """
        Downloads an STL file in memory, converts it to a NumPy mask, and saves the result as a .npz file.

        :param bucket_name: Name of the S3 bucket or dataset.
        :param stl_file: Path to the STL file in the dataset.
        :param output_dir: Directory where the converted NumPy file should be saved.
        """

        if '/' in bucket_name:
            # Handle case where bucket_name includes folder path
            bucket_name, dataset = bucket_name.split('/')
        else:
            dataset = bucket_name

        if output_dir is None:
            output_dir = self.download_dir / dataset / 'masks_numpy'

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download the STL file to a temporary location in memory
            with tempfile.NamedTemporaryFile(suffix='.stl') as temp_stl:
                self.download_file(bucket_name, stl_file, temp_stl.name, print_output=False)

                # Load the STL file
                stl_mesh = mesh.Mesh.from_file(temp_stl.name)

                # Extract vertices and faces
                vertices = stl_mesh.vectors.reshape(-1, 3)
                faces = np.arange(len(vertices)).reshape(-1, 3)

                # Save vertices and faces into the .npz file
                npz_file = output_dir / (Path(stl_file).stem + '.npz')
                np.savez_compressed(npz_file, vertices=vertices, faces=faces)

            if print_output:
                print(f"Downloaded and converted STL: {stl_file} -> to numpy, from dataset/bucket: {bucket_name} and stored in: {npz_file}")

        except Exception as e:
            print(f"An error occurred while processing STL file {stl_file}: {e}")
        

    def download_dataset_masks(self, dataset_name: str, download_dir: Path = None, num_threads: int = 4, print_output: bool = True) -> None:
        """
        Downloads files from a dataset, converting STL files to NumPy masks before saving them,
        and saves non-STL files directly to the specified directory.

        STL files are downloaded in memory and converted to NumPy arrays, which are then saved
        as compressed .npz files. Non-STL files are saved directly to the download directory.

        :param dataset_name: Name of the dataset, which may include a bucket name and an optional folder path.
        :param download_dir: Directory where the dataset should be saved. If None, the directory is determined
                            based on the dataset_name and stored in the class's download directory.
        :param num_threads: Number of threads to use for parallel downloading and processing.
        :param print_output: Whether to print progress messages.
        """
        # Get correct bucket & file paths depending on dataset_name
        try:
            if '/' in dataset_name:
                bucket_name, folder_path = dataset_name.split('/', 1)
            else:
                bucket_name = dataset_name
                folder_path = None

            # Create a local download directory
            if download_dir is None:
                masks_numpy_dir = self.download_dir / (folder_path if folder_path else bucket_name) / 'masks_numpy'
                masks_numpy_dir.mkdir(parents=True, exist_ok=True)
            else:
                download_dir.mkdir(parents=True, exist_ok=True)
                masks_numpy_dir = download_dir

            # Get list of all paths of all files in the dataset
            files = self.dataset_files(dataset_name)

            # get stl and non-stl file paths
            stl_files = [file for file in files if file.lower().endswith('.stl')]
            other_files = [file for file in files if not file.lower().endswith('.stl')]

            # Multithreaded download and converting
            num_of_files = len(files)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                with tqdm(total=num_of_files, desc="Downloading and processing files", unit="file") as pbar:
                    futures = []

                    # Download non-stl files directly
                    for file in other_files:
                        file_path = masks_numpy_dir / Path(file).name
                        futures.append(executor.submit(self.download_file, bucket_name, file, file_path, False))

                    # Handle STL files (download in memory and convert to NumPy using a helper function)
                    for stl_file in stl_files:
                        futures.append(executor.submit(self.download_stl_as_numpy, bucket_name, stl_file, masks_numpy_dir, False))

                     # Wait for all downloads/conversions to complete
                    for future in as_completed(futures):
                        try:
                            future.result()  # This will raise exceptions if any
                        except Exception as e:
                            print(f"Error occurred: {e}")
                        pbar.update(1)
            
            # print output
            if print_output:
                print(f"Dataset {dataset_name} STL files converted to NumPy masks, other files download, all files stored in {masks_numpy_dir}.")

            # print licence and citation info ALWAYS when downloading entire dataset.
            self.dataset_info(dataset_name) 

        except S3Error as e:
            print(f"Error occured: {e}")

    # Methods based on Sciebo
    @staticmethod
    def search_by_name(name: str = None, print_output = True) -> list:
        '''
        This is a function that still uses the download links from sciebo and will be eventually replaced.
        Make sure to look up the correct citations/licence per shape based on the paper "https://arxiv.org/abs/2308.16139"

        Search the database by using keywords such as 'liver' or 'tumour'.

        :param name: string of the organ or disease we want to search for 
        '''
        # we mustw have a name to search by
        if name is None: raise ValueError("The 'name' parameter must be provided.")

        # Download the dataset into memory
        response = requests.get(r"https://medshapenet.ikim.nrw/uploads/MedShapeNetDataset.txt")
        # Raise an exception for HTTP errors
        response.raise_for_status() 

        # Read the content into a list of lines
        lines = response.text.splitlines()

        # Convert the search term to lowercase for case-insensitive search
        search_term = name.lower()
        
        # List to store matched URLs
        matched_urls = []
        
        # Search for URLs containing the specified name (case-insensitive)
        for line in lines:
            if search_term in line.lower():
                matched_urls.append(line)
        
        if print_output:
            # Print results
            if len(matched_urls) > 0:
                print('_________ URLs:')
                for url in matched_urls:
                    print(url)
            else:
                print('No matching entries found.')
            print(f'\n\nFound {len(matched_urls)} entries for "{name}"')
            
        '''To download the .txt to current directory
        # Specify the file path where the downloaded file will be saved
        file_path = 'MedShapeNetDataset.txt'

        # Write the content of the response to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded and saved as {file_path}')
        '''

        return matched_urls
    

    # Download and search by url (based on files on Sciebo)
    def search_and_download_by_name(self, name: str = None, max_threads: int = 5, save_folder: Path = None, extension: str = '.stl', print_output = True) -> None:
        '''
        Search and download STL files based on a name search. Files will be downloaded in parallel.

        :param name: string of the organ or disease we want to search for
        :param max_threads: maximum number of threads for downloading files
        :param save_folder: the path to save the found stls -> if None, a folder with the name of the search + "_stl" will be created
        '''
        # Perform the search using the search_by_name function
        matched_urls = self.search_by_name(name, print_output=False)
        
        # If no URLs were found, stop
        if len(matched_urls) == 0 or None:
            print(f'No entries found for "{name}". Exiting.')
            return

        # Setup save folder if not defined by the user
        if save_folder is None:
            save_folder = self.download_dir/ f'{name}_stl'

        # Create the save folder if it doesn't exist
        if not os.path.exists(save_folder):
            Path(save_folder).mkdir(parents=True, exist_ok=True)

        # Download files using multithreading with a tqdm progress bar
        if print_output:
            print(f"Starting download of {len(matched_urls)} files for search '{name}'...")

        # Prepare the progress bar
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit the download tasks and track their progress with tqdm
            futures = {executor.submit(download_file_from_url, url = url, save_folder = save_folder, filename = None, extension = extension, print_output = False): url
                    for url in matched_urls}
            
            # Use tqdm to show the progress bar
            for future in tqdm(as_completed(futures), total=len(matched_urls), desc="Downloading files"):
                pass  # tqdm updates the progress bar automatically
        
        if print_output:
            print(f'Download complete! Files are stored in folder: {save_folder}')




    ''' npz file info - maybe usefull later
    def get_npz_file_info(self, npz_file_path: Path) -> None:
        """
        Load an NPZ file, print its size, and find the maximum and minimum values.
        Also prints all values in each array in matrix form and provides their shapes.

        :param npz_file_path: Path to the NPZ file.
        """
        try:
            # Load the NPZ file
            with np.load(npz_file_path) as data:
                # Get file size
                file_size = npz_file_path.stat().st_size  # Size in bytes

                # Print file size
                print(f"NPZ File: {npz_file_path}")
                print(f"File Size: {file_size / 1024:.2f} KB")  # Size in kilobytes

                # Process each array in the NPZ file
                for key in data.files:
                    array = data[key]

                    # Print shape of the array
                    print(f"\nArray '{key}' shape: {array.shape}")

                    # Print the matrix values
                    print(f"Array '{key}' values:")
                    print(array)

                    # Calculate and print max and min values for the array
                    max_value = np.max(array)
                    min_value = np.min(array)
                    print(f"Max Value in '{key}': {max_value}")
                    print(f"Min Value in '{key}': {min_value}")

        except Exception as e:
            print(f"An error occurred: {e}")
    '''
    
    ''' For interactive image -> manages to crash at my limited pc
    def visualize_random_stl_files(self, dataset_name: str, num_files: int = 4) -> None:
        """
        Selects a number of random STL files from the dataset, downloads them to a temporary directory,
        and plots them in a 2x2 grid as static images.

        :param dataset_name: Name of the dataset.
        :param num_files: Number of STL files to visualize.
        """
        try:
            # Create a temporary directory
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Get list of STL files in the dataset
            stl_files = self.dataset_files(dataset_name, 'stl')

            # Select random STL files
            selected_files = random.sample(stl_files, num_files)

            # Download STL files to the temporary directory
            for stl_file in selected_files:
                temp_file_path = os.path.join(temp_dir, os.path.basename(stl_file))
                self.download_file(dataset_name, stl_file, temp_file_path)

            # Create a figure for the subplots (2x2 grid)
            fig = plt.figure(figsize=(10, 10))

            for i, temp_file_name in enumerate(os.listdir(temp_dir)):
                if i >= num_files:
                    break

                temp_file_path = os.path.join(temp_dir, temp_file_name)

                # Load the STL file from the temporary directory
                stl_mesh = mesh.Mesh.from_file(temp_file_path)

                # Create a subplot for the current STL file
                ax = fig.add_subplot(2, 2, i + 1, projection='3d')
                ax.set_title(os.path.splitext(temp_file_name)[0])

                # Plot the STL file
                collection = Poly3DCollection(stl_mesh.vectors, alpha=0.5, facecolors='cyan')
                ax.add_collection3d(collection)

                # Fix the camera angle
                ax.view_init(elev=30, azim=30)

                # Adjust the aspect ratio
                scale = np.array([stl_mesh.points[:, 0], stl_mesh.points[:, 1], stl_mesh.points[:, 2]])
                ax.set_xlim([scale[0].min(), scale[0].max()])
                ax.set_ylim([scale[1].min(), scale[1].max()])
                ax.set_zlim([scale[2].min(), scale[2].max()])

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"An error occurred while visualizing STL files: {e}")

        finally:
            # Cleanup the temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error occurred while deleting temporary directory {temp_dir}: {e}")
    '''
    
    ''' For saving it as png -> def visualize_random_stl_files
    def visualize_random_stl_files(self, dataset_name: str, num_files: int = 4) -> None:
        """
        Selects a number of random STL files from the dataset, downloads them to a temporary directory,
        and saves them as static PNG images in a specified directory.

        :param dataset_name: Name of the dataset.
        :param num_files: Number of STL files to visualize.
        """
        try:
            # Create a temporary directory
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Get list of STL files in the dataset
            stl_files = self.dataset_files(dataset_name, 'stl')

            # Select random STL files
            selected_files = random.sample(stl_files, num_files)

            # Download STL files to the temporary directory
            for stl_file in selected_files:
                temp_file_path = os.path.join(temp_dir, os.path.basename(stl_file))
                self.download_file(dataset_name, stl_file, temp_file_path)

            # Create a figure for the subplots (2x2 grid)
            fig = plt.figure(figsize=(10, 10))

            for i, temp_file_name in enumerate(os.listdir(temp_dir)):
                if i >= num_files:
                    break

                temp_file_path = os.path.join(temp_dir, temp_file_name)

                # Load the STL file from the temporary directory
                stl_mesh = mesh.Mesh.from_file(temp_file_path)

                # Create a subplot for the current STL file
                ax = fig.add_subplot(2, 2, i + 1, projection='3d')
                ax.set_title(os.path.splitext(temp_file_name)[0])

                # Plot the STL file
                collection = Poly3DCollection(stl_mesh.vectors, alpha=0.5, facecolors='cyan')
                ax.add_collection3d(collection)

                # Fix the camera angle
                ax.view_init(elev=30, azim=30)

                # Adjust the aspect ratio
                scale = np.array([stl_mesh.points[:, 0], stl_mesh.points[:, 1], stl_mesh.points[:, 2]])
                ax.set_xlim([scale[0].min(), scale[0].max()])
                ax.set_ylim([scale[1].min(), scale[1].max()])
                ax.set_zlim([scale[2].min(), scale[2].max()])

            # Save the figure as a PNG file
            output_path = os.path.join(os.getcwd(), 'stl_visualization.png')
            plt.tight_layout()
            plt.savefig(output_path, format='png')
            plt.close(fig)  # Close the figure to free up memory

            print(f"STL visualizations saved as {output_path}")

        except Exception as e:
            print(f"An error occurred while visualizing STL files: {e}")

        finally:
            # Cleanup the temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error occurred while deleting temporary directory {temp_dir}: {e}")
    '''


# Entry point for direct execution
if __name__ == "__main__":
    # This is not intented to run solo, it's a module
    print("You are running the main.py from MedShapeNet directly, please install the PYPI 'MedShapeNet' package, import MedShapeNet and its methods in your python script.")
    
    # Instantiate the class object
    print("\n")
    msn = MedShapeNet()

    # # Print the help statement directly
    # msn.help()

    # Print and create a list of datasets
    print("\n")
    list_of_datasets = msn.datasets(True)

    # # Print the dataset info
    # for dataset in list_of_datasets:
    #     msn.dataset_info(dataset)

    # # Get a list of files per dataset, and based on file type
    # for dataset in list_of_datasets:
    #     print("\n")
    #     list_of_files = msn.dataset_files(dataset, print_output=False) # Print output is optional
    #     print(f"files in {dataset}:\n{list_of_files}\n")
    #     list_of_stl_files = msn.dataset_files(dataset, '.stl', print_output=False)
    #     print(f"STL files in {dataset}:\n{list_of_stl_files}\n")
    #     list_of_json_files = msn.dataset_files(dataset, '.json', print_output=False)
    #     print(f"JSON files in {dataset}:\n{list_of_json_files}\n")
    #     list_of_files = msn.dataset_files(dataset, '.txt', print_output=False)
    #     print(f"TXT files in {dataset}:\n{list_of_files}\n")
    
    # # Print access a specific file within the list
    # for stl_file in list_of_stl_files:
    #     print(stl_file)
    
    # # Download a specific file
    # msn.download_file(dataset, list_of_stl_files[0], file_path=None, print_output=True)

    # # Download a specific file
    # dataset = list_of_datasets[0]
    # stl_file = msn.dataset_files(dataset, '.stl', print_output=False)
    # stl_file = stl_file[0]
    # print(dataset, ' : ', stl_file)
    # msn.download_file(dataset, stl_file, file_path=None, print_output=True)

    # # Download entire dataset
    # print('\n')
    # for dataset in list_of_datasets:
    #     msn.download_dataset(dataset, download_dir = None, num_threads=4, print_output= False)
    #     print('\n')

    # # Convert downloaded dataset to npz masks
    # for dataset in list_of_datasets:
    #     print(f'\n{dataset}')
    #     msn.dataset_stl_to_npz(dataset, num_threads = 4, output_dir = None, print_output = True)

    # # download a single stl as a mask
    # for dataset in list_of_datasets:
    #     print(dataset)
    #     files = msn.dataset_files(dataset, '.stl', False)
    #     stl_file = files[0]
    #     msn.download_stl_as_numpy(dataset, stl_file, None, True)

    '''
    Convert a single previously downloaded stl to a numpy mask
    # Convert a single previously downloaded stl to a numpy mask
    # Define the paths
    stl_file = r"msn_downloads\asoca1\CoronaryArtery_0.stl"
    npz_file = r"msn_downloads\asoca1\masks_numpy\CoronaryArtery_0.npz"
    # Call the stl_to_npz method
    msn.stl_to_npz(stl_file, npz_file, print_output=True)

    # Download the dataset directly as numpy masks instead of STLs - time different num of workers
    start_time_1 = time()
    for dataset in list_of_datasets:
        msn.download_dataset_masks(dataset, download_dir=None, num_threads = 1, print_output = True)
    end_time_1 = time()
    elapsed_time_1 = end_time_1 - start_time_1

    start_time_4 = time()
    for dataset in list_of_datasets:
        msn.download_dataset_masks(dataset, download_dir=None, num_threads = 4, print_output = True)
    end_time_4 = time()
    elapsed_time_4 = end_time_4 - start_time_4

    start_time_8 = time()
    for dataset in list_of_datasets:
        msn.download_dataset_masks(dataset, download_dir=None, num_threads = 8, print_output = True)
    end_time_8 = time()
    elapsed_time_8 = end_time_8 - start_time_8

    start_time_32 = time()
    for dataset in list_of_datasets:
        msn.download_dataset_masks(dataset, download_dir=None, num_threads = 32, print_output = True)
    end_time_32 = time()
    elapsed_time_32 = end_time_32 - start_time_32

    print(f"Elapsed time num_of_workers(32): {elapsed_time_32:.6f} seconds")
    print(f"Elapsed time num_of_workers(8): {elapsed_time_8:.6f} seconds")
    print(f"Elapsed time num_of_workers(4): {elapsed_time_4:.6f} seconds")
    print(f"Elapsed time num_of_workers(1): {elapsed_time_1:.6f} seconds")

    # # Tested:
    # Elapsed time num_of_workers(32): 8.886791 seconds
    # Elapsed time num_of_workers(8): 9.596343 seconds
    # Elapsed time num_of_workers(4): 13.416085 seconds
    # Elapsed time num_of_workers(1): 41.731470 seconds
    '''

    # # Example usage get NPZ info
    # npz_file_path = Path("msn_downloads/asoca1/masks_numpy/CoronaryArtery_7.npz")
    # msn.get_npz_file_info(npz_file_path)

    # msn.visualize_random_stl_files(list_of_datasets[0], num_files = 4)

    # # search by organ
    # liver_download_urls = msn.search_by_name('liver', True)
    # tumor_download_urls = msn.search_by_name('tumor', True)
    # instrument_download_urls = msn.search_by_name('instrument', True)
    # # print(liver_download_urls, "number of liver's found: ", len(liver_download_urls))


    # test download file from url and search dataset and download
    # instrument_download_urls = msn.search_by_name('instrument', False)
    # url = instrument_download_urls[0]
    # save_folder = r'test_download_from_url'
    # for i in range(3):
    #     download_file_from_url(url, save_folder,filename=None ,extension='.stl', print_output=True)

    '''
    from time import time
    # start_time_1 = time()
    # msn.search_and_download_by_name(name='face', max_threads=1, save_folder=None, extension='.stl', print_output=True)
    # end_time_1 = time()
    # elapsed_time_1 = end_time_1 - start_time_1

    # start_time_2 = time()
    # msn.search_and_download_by_name(name='face', max_threads=2, save_folder=None, extension='.stl', print_output=True)
    # end_time_2 = time()
    # elapsed_time_2 = end_time_2 - start_time_2

    # start_time_4 = time()
    # msn.search_and_download_by_name(name='face', max_threads=4, save_folder=None, extension='.stl', print_output=True)
    # end_time_4 = time()
    # elapsed_time_4 = end_time_4 - start_time_4

    # start_time_8 = time()
    # msn.search_and_download_by_name(name='face', max_threads=8, save_folder=None, extension='.stl', print_output=True)
    # end_time_8 = time()
    # elapsed_time_8 = end_time_8 - start_time_8

    # start_time_16 = time()
    # msn.search_and_download_by_name(name='face', max_threads=16, save_folder=None, extension='.stl', print_output=True)
    # end_time_16 = time()
    # elapsed_time_16 = end_time_16 - start_time_16
    start_time_16 = time()
    msn.search_and_download_by_name(name='face', max_threads=64, save_folder=None, extension='.stl', print_output=True)
    end_time_16 = time()
    elapsed_time_16 = end_time_16 - start_time_16

    # print("FACE")
    # print(f"Elapsed time num_of_workers(1): {elapsed_time_1:.6f} seconds")
    # print(f"Elapsed time num_of_workers(2): {elapsed_time_2:.6f} seconds")
    # print(f"Elapsed time num_of_workers(4): {elapsed_time_4:.6f} seconds")
    # print(f"Elapsed time num_of_workers(8): {elapsed_time_8:.6f} seconds")
    print(f"Elapsed time num_of_workers(16): {elapsed_time_16:.6f} seconds")
    '''

    """
    Elapsed time num_of_workers(1): 85.026999 seconds
    Elapsed time num_of_workers(2): 106.440322 seconds
    Elapsed time num_of_workers(4): 92.529551 seconds
    Elapsed time num_of_workers(8): 99.029089 seconds
    Elapsed time num_of_workers(16): 79.506153 seconds
    Elapsed time num_of_workers(64): 74.478786 seconds
    _________ URLs:
    """

    from time import time
    start_time_16 = time()
    # msn.search_and_download_by_name(name='instrument', max_threads=16, save_folder=None, extension='.stl', print_output=True)
    msn.search_and_download_by_name(name='face', max_threads=16, save_folder=None, extension='.stl', print_output=True)
    end_time_16 = time()
    elapsed_time_16 = end_time_16- start_time_16
    print(f"INSTRUMENTS Elapsed time num_of_workers(128): {elapsed_time_16:.6f} seconds")
    # INSTRUMENTS Elapsed time num_of_workers(64): 498.668118 seconds
    # INSTRUMENTS Elapsed time num_of_workers(128): 486.082569 seconds

    