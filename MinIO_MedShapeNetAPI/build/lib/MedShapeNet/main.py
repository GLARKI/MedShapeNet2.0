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
    """
    Prints a line of underscores across the terminal width for visual separation between commands.

    This function determines the terminal width and prints a line of underscores to enhance readability
    between terminal outputs. If the terminal width cannot be determined, a default width of 80 characters
    is used.
    """
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

    :param minio_client: MinIO client object used to interact with the MinIO server.
    :param bucket_name: Name of the bucket containing the file.
    :param object_name: Name of the object (file) in the bucket.
    :param file_path: Path where the downloaded file will be saved.
    """
    try:
        minio_client.fget_object(bucket_name, object_name, str(file_path))
    except S3Error as e:
        print(f"Error occurred: {e}")

def download_file_from_url(url: str, save_folder: str, filename: str = None, extension: str = '.stl', print_output=False) -> None:
    """
    Downloads a file from a URL and saves it to a specified folder.

    :param url: URL of the file to be downloaded.
    :param save_folder: Directory where the downloaded file will be saved.
    :param filename: Optional custom filename for the downloaded file. If None, filename is extracted from the URL.
    :param extension: File extension to be used if filename is not provided. Defaults to '.stl'.
    :param print_output: Whether to print progress messages.
    """
    
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
    If this API was found useful within your research please cite MedShapeNet:
     @article{li2023medshapenet,
     title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
     author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
     journal={arXiv preprint arXiv:2308.16139},
     year={2023}
     }
    
    This class holds methods to:
     - Setup connection with the database as an object through the MedShapeNet init, and create a directory to store related downloads. (def __init__)
     - Ask for help specifying all functions. (def help)
     - Get a list of all datasets included in MedShapeNet and stored on the S3 storage. (def datasets)
     - Print licence, citation and number of files for a dataset. (def dataset_info)
     - Get a list of files inside the dataset. (def dataset_files)
     - Download a file from a dataset. (def download_file)
     - Download seperate datasets completely based on the name. (def download_dataset)
     - Convert a STL file to a numpy mask. (def stl_to_npz)
     - Convert an already downloaded dataset to numpy masks stored in the same directory. (def dataset_stl_to_npz)
     - Download a shape from the database directly as numpy mask, conversion from stl to npz happens in memory. (def download_stl_as_numpy)
     - Download a complete dataset but get masks instead of stl files, conversion from stl to npz happens in memory. (def download_dataset_masks)
     - Search the original MedShapeNet (Sciebo storage) by name and get a list of all download urls, downside: you have to look up the reference and licence manually based on the paper. (def search_by_name)
     - Search and download the original MedShapeNet (Sciebo storage) by name, downside: you have to look up the reference and licence manually based on the paper. (def search_and_download_by_name)

     - *Under construction, more to come*
    '''
    # Initialize the class (minio)
    def __init__(self,
                minio_endpoint: str = "127.0.0.1:9000", # Local host
                # minio_endpoint: str = "10.49.131.44:9000", # Wireless LAN adaptor wifi 04/09/2024 -> open access will come soon.
                access_key: str = "admin", 
                secret_key: str = "ikim1234",
                secure: bool = False,
                timeout: int = 5
            ) -> None:
        """
        Initializes an instance of the class with a MinIO client and sets up a default download directory.
        Configures the MinIO client with the provided endpoint, access credentials, and security settings. Also creates a 
        local directory for storing downloaded files if it does not already exist.

        :param minio_endpoint: Endpoint for the MinIO server (e.g., 'localhost:9000'). 
                                Can also be set to a remote server address.
        :param access_key: Access key for authenticating with the MinIO server.
        :param secret_key: Secret key for authenticating with the MinIO server.
        :param secure: Boolean flag indicating whether to use HTTPS for communication with the MinIO server. Defaults to False.
        :param timeout: Timeout duration in seconds for connection and read operations. Defaults to 5 seconds.

        :raises urllib3.exceptions.TimeoutError: If the connection to MinIO times out.
        :raises urllib3.exceptions.HTTPError: For general HTTP errors during connection.
        :raises S3Error: If an error occurs related to MinIO operations.
        :raises socket.timeout: If a socket operation times out.

        Initializes the `self.minio_client` with the provided settings and verifies the connection by listing the buckets. 
        Creates the download directory at `self.download_dir` if it does not exist.
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
        Returns: None
        '''
        print_dynamic_line()
        print("""
                NOTE:
                This package is currently under heavy construction, more functionality will come soon!
                Current S3 access is for development only and reachable via https://xrlab.ikim.nrw wifi, full access will come soon!
                Want to contribute, contact me Gijs Luijten via LinkedIn, my work email or the MedShapeNet 2.0 GitHub.
                
                SAMPLE USAGE - With Jupyter Notebook examples:
                https://github.com/GLARKI/MedShapeNet2.0/tree/main/Samples

                CITE:
                If you used MedShapeNet within your research please CITE:
                @article{li2023medshapenet,
                title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
                author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
                journal={arXiv preprint arXiv:2308.16139},
                year={2023}
                }
              
                CALL DOCSTRING/INFO of MedShapeNet class or method (in Python) using:
                print(MedShapeNet.__doc__) OR print(MedShapeNet.'{'method_name'}'.__doc__)
              
                CURRENT (PYTHON) METHODS:
                    - def __init__: Setup connection with the database as an object through the MedShapeNet init, and create a directory to store related downloads.
                    - def help(): Ask for help specifying all functions.
                    - def datasets(): Get a list of all datasets included in MedShapeNet and stored on the S3 storage. 
                    - def dataset_info(): Print licence, citation and number of files for a dataset.
                    - def dataset_files(): Get a list of files inside the dataset.
                    - def download_file(): Download a file from a dataset
                    - def download_dataset(): Download seperate datasets completely based on the name.
                    - def stl_to_npz(): Convert a STL file to a numpy mask.
                    - def dataset_stl_to_npz(): Convert an already downloaded dataset to numpy masks stored in the same directory.
                    - def download_stl_as_numpy(): Download a shape from the database directly as numpy mask, conversion from stl to npz happens in memory.
                    - def download_dataset_masks(): Download a complete dataset but get masks instead of stl files, conversion from stl to npz happens in memory.
                    - def search_by_name(): Search the original MedShapeNet (Sciebo storage) by name and get a list of all download urls, downside: you have to look up the reference and licence manually based on the paper. 
                    - def search_and_download_by_name(): Search and download the original MedShapeNet (Sciebo storage) by name, downside: you have to look up the reference and licence manually based on the paper.

                COMMAND LINE INTERFACE
                    - Methods are callable via the command line interface as well.
                    e.g., call in cli using: python CLI.py download_file --bucket_name "example-bucket" --object_name "file.stl" --file_path "./local/path/"
                    CLI Methods:
                        'MedShapeNet': MedShapeNet,
                        'help':MedShapeNet.help,
                        'datasets':MedShapeNet.datasets,
                        'dataset_info': MedShapeNet.dataset_info,
                        'dataset_files':MedShapeNet.dataset_files,
                        'download_file':MedShapeNet.download_file,
                        'download_dataset':MedShapeNet.download_dataset,
                        'stl_to_npz':MedShapeNet.stl_to_npz,
                        'dataset_stl_to_npz':MedShapeNet.dataset_stl_to_npz,
                        'download_stl_as_numpy':MedShapeNet.download_stl_as_numpy,
                        'download_dataset_masks':MedShapeNet.download_dataset_masks,
                        'search_by_name':MedShapeNet.search_by_name,
                        'search_and_download_by_name':MedShapeNet.search_and_download_by_name

                Datasets within the s3 bucket:
                *under construction*
              
                """)
        
        print_dynamic_line()
        
        print("""
                METHODS IN DETAIL
              
                    - def __init__(self, minio_endpoint: str, minio_endpoint: str, access_key: str, secret_key: str, secure: bool, timeout: int) -> None:
                        Initializes an instance of the class with a MinIO client and sets up a default download directory.
                        Configures the MinIO client with the provided endpoint, access credentials, and security settings. Also creates a 
                        local directory for storing downloaded files if it does not already exist.

                        :param minio_endpoint: Endpoint for the MinIO server (e.g., 'localhost:9000'). 
                                                Can also be set to a remote server address.
                        :param access_key: Access key for authenticating with the MinIO server.
                        :param secret_key: Secret key for authenticating with the MinIO server.
                        :param secure: Boolean flag indicating whether to use HTTPS for communication with the MinIO server. Defaults to False.
                        :param timeout: Timeout duration in seconds for connection and read operations. Defaults to 5 seconds.

                        :raises urllib3.exceptions.TimeoutError: If the connection to MinIO times out.
                        :raises urllib3.exceptions.HTTPError: For general HTTP errors during connection.
                        :raises S3Error: If an error occurs related to MinIO operations.
                        :raises socket.timeout: If a socket operation times out.

                        Initializes the `self.minio_client` with the provided settings and verifies the connection by listing the buckets. 
                        Creates the download directory at `self.download_dir` if it does not exist.
              
                    - def help() -> None:
                        Prints a help message regarding current functionality of MedShapeNet API.
                        Returns: None
              
                    - def datasets(self, print_output: bool = False) -> list:
                        Retrieves a list of top-level datasets from the MinIO server. This includes both bucket names and top-level folders 
                        within the buckets. It excludes nested folders to provide a clear list of primary datasets.
                        This method first lists all buckets and then inspects each bucket to find top-level folders. It ensures that nested 
                        folders are not included in the final dataset list. 

                        :param print_output: If True, prints the list of datasets with corresponding indices. Defaults to False.
                        :return: A list of dataset names, including both bucket names and top-level folder names. Nested folders are excluded.
              
                    - def dataset_info(self, bucket_name: str) -> None:
                        Displays detailed information about a dataset stored in a specified bucket.
                        
                        This method performs the following actions:
                        1. Prints the contents of 'cite.txt' and 'licence.txt' files if they exist in the dataset.
                        2. Counts and prints the total number of '.txt', '.json', and '.stl' files in the dataset.

                        :param bucket_name: Name of the bucket or a path within a bucket from which to extract dataset information.
                        :raises S3Error: If an error occurs while accessing or reading files from the bucket.
                    
                    - def dataset_files(self, bucket_name: str, file_extension: str = None, print_output: bool = False) -> list:
                        Lists all files in a specified bucket and optionally filters by file extension.

                        This method performs the following actions:
                        1. Lists all files in the specified bucket, optionally filtering by a given file extension.
                        2. Counts and prints the total number of '.txt', '.json', and '.stl' files in the bucket.
                        3. Prints the file names and summary statistics if requested.

                        :param bucket_name: Name of the bucket or dataset to list files from. May include a folder prefix.
                        :param file_extension: (Optional) File extension to filter the files by (e.g., '.stl', '.txt', '.json'). If None, all files are listed.
                        :param print_output: Whether to print the file names and file type counts to the console.
                        :return: A list of file names in the bucket, filtered by the specified file extension if provided.
                        :raises S3Error: If an error occurs while listing objects from the bucket.
              
                    - def download_file(self, bucket_name: str, object_name: str, file_path: Path = None, print_output: bool = True) -> None:
                        Downloads a file from a specified bucket in MinIO to a local path.

                        This method handles both cases where the bucket name includes a folder path or not:
                        - If no file path is provided, it automatically creates a directory structure based on the bucket name and saves the file there.
                        - If a folder path is included in the bucket name, it extracts the bucket name and creates a subdirectory for the dataset.

                        :param bucket_name: Name of the bucket from which the file is to be downloaded. It may include a folder path.
                        :param object_name: Name of the file (object) within the bucket.
                        :param file_path: (Optional) Local path where the downloaded file should be saved. If None, the file is saved in a directory created based on the bucket name.
                        :param print_output: Whether to print a success message after the file is downloaded.
                        :raises S3Error: If an error occurs during the download process.
              
                    - def download_dataset(self, dataset_name: str, download_dir: Path = None, num_threads: int = 4, print_output: bool = True) -> list:
                        Downloads all files from a specified dataset to a local directory, with parallel downloading for increased speed.
                        Retries failed downloads once and returns a list of failed downloads after retry attempts.

                        :param dataset_name: Name of the dataset, which can be a bucket name or a folder within a bucket.
                        :param download_dir: Path to the local directory where files will be saved. If None, a default directory based on 
                                            the dataset name is created.
                        :param num_threads: Number of threads to use for parallel downloading.
                        :param print_output: Whether to print progress and error messages during the download process.
                        :return: A list of tuples, each containing (bucket_name, file, file_path) for downloads that failed after retrying.
              
                    - def stl_to_npz(self, stl_file: str, npz_file: str, print_output = True) -> None:
                        Converts an STL file to a NumPy .npz file, compressing the data.

                        :param stl_file: Path to the input .stl file containing 3D shape data.
                        :param npz_file: Path where the converted .npz file will be saved.
                        :param print_output: Whether to print a success message upon successful conversion

                    - def dataset_stl_to_npz(self, dataset: str, num_threads: int = 4, output_dir: Path = None, print_output: bool = False) -> None:                
                        Converts all .stl files in the specified dataset directory to NumPy .npz format and saves them in a designated output directory.
                        Additionally, copies any .txt and .json files from the dataset to the output directory.

                        :param dataset: Name of the dataset, which can be a bucket or folder name.
                        :param num_threads: Number of threads for parallel processing of .stl files.
                        :param output_dir: Directory to save the converted .npz files. Defaults to 'masks_numpy' within the dataset directory.
                        :param print_output: Whether to print progress messages during the conversion process.

                    - def download_stl_as_numpy(self, bucket_name: str, stl_file: str, output_dir: Path, print_output: True) -> None:
                        Downloads an STL file from a specified bucket, converts it to a NumPy .npz file, and saves it in the given directory.

                        :param bucket_name: Name of the bucket or dataset where the STL file is located.
                        :param stl_file: Path to the STL file within the bucket.
                        :param output_dir: Directory where the converted .npz file will be saved.
                        :param print_output: Whether to print progress messages.
              
                    - def download_dataset_masks(self, dataset_name: str, download_dir: Path = None, num_threads: int = 4, print_output: bool = True) -> list:
                        Downloads files from a dataset, converting STL files to NumPy masks and saving them as .npz files,
                        while directly saving non-STL files to the specified directory.

                        STL files are processed in memory and converted to NumPy arrays, which are saved as compressed .npz files.
                        Non-STL files (e.g., .txt, .json) are copied directly to the download directory.

                        :param dataset_name: Name of the dataset, which may include a bucket name and an optional folder path.
                        :param download_dir: Directory where the converted and non-STL files should be saved. If None, a default directory
                                            within the class's download directory is used.
                        :param num_threads: Number of threads for parallel downloading and processing.
                        :param print_output: Whether to print progress messages.
                        :return: A list of STL files that failed to download or convert, even after retry attempts.
              
                    - def search_and_download_by_name(self, name: str = None, max_threads: int = 5, save_folder: Path = None, extension: str = '.stl', print_output = True) -> list:
                        This is a function that still uses the download links from sciebo and will be eventually replaced.
                        Make sure to look up the correct citations/licence per shape based on the paper "https://arxiv.org/abs/2308.16139"
                        
                        Searches for and downloads files based on a name search. STL files will be downloaded in parallel.

                        The method searches for files related to the specified name, downloads them, and saves them in the designated folder.
                        Failed downloads are retried.

                        :param name: The name of the organ or disease to search for.
                        :param max_threads: Maximum number of threads to use for downloading files.
                        :param save_folder: Path to the folder where the found STL files should be saved. If None, a folder named after the search query with "_stl" appended will be created.
                        :param extension: File extension to use when saving files (default is '.stl').
                        :param print_output: Whether to print progress messages.
                        :return: A list of URLs for downloads that failed even after retry attempts.
              
                HELPER FUNCTIONS WITHIN THE CLASS
                - def print_dynamic_line():
                    Prints a line of underscores across the terminal width for visual separation between commands.
                    This function determines the terminal width and prints a line of underscores to enhance readability between terminal outputs.
                    If the terminal width cannot be determined, a default width of 80 characters is used.

                - def download_file(minio_client: Minio, bucket_name: str, object_name: str, file_path: Path) -> None:
                    Downloads a file from a specified bucket in MinIO.
                    :param minio_client: MinIO client object used to interact with the MinIO server.
                    :param bucket_name: Name of the bucket containing the file.
                    :param object_name: Name of the object (file) in the bucket.
                    :param file_path: Path where the downloaded file will be saved.

                - def download_file_from_url(url: str, save_folder: str, filename: str = None, extension: str = '.stl', print_output=False) -> None:
                    Downloads a file from a URL and saves it to a specified folder.
                    :param url: URL of the file to be downloaded.
                    :param save_folder: Directory where the downloaded file will be saved.
                    :param filename: Optional custom filename for the downloaded file. If None, filename is extracted from the URL.
                    :param extension: File extension to be used if filename is not provided. Defaults to '.stl'.
                    :param print_output: Whether to print progress messages.
                """)
        print_dynamic_line()


    # Create a list of datasets within the S3 storage
    def datasets(self, print_output: bool = False) -> list:
        """
        Retrieves a list of top-level datasets from the MinIO server. This includes both bucket names and top-level folders 
        within the buckets. It excludes nested folders to provide a clear list of primary datasets.

        This method first lists all buckets and then inspects each bucket to find top-level folders. It ensures that nested 
        folders are not included in the final dataset list. 

        :param print_output: If True, prints the list of datasets with corresponding indices. Defaults to False.

        :return: A list of dataset names, including both bucket names and top-level folder names. Nested folders are excluded.
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
        Displays detailed information about a dataset stored in a specified bucket.

        This method performs the following actions:
        1. Prints the contents of 'cite.txt' and 'licence.txt' files if they exist in the dataset.
        2. Counts and prints the total number of '.txt', '.json', and '.stl' files in the dataset.

        :param bucket_name: Name of the bucket or a path within a bucket from which to extract dataset information.
        :raises S3Error: If an error occurs while accessing or reading files from the bucket.
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
                    print("\nLICENCE INFO:")
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
        Downloads a file from a specified bucket in MinIO to a local path.

        If file_path is a full file path (e.g., a temporary file), it skips the directory creation logic.

        This method handles both cases where the bucket name includes a folder path or not:
        - If no file path is provided, it automatically creates a directory structure based on the bucket name and saves the file there.
        - If a folder path is included in the bucket name, it extracts the bucket name and creates a subdirectory for the dataset.

        :param bucket_name: Name of the bucket from which the file is to be downloaded. It may include a folder path.
        :param object_name: Name of the file (object) within the bucket.
        :param file_path: (Optional) Local path where the downloaded file should be saved. If None, the file is saved in a directory created based on the bucket name.
        :param print_output: Whether to print a success message after the file is downloaded.

        :raises S3Error: If an error occurs during the download process.
        """
        # To later print the dataset name which it is downloaded from.
        dataset = bucket_name


        # Handle if file_path is not choosen -> i.e. use default folder from the self.download_dir
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
                file_path = bucket_dir / object_name.split('/')[-1]
                bucket_name = bucket_name.split('/')[0]


            else:
                # Handle case where bucket_name does not include folder path
                bucket_dir = self.download_dir / bucket_name
                bucket_dir.mkdir(parents=True, exist_ok=True)
                file_path = bucket_dir / object_name
        else:
            # to later check if file_path is a directory or a temporary file used in other functions
            file_path = Path(file_path)

             # If the file_path has a suffix, it's a file (like a temporary file), so skip directory creation logic
            if file_path.suffix:
                # Proceed with the download immediately, skipping directory creation
                pass
            else:
                if '/' in bucket_name:
                    # Get the propper bucket name
                    bucket_name = bucket_name.split('/')[0]

                # create the file_path if it doesn't exist
                file_path = Path(file_path) 
                if not file_path.exists():
                    file_path.mkdir(parents=True, exist_ok=True)

                # create the file path of the file
                file_path = file_path / object_name.split('/')[-1]

        
        try:
            self.minio_client.fget_object(bucket_name, object_name, str(file_path))
            if print_output:
                print(f"'{object_name}', from dataset '{dataset}', successfully downloaded to '{file_path}'")
        except S3Error as e:
            print(f"Error occurred: {e}")


    # Download a dataset (multithreathed -> increase download speed with factor 2)
    def download_dataset(self, dataset_name: str, download_dir: Path = None, num_threads: int = 4, print_output: bool = True) -> list:
        """
        Downloads all files from a specified dataset to a local directory, with parallel downloading for increased speed.
        Retries failed downloads once and returns a list of failed downloads after retry attempts.

        :param dataset_name: Name of the dataset, which can be a bucket name or a folder within a bucket.
        :param download_dir: Path to the local directory where files will be saved. If None, a default directory based on 
                            the dataset name is created.
        :param num_threads: Number of threads to use for parallel downloading.
        :param print_output: Whether to print progress and error messages during the download process.
        :return: A list of tuples 'failed_retry_downloads', each containing (bucket_name, file, file_path) for downloads that failed after retrying.
        """

        # Initialize for retry here so in case a 'try' doesn't work, it won't error out on the return.
        failed_retry_downloads = []
        
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
            else:
                download_dir = Path(download_dir)

            # Get a list of all files in the dataset
            files = self.dataset_files(dataset_name)  # Assuming this function returns all files including paths
            
            # num of files
            num_of_files = len(files)

            # list to store futures (multithreading)
            futures = []
            failed_downloads = []

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
                            failed_downloads.append((bucket_name, file, file_path))
                        pbar.update(1)

            # Retry failed downloads if any
            if failed_downloads:
                print(f"Retrying {len(failed_downloads)} failed downloads...")
                with tqdm(total=len(failed_downloads), desc="Retrying failed downloads") as retry_pbar:
                    with ThreadPoolExecutor(max_workers=num_threads) as retry_executor:
                        retry_futures = []
                        for bucket_name, file, file_path in failed_downloads:
                            retry_future = retry_executor.submit(download_file, self.minio_client, bucket_name, file, file_path)
                            retry_futures.append(retry_future)

                        for retry_future, (bucket_name, file, file_path) in zip(as_completed(retry_futures), failed_downloads):
                            try:
                                retry_future.result()
                            except Exception as retry_error:
                                print(f"Retry failed for {file}: {retry_error}")
                                failed_retry_downloads.append(bucket_name, file, file_path)
                            retry_pbar.update(1)
            
            # Print dataset info to make sure the researchers sees the citations and licence info.
            self.dataset_info(dataset_name)
        
        except S3Error as e:
            print(f"Error occurred: {e}")

        if print_output:
            print(f'\nDataset {dataset_name} is downloaded with {len(failed_retry_downloads)} failures')

        return failed_retry_downloads


    # Convert .stl (STL format) to .npz (NumPy compressed)
    def stl_to_npz(self, stl_file: str, npz_file: str, print_output = True) -> None:
        """
        Converts an STL file to a NumPy .npz file, compressing the data.

        :param stl_file: Path to the input .stl file containing 3D shape data.
        :param npz_file: Path where the converted .npz file will be saved.
        :param print_output: Whether to print a success message upon successful conversion
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
        Converts all .stl files in the specified dataset directory to NumPy .npz format and saves them in a designated output directory.
        Additionally, copies any .txt and .json files from the dataset to the output directory.

        :param dataset: Name of the dataset, which can be a bucket or folder name.
        :param num_threads: Number of threads for parallel processing of .stl files.
        :param output_dir: Directory to save the converted .npz files. Defaults to 'masks_numpy' within the dataset directory.
        :param print_output: Whether to print progress messages during the conversion process.
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
            try:
                # Convert each STL to NPZ with a correlating name
                npz_file = output_dir / (file_path.stem + '.npz')  # Create the output .npz file path
                self.stl_to_npz(str(file_path), str(npz_file),  False)
            except Exception as e:
                print(f"Failed to convert {file_path}: {e}")

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
    def download_stl_as_numpy(self, bucket_name: str, stl_file: str, output_dir: Path = None, print_output: bool = True) -> None:
        """
        Downloads an STL file from a specified bucket, converts it to a NumPy .npz file, and saves it in the given directory.

        :param bucket_name: Name of the bucket or dataset where the STL file is located.
        :param stl_file: Path to the STL file within the bucket.
        :param output_dir: Directory where the converted .npz file will be saved.
        :param print_output: Whether to print progress messages.
        """

        # Handle a folder inside a bucket
        if '/' in bucket_name:
            # Handle case where bucket_name includes folder path
            bucket_name, dataset = bucket_name.split('/')
        else:
            dataset = bucket_name

        # create the default output dir if none is set
        if output_dir is None:
            output_dir = self.download_dir / dataset / 'masks_numpy'

        # Convert to path and create the directory
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download the STL file to a temporary location in memory
            with tempfile.NamedTemporaryFile(suffix='.stl') as temp_stl:
                self.download_file(bucket_name, stl_file, temp_stl.name, print_output=False)
                temp_stl.flush()  # Ensure the file is written properly

                # Load the STL file
                stl_mesh = mesh.Mesh.from_file(temp_stl.name)

                # Extract vertices and faces
                vertices = stl_mesh.vectors.reshape(-1, 3)
                faces = np.arange(len(vertices)).reshape(-1, 3)

                # Save vertices and faces into the .npz file
                npz_file = output_dir / (Path(stl_file).stem + '.npz')
                np.savez_compressed(npz_file, vertices=vertices, faces=faces)

            if print_output:
                print(f"Downloaded and converted STL: {stl_file} -> to numpy, from dataset: {dataset} and stored in: {npz_file}")

        except Exception as e:
            print(f"An error occurred while processing STL file {stl_file}: {e}")
        

    # download the dataset stl and convert it to npz in memory and only return the dataset with npz files instead of stl files
    def download_dataset_masks(self, dataset_name: str, download_dir: Path = None, num_threads: int = 4, print_output: bool = True) -> list:
        """
        Downloads files from a dataset, converting STL files to NumPy masks and saving them as .npz files,
        while directly saving non-STL files to the specified directory.

        STL files are processed in memory and converted to NumPy arrays, which are saved as compressed .npz files.
        Non-STL files (e.g., .txt, .json) are copied directly to the download directory.

        :param dataset_name: Name of the dataset, which may include a bucket name and an optional folder path.
        :param download_dir: Directory where the converted and non-STL files should be saved. If None, a default directory
                            within the class's download directory is used.
        :param num_threads: Number of threads for parallel downloading and processing.
        :param print_output: Whether to print progress messages.

        :return: A list of STL files that failed to download or convert, even after retry attempts.
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
                download_dir = Path(download_dir)
                download_dir.mkdir(parents=True, exist_ok=True)
                masks_numpy_dir = download_dir

            # Get list of all paths of all files in the dataset
            files = self.dataset_files(dataset_name)

            # get stl and non-stl file paths
            stl_files = [file for file in files if file.lower().endswith('.stl')]
            other_files = [file for file in files if not file.lower().endswith('.stl')]

            # Multithreaded download and converting
            failed_download_and_conversion = []
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
                             # Determine which file failed
                            if future in futures[len(other_files):]:  # Check if it's in the STL futures
                                failed_download_and_conversion.append(stl_files[futures.index(future) - len(other_files)])
                        pbar.update(1)

            # retry failures if any
            if failed_download_and_conversion:
                print(f"Retrying {len(failed_download_and_conversion)} failed downloads...")
                num_of_files = len(failed_download_and_conversion)
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    with tqdm(total=num_of_files, desc="Downloading and processing files", unit="file") as pbar:
                        futures = []
                        # Handle STL files (download in memory and convert to NumPy using a helper function)
                        for stl_file in failed_download_and_conversion:
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
                print(f"Dataset {dataset_name} STL files converted to NumPy masks with {len(failed_download_and_conversion)} failures.")
                print(f"Other files (licence, citation and labels) and all STL files are stored in {masks_numpy_dir}.")

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
    def search_and_download_by_name(self, name: str = None, max_threads: int = 5, save_folder: Path = None, extension: str = '.stl', print_output = True) -> list:
        '''
        This is a function that still uses the download links from sciebo and will be eventually replaced.
        Make sure to look up the correct citations/licence per shape based on the paper "https://arxiv.org/abs/2308.16139"

        Searches for and downloads files based on a name search. STL files will be downloaded in parallel.

        The method searches for files related to the specified name, downloads them, and saves them in the designated folder.
        Failed downloads are retried.

        :param name: The name of the organ or disease to search for.
        :param max_threads: Maximum number of threads to use for downloading files.
        :param save_folder: Path to the folder where the found STL files should be saved. If None, a folder named after the search query with "_stl" appended will be created.
        :param extension: File extension to use when saving files (default is '.stl').
        :param print_output: Whether to print progress messages.

        :return: A list of URLs for downloads that failed even after retry attempts.
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

        # track failed downloads
        failed_downloads = []

        # Prepare the progress bar
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit the download tasks and track their progress with tqdm
            futures = {executor.submit(download_file_from_url, url = url, save_folder = save_folder, filename = None, extension = extension, print_output = False): url
                    for url in matched_urls}
            
            # Use tqdm to show the progress bar
            for future in tqdm(as_completed(futures), total=len(matched_urls), desc="Downloading files"):
                url = futures[future]
                try:
                    future.result()  # Retrieve the result to raise any exceptions
                except Exception as e:
                    print(f"Error occurred for URL {url}: {e}")
                    failed_downloads.append(url)

        # Retry failed downloads if any
        if failed_downloads:
            print(f"Retrying {len(failed_downloads)} failed downloads...")
            retry_failed_downloads = []
            for url in failed_downloads:
                try:
                    download_file_from_url(url=url, save_folder=save_folder, filename=None, extension=extension, print_output=False)
                except Exception as e:
                    print(f"Retry failed for URL {url}: {e}")
                    retry_failed_downloads.append(url)

            failed_downloads = retry_failed_downloads
        
        if print_output:
            print(f'Download complete! Files are stored in folder: {save_folder}')
            print(f'{len(failed_downloads)} downloads failed.')

        return failed_downloads
    

        # work / testing in progress
    
    
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
    # This is not intented to run solo, it's a module: Print help and citation info:
    print("You are running the main.py from MedShapeNet directly,\nPlease install the PYPI MedShapeNet package 'pip install MedShapeNet'.\nUse it in your .py or .ipynb scripts there 'from MedShapeNet import MedShapeNet'.\n")
    MedShapeNet.help()
    print(
        '''
        If this API was found useful within your research please cite MedShapeNet:

        @article{li2023medshapenet,
        title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
        author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
        journal={arXiv preprint arXiv:2308.16139},
        year={2023}
        }
        '''
    )
    