# main.py

# Imports
# Imports minio
from minio import Minio
from minio.error import S3Error
# Handle paths system agnostic
from pathlib import Path
# Imports (save) multithread
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                # minio_endpoint: str = "127.0.0.1:9000", # Local host
                minio_endpoint: str = "10.49.131.44:9000", # Wireless LAN adaptor wifi 04/09/2024 -> open access will come soon.
                access_key: str = "msn_user_readwrite", 
                secret_key: str = "ikim1234",
                secure: bool = False
            ) -> None:
        """
        Initializes the MedShapeNet instance with a MinIO client and sets up a download directory.

        :param minio_endpoint: MinIO server endpoint (e.g., 'localhost:9000').
        :param access_key: Access key for MinIO.
        :param secret_key: Secret key for MinIO.
        :param secure: Whether to use HTTPS (default is False).
        """
        # Create and connect minio client
        self.minio_client = Minio(minio_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

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
    

    # List all buckets, i.e. all datasets on the S3 (MinIO) storage
    def datasets(self, print_output: bool = False) -> list:
        """
        Lists all buckets in the MinIO server via print statement
        Return: A list of bucket names 'list_of_buckets'
        """
        # Create list to return
        list_of_buckets = []

        try:
            # Get the buckets object from minio
            buckets = self.minio_client.list_buckets()

            # Warning if there are no buckets found
            if not buckets:
                print('No buckets found, please contact the owner of the database.')
                return list_of_buckets

            # Print buckets and add to list
            for bucket in buckets:
                list_of_buckets.append(bucket.name)
                if print_output:
                    print(f"Bucket: {bucket.name}")

        except S3Error as e:
            print(f"Error occurred: {e}")

        return list_of_buckets
    

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
            # Use dataset_files to get a list of all files in the bucked for file counts and details
            files = self.dataset_files(bucket_name)

            # Initialize counters for different file types
            txt_count = 0
            json_count = 0
            stl_count = 0

            # Read and print the contents of cite.txt and licence.txt if available
            for obj in files:
                if obj.endswith("licence.txt"):
                    licence_content = self.minio_client.get_object(bucket_name, obj)
                    print("\nLICENCE INFO:")
                    print(licence_content.read().decode('utf-8'))
                elif obj.endswith("cite.txt"):
                    cite_content = self.minio_client.get_object(bucket_name, obj)
                    print("\nCITATION INFO:")
                    print(cite_content.read().decode('utf-8'))
                
                # Count file types
                if obj.endswith('.txt'):
                    txt_count += 1
                elif obj.endswith('.json'):
                    json_count += 1
                elif obj.endswith('.stl'):
                    stl_count += 1

            # Print total count of specific file types with aligned formatting
            print(f"\n{'Total .txt files (how to cite, license file):':<45} {txt_count:>5}")
            print(f"{'Total .json files (labels, if in dataset as json):':<45} {json_count:>5}")
            print(f"{'Total .stl files (shapes):':<45} {stl_count:>5}")

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

        try:
            # List all objects in the bucket
            objects = self.minio_client.list_objects(bucket_name)

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

            return files_list

        except S3Error as e:
            print(f"Error occurred: {e}")
            return []


    # Multi-threaded downloading from the S3 (MinIO) storage
    # The bucket is currently hosted locally and thus not available for others until I'm granted the storage solution from work.
    def download_file(self, bucket_name: str, object_name: str, file_path: Path = None) -> None:
        """
        Downloads a file from a specified bucket in MinIO.

        :param bucket_name: Name of the bucket where the file is located.
        :param object_name: Name of the object in MinIO.
        :param file_path: Path to save the downloaded file.
        """
        if file_path is None:
            # Create a directory for the bucket inside the download directory
            bucket_dir = self.download_dir / bucket_name
            bucket_dir.mkdir(parents=True, exist_ok=True)
            file_path = bucket_dir / object_name
        
        try:
            self.minio_client.fget_object(bucket_name, object_name, str(file_path))
            print(f"'{object_name}' successfully downloaded to '{file_path}'")
        except S3Error as e:
            print(f"Error occurred: {e}")



# Entry point for direct execution
if __name__ == "__main__":
    # Print the help statement directly
    print("You are running the main.py from MedShapeNet directly, please install the PYPI 'MedShapeNet' package, import MedShapeNet and its methods in your python script.")
    
    print("\n")
    msn = MedShapeNet()
    msn.help()

    print("\n")
    list_of_datasets = msn.datasets()
    print('Example: List of datasets within the S3 storage accessing the first dataset from the list:')
    print(list_of_datasets)
    print(list_of_datasets[0])

    for dataset in list_of_datasets:
        print("\n")
        list_of_files = msn.dataset_files(dataset, print_output=False) # Print output is optional
        print(f"files in {dataset}:\n{list_of_files}\n")
        list_of_stl_files = msn.dataset_files(dataset, '.stl', print_output=False)
        print(f"STL files in {dataset}:\n{list_of_stl_files}\n")
        list_of_json_files = msn.dataset_files(dataset, '.json', print_output=False)
        print(f"JSON files in {dataset}:\n{list_of_json_files}\n")
        list_of_files = msn.dataset_files(dataset, '.txt', print_output=False)
        print(f"TXT files in {dataset}:\n{list_of_files}\n")
    
    print('\n')
    msn.dataset_info(dataset)



        