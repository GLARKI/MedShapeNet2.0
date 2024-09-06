# main.py

# Imports
# Imports minio
from minio import Minio
from minio.error import S3Error
# Imports (save) multithread
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
                minio_endpoint: str = "10.49.131.44", # Wireless LAN adaptor wifi 04/09/2024 -> open access will come soon.
                access_key: str = "msn_user_readwrite", 
                secret_key: str = "ikim1234",
                secure: bool = False
            ) -> None:
        """
        Initializes the MedShapeNet instance with a MinIO client and sets up a download directory.

        :param minio_endpoint: MinIO server endpoint (e.g., 'localhost:9000').
        :param access_key: Access key for MinIO.
        :param secret_key: Secret key for MinIO.
        :param secure: Whether to use HTTPS (default is False) -> no encryption needed -> thus faster.
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
    def msn_help() -> None:
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
    def msn_list_datasets(self) -> None:
        """
        Lists all buckets in the MinIO server.
        """
        try:
            buckets = self.minio_client.list_buckets()
            for bucket in buckets:
                print(f"Bucket: {bucket.name}")
        except S3Error as e:
            print(f"Error occurred: {e}")

    # Multi-threaded downloading from the S3 (MinIO) storage
    # The bucket is currently hosted locally and thus not available for others until I'm granted the storage solution from work.
    def msn_download_file(self, bucket_name: str, object_name: str, file_path: Path = None) -> None:
        """
        Downloads a file from a specified bucket in MinIO.

        :param minio_client: Minio client object.
        :param bucket_name: Name of the bucket where the file is located.
        :param object_name: Name of the object in MinIO.
        :param file_path: Path to save the downloaded file.
        """
        if file_path is None:
            file_path = self.download_dir
        
        try:
            self.minio_client.fget_object(bucket_name, object_name, str(file_path))
            print(f"'{object_name}' successfully downloaded to '{file_path}'")
        except S3Error as e:
            print(f"Error occurred: {e}")

    # additional new method


# Entry point for direct execution
if __name__ == "__main__":
    # Print the help statement directly
    print("You are running the main.py from MedShapeNet directly, please install the PYPI 'MedShapeNet' package, import MedShapeNet and its methods in your python script.")
    msn = MedShapeNet()
    msn.msn_help()