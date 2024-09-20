# cli.py to open up the medshapenet commands to a command line interface as well

import fire
from .main import MedShapeNet

def main():
    # Create an instance of MedShapeNet
    msn_instance = MedShapeNet()
    
    # Expose methods of the instance to the CLI
    fire.Fire({
        'help': msn_instance.help,
        'search_by_name': msn_instance.search_by_name,
        'search_and_download_by_name': msn_instance.search_and_download_by_name,
        'datasets': msn_instance.datasets,
        'dataset_info': msn_instance.dataset_info,
        'dataset_files': msn_instance.dataset_files,
        'download_file': msn_instance.download_file,
        'download_dataset': msn_instance.download_dataset,
        'stl_to_npz': msn_instance.stl_to_npz,
        'dataset_stl_to_npz': msn_instance.dataset_stl_to_npz,
        'download_stl_as_numpy': msn_instance.download_stl_as_numpy,
        'download_dataset_masks': msn_instance.download_dataset_masks
    })

if __name__ == '__main__':
    main()

    # Calls to functions in the command like interface can be made like:
    # msn search_by_name --name "face" 
    # msn search_by_name --name "face" > face_result.json