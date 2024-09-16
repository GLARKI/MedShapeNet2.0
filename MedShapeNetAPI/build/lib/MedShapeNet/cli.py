import fire
from main import MedShapeNet
import sys

class DevNull:
    def write(self, msg):
        pass

sys.stderr = DevNull()

def main():
    fire.Fire({
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
    })

if __name__ == '__main__':
    main()
    # Call in cli using: python CLI.py download_file --bucket_name "example-bucket" --object_name "file.stl" --file_path "./local/path/"
