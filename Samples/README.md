# Showcases of (research) projects using MedShapeNet (API)
*Under construction* <br>
This is the root folder for showcase (research) projects of MedShapeNet(API) usage.<br>
Each showcase has its own folder/ipynb within this directory, [Example of a showcase](https://github.com/glarki/medshapenet-feedback/tree/main/anatomy-completor)<br>

You can use the [GettingStarted.ipynb](https://github.com/GLARKI/MedShapeNet2.0/blob/main/Samples/GettingStarted/GettingStarted.ipynb) to see the basic functionality of the [MedShapeNet API](https://pypi.org/project/MedShapeNet/).

<br><br>
Currently we have the following showcases:

- GettingStarted
- PointCloudCompletion
- ClassificationWithMonaiAndTensorFlow
- Anatomy completer
- ... to be continued ...

| **Title**     | **Folder name**   | **Description**   | **Used subset Dataset of MedShapeNet**  |
| ------------- | ----------------- | ----------------- | ----------------- |
| [GettingStarted](https://github.com/GLARKI/MedShapeNet2.0/blob/main/Samples/GettingStarted/GettingStarted.ipynb) | GettingStarted | Install and demonstrate the current functionality of MedShapeNet (to be updated with the Transformation() Class) | All datsets can be used |
| [PointCloudCompletion](https://github.com/GLARKI/MedShapeNet2.0/blob/main/Samples/PointCloudCompletion) | PointCloudCompletion AE model | PC AE model demonstrating how to search the database for ribs, prepare the data, train the model and run inference, based on its [Foundation Model](https://github.com/jrHoss/MedShapeNet-Foundation-Model?tab=readme-ov-file)| Mainly [TotalSegmentator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10546353/) |
| [ClassificationWithMonaiAndTensorflow](https://github.com/glarki/medshapenet-feedback/tree/main/anatomy-completor)| ClassificationWIthMonaiAndTensorflow | A classification model to classify healthy and unhealthy shapes with Monai or Tensorflow. Minimal example, use other datasets e.g., KITS for more elaborate classification | Based on [MedShapeNetCore's](https://zenodo.org/records/10423181) (lightweight) version of [ASOCA](https://www.nature.com/articles/s41597-023-02016-2)|
| [Anatomy Completor](https://github.com/glarki/medshapenet-feedback/tree/main/anatomy-completor)|See link|Many-to-one map to complete missing organs and create pseudo-labels for wholebody CT-scans|[TotalSegmentator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10546353/)|
