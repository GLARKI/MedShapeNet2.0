# Simple MinIO Object Storage in Docker
### Develop API Functions Locally
A simple Docker Compose and environment file to set up your own MinIO.<br>This can be helpful when developing and debugging API methods or creating specific classes.<br><br>
To work locally with MedShapeNet use the datasets from [Test_datasets_for_MinIO_bucket](https://github.com/GLARKI/MedShapeNet2.0/tree/main/minio/Test_datasets_for_MinIO_bucket) with the MinIO setup.<br>[pip install PYPI package MedShapeNet 0.1.8](https://pypi.org/project/MedShapeNet/0.1.8/) works with this setup -> 'pip install MedShapeNet==0.1.8'; later versions might not point to local host and the 'MINIO_SERVER_URL' in the .env should be adapted for local development.<br><Br>
These test shapes are made using the data from the [MedShapeNetCore dataset](https://zenodo.org/records/10423181) which is a lightweight subset of [MedShapeNet Database](https://medshapenet-ikim.streamlit.app/Download) / [website](https://medshapenet.ikim.nrw/) and therefore are suboptimal.

**Setup Instructions**
- Install Docker Desktop.
- Navigate to the MinIO directory in your command line interface (CMD).
- Run: docker-compose up -d.
- Open your browser and go to: localhost:9001
- Configure/upload the datasets within [Test_datasets_for_MinIO_bucket](https://github.com/GLARKI/MedShapeNet2.0/tree/main/minio/Test_datasets_for_MinIO_bucket)<br><br>
- Check available ports and status: docker-compose ps.
- Monitor Docker logs in real-time: docker-compose logs -f.
- Restart Docker: docker-compose down && docker-compose up -d.
- To stop Docker: docker-compose down.<br><Br>
Awesome tutorial for [setting this up](https://www.youtube.com/watch?v=2SDgIyrXmKc), or for [replicating/triple redundancy](https://www.youtube.com/watch?v=7fE4JayU5IU) by Medium Guy on YouTube.
