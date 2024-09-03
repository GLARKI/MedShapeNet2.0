# Simple MinIO Object Storage in Docker
### Develop API Functions Locally
A simple Docker Compose and environment file to set up your own MinIO. This can be helpful when developing and debugging API methods or creating specific classes.<br><Br>
**Setup Instructions**
- Install Docker Desktop.
- Navigate to the MinIO directory.
- Run: docker-compose up -d in CMD/CLI.
- Open your browser and go to: localhost:9001 -> Configure your files and settings as desired.
- Check available ports and status: docker-compose ps.
- Monitor Docker logs in real-time: docker-compose logs -f.
- Restart Docker: docker-compose down && docker-compose up -d.
- To stop Docker: docker-compose down.<br><Br>
Awesome tutorial for [setting this up](https://www.youtube.com/watch?v=2SDgIyrXmKc), or for [replicating/triple redundancy](https://www.youtube.com/watch?v=7fE4JayU5IU) by Medium Guy on YouTube.
