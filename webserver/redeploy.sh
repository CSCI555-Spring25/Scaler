docker build -t simpleweb:latest .
docker tag simpleweb:latest 10.10.1.1:5000/simpleweb:latest
docker push 10.10.1.1:5000/simpleweb:latest
kubectl rollout restart deployment simpleweb-deployment
