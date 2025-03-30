# How It Works

The controller:
1. Monitors the creation, update, and deletion of PredictiveAutoscaler custom resources
2. On a schedule (every 5 minutes by default), collects current pod counts
3. Calculates required pods using the predictive formula
4. Updates the target HPA's minReplicas
5. Updates the historical data with the weighted formula
6. Stores this data persistently for future predictions

This implementation provides a clean separation of code and configuration, making it easier to develop, test, and maintain the controller.

## Deployment Instructions

### Build and Push the Controller Image

Navigate to the controller directory
cd CSCI555/Scaler/controller

Build the Docker image
`docker build -t predictive-autoscaler:latest .`

Tag and push to your container registry
`docker tag predictive-autoscaler:latest anirudhr120100/csci555-predictive-autoscaler:latest`

`docker push anirudhr120100/csci555-predictive-autoscaler:latest`

### Update the controller deployment with your image registry
Edit /Scaler/deploy/controller-deployment.yaml 

### Apply CRD first
`kubectl apply -f Scaler/crd/predictive-autoscaler-crd.yaml`

### Apply RBAC
`kubectl apply -f Scaler/deploy/rbac.yaml`

### Apply controller deployment
`kubectl apply -f Scaler/deploy/controller-deployment.yaml`

### Check if the controller is running
`kubectl get pods -l app=predictive-autoscaler-controller`

### Apply the PredictiveAutoscaler instance
`kubectl apply -f Scaler/deploy/predictive-autoscaler-instance.yaml`

### Check if the PredictiveAutoscaler custom resource was created 
`kubectl get predictiveautoscalers` 
or
`kubectl get pa` (using the short name)

### Test the controller
#### Create the directory in your persistent volume if needed
`kubectl exec -it predictive-autoscaler-controller-xxx -- mkdir -p /data`

#### Copy the sample data file to your controller pod
`kubectl cp Scaler/test-data/realistic-traffic.json predictive-autoscaler-controller-xxx:/data/default_simpleweb-predictor_history.json`

#### Check the logs to see if the controller is using the data correctly
`kubectl logs -f deployment/predictive-autoscaler-controller`

#### Check if the HPA is being updated
`kubectl get hpa`

#### Check if the historical data is being updated
`kubectl get predictiveautoscalers`


# Kubernetes Cluster Deployment instructions on Cloudlab

Cloudlabs has pre-existing profiles. Select the K8s

1. Click Experiment -> Start experiment
2. Click change profile
3. Click on K8s profile from select window
4. Click confirm and next to the parameterize page. 
5. Make edits to the parameters(optional)
6. Click next to the Finalize Page.
-> Assign a name and cluster location
7. Click next to the schedule page.
8. Pick a time to deploy and click next

Once the cluster starts, click "extend" to extend the cluster
expiration by 7 days.

## Starting K8s and docker on the cluster

### SSH to node-0
Click on the node in the node graph
click on "Shell" from the pop-up menu

### Install Docker
#### Only do the following if docker is not installed.

Dependencies:
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

Docker GPG keys
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

Create the docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list

Install docker:
sudo apt install -y docker-ce docker-ce-cli containerd.io

Start docker
sudo systemctl enable docker
sudo systemctl start docker

### Install Kubectl
#### Only do the following if Kubernetes is not installed
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client


### Clone GitHub repo
#### Git installation
sudo apt install -y git

#### Clone the repo
git clone https://github.com/CSCI555-Spring25/Scaler.git


### Building and registering docker container on the registry

1. navigate to "webserver" folder
2. Create docker container using "docker build -t simpleweb:latest ."
3. Find the IP of the node/registry using the following command:
    docker ps | grep "registry:2"
4. If the local registry is not running, run the following to start the registry and go back to step 3: 
    "docker run -d -p 5000:5000 --name registry registry:2"
5. Once IP is known, for example 10.10.1.1
Tag and push the docker container to the registry
    docker tag simpleweb:latest 10.10.1.1:5000/simpleweb:latest
    docker push 10.10.1.1:5000/simpleweb:latest
6. If the ip address is not 10.10.1.1:5000, edit spec/template/spec/containers/image to have the correct IP address


### Starting kubernetes service
1. navigate to root directory of Scaler github repo
2. Start the webserver with kubernetes:
    kubectl apply -f echo-server.yaml
3. Check status of the running pods with "kubectl get pods"


# kubectl not returning nodes
If kubectl is giving an error while returning nodes, make sure there is a 
K8s profile for your user on cloud labs. Use following commands to 
copy the admin profile into your user directory.

sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
