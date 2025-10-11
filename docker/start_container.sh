#!/bin/bash
# This script starts a specific service from a docker-compose.yml file.
 
# Check if docker-compose.yml exists in the current directory
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: 'docker-compose.yml' not found in the current directory."
    exit 1
fi
 
# Display all services defined in the docker-compose.yml file
echo "List of services in 'docker-compose.yml':"
declare -a arr
i=0
 
# Use 'docker compose config --services' to get a clean list of services
services=$(docker compose config --services)
for service in $services
do
    arr[$i]=$service
    let "i+=1"
done
 
# Loop through the services array to display the numbered list
let "i-=1"
for j in $(seq 0 $i)
do
    echo $j")" ${arr[$j]}
done
 
# Obtain the service name from user input
read -p "Service name or number to start: " SERVICE_IDENTIFIER
 
# Check if the input is a number corresponding to an index in the array
if [[ $SERVICE_IDENTIFIER =~ ^[0-9]+$ ]] && [[ -n ${arr[$SERVICE_IDENTIFIER]} ]]
then
    SERVICE_TO_START=${arr[$SERVICE_IDENTIFIER]}
else
    # If not a number, assume it's the service name directly
    SERVICE_TO_START=$SERVICE_IDENTIFIER
fi
 
# Validate that the selected service exists in the list
SERVICE_EXISTS=false
for service in "${arr[@]}"; do
    if [[ "$service" == "$SERVICE_TO_START" ]]; then
        SERVICE_EXISTS=true
        break
    fi
done
 
if [ "$SERVICE_EXISTS" == false ]; then
    echo "Error: Service '$SERVICE_TO_START' does not exist in 'docker-compose.yml'."
    exit 1
fi
 
echo "Starting container for service '$SERVICE_TO_START'..."
# Use 'docker compose up -d' to start the service in detached mode
docker compose up -d $SERVICE_TO_START
 
echo "Service '$SERVICE_TO_START' has been started in detached mode."
echo "You can check its status with 'docker compose ps' or view logs with 'docker compose logs -f $SERVICE_TO_START'."
