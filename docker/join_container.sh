#!/bin/bash
# This script joins this terminal to specific docker container
 
# Display all container's name
echo "List of containers:"
declare -a arr
i=0
 
# Make container name into an array
containers=$(docker ps -a | awk '{if(NR>1) print$NF}')
for container in $containers
do
    arr[$i]=$container
    let "i+=1"
done
 
# Loop through name array
let "i-=1"
for j in $(seq 0 $i)
do
    echo $j")" ${arr[$j]}
done
 
# Obtain container name
read -p "Container name to be connected: " CONTAINERNAME
 
if [[ -z ${arr[$CONTAINERNAME]} ]]
then
    if [ "$(docker ps -aq -f name=$CONTAINERNAME)" ]; then
        echo "Container $CONTAINERNAME exists but is not running. Starting container..."
        docker start $CONTAINERNAME
        docker exec --privileged -e DISPLAY=${DISPLAY} -ti $CONTAINERNAME bash
    else
        echo "Container $CONTAINERNAME does not exist. Please enter a valid container name."
        exit 1
    fi
else
    running_container=${arr[$CONTAINERNAME]}
    if [ "$(docker ps -q -f name=$running_container)" ]; then
        echo "Connecting to running container $running_container..."
        docker exec --privileged -e DISPLAY=${DISPLAY} -ti $running_container bash
    else
        echo "Container $running_container exists but is not running. Starting container..."
        docker start $running_container
        docker exec --privileged -e DISPLAY=${DISPLAY} -ti $running_container bash
    fi
fi

