#!/bin/bash
#
# build_docker_images.sh builds the necessary docker images for setting up the 
# dl-playground environment and repository, using the Dockerfile.base and 
# Dockerfile files in this directory.

function usage {
    echo "Usage: build_docker_image.sh  [--conda_version version] [--fpath_custom_dockerfile fpath]"
    echo "                              [--user username]"
    echo ""
    echo "   --conda_version                     Conda version for the Docker images - defaults "
    echo "                                       to 4.5.1, which is compatible with the repository "
    echo "                                       YMLs."
    echo ""
    echo "    --fpath_custom_dockerfile fpath:   Build a custom Docker image from a provided "
    echo "                                       Dockerfile after the Dockerfile.base and "
    echo "                                       Dockerfile image builds, meant to allow for "
    echo "                                       further customization of a user's environment. "
    echo "                                       This Dockerfile must use dl-playground/final as "
    echo "                                       the base image."
    echo ""
    echo "    --user                             Username for the Docker images. "
    exit 1
}

while [ ! $# -eq 0 ]
do 
    case "$1" in
            --conda_version)
                CONDA_VERSION=$2
                echo $CONDA_VERSION
                shift 2
                ;;
            --fpath_custom_dockerfile)
                FPATH_CUSTOM_DOCKERFILE=$2
                shift 2
                ;;
            --user)
                USER=$2
                shift 2
                ;;
            --help)
                usage
                shift
                ;;
    esac
done

docker build --build-arg user=$USER --build-arg conda_version=$CONDA_VERSION -t dl-playground/base --file ./Dockerfile.base ./
docker build --build-arg user=$USER --build-arg conda_version=$CONDA_VERSION -t dl-playground/final --file ./Dockerfile ./
if [ ! -z "$FPATH_CUSTOM_DOCKERFILE" ]; then
    echo "Building custom Docker image based off of $fpath_custom_dockerfile..."
    docker build --build-arg user=$USER -t dl-playground/custom --file $FPATH_CUSTOM_DOCKERFILE ./
fi
