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
    echo "    --docker_user                      Username for the Docker images. "
    echo ""
    echo "    --fpath_custom_dockerfile fpath:   Build a custom Docker image from a provided "
    echo "                                       Dockerfile after the Dockerfile.base and "
    echo "                                       Dockerfile image builds, meant to allow for "
    echo "                                       further customization of a user's environment. "
    echo "                                       This Dockerfile must use dl-playground/final as "
    echo "                                       the base image."
    exit 1
}

while [ ! $# -eq 0 ]
do 
    case "$1" in
            --conda_version)
                CONDA_VERSION=$2
                shift 2
                ;;
            --fpath_custom_dockerfile)
                FPATH_CUSTOM_DOCKERFILE=$2
                shift 2
                ;;
            --docker_user)
                DOCKER_USER=$2
                shift 2
                ;;
            --help)
                usage
                shift
                ;;
    esac
done

if [ -z "$DOCKER_USER" ]; then
    echo "--docker_user flag must be specified!"
    exit 1
fi

if [ -z "$CONDA_VERSION" ]; then
    echo "--conda_version not specified, using the default of 4.5.1."
    CONDA_VERSION=4.5.1
fi

echo "Creating images with docker username $DOCKER_USER and miniconda version $CONDA_VERSION..."
docker build --build-arg user=$DOCKER_USER --build-arg conda_version=$CONDA_VERSION -t dl-playground/base --file ./Dockerfile.base ./
docker build --build-arg user=$DOCKER_USER --build-arg conda_version=$CONDA_VERSION -t dl-playground/final --file ./Dockerfile ./
if [ ! -z "$FPATH_CUSTOM_DOCKERFILE" ]; then
    echo "Building custom Docker image based off of $FPATH_CUSTOM_DOCKERFILE ..."
    docker build --build-arg user=$DOCKER_USER -t dl-playground/custom --file $FPATH_CUSTOM_DOCKERFILE ./
fi
