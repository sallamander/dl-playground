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
    echo ""
    echo "    --rebuild_image                    Rebuild this image and any subsequent images that "
    echo "                                       use this one as base. Useful if you know that "
    echo "                                       something inside has changed (e.g. the conda "
    echo "                                       environment) but Docker won't be able to register "
    echo "                                       that change. Valid options are (base, final, custom)."
    exit 1
}

while [ ! $# -eq 0 ]
do 
    case "$1" in
            --conda_version)
                CONDA_VERSION=$2
                shift 2
                ;;
            --docker_user)
                DOCKER_USER=$2
                shift 2
                ;;
            --fpath_custom_dockerfile)
                FPATH_CUSTOM_DOCKERFILE=$2
                shift 2
                ;;
            --rebuild_image)
                REBUILD_IMAGE=$2
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

if [ ! -z "$REBUILD_IMAGE" ]; then
    if ! [[ "$REBUILD_IMAGE" =~ ^(base|final|custom)$ ]]; then
        echo "--rebuild_image option \"$REBUILD_IMAGE\" is not one of the "
        echo "accepted options (base, final, or custom). If you'd like to "
        echo "delete and remove one of the images, please specify one of "
        echo "these options."
        exit 1
    fi

    if [[ "$REBUILD_IMAGE" == "base" ]]; then
        echo "--rebuild_image equal to \"base\"... will delete any existing, "
        echo "dl-playground/base, dl-playground/final, and "
        echo "dl-playground/custom images to build them anew."
        DELETE_BASE=true
        DELETE_FINAL=true
        DELETE_CUSTOM=true
    elif [[ "$REBUILD_IMAGE" == "final" ]]; then
        echo "--rebuild_image equal to \"final\"... will delete any existing, "
        echo "dl-playground/final and dl-playground/custom images to build "
        echo "them anew."
        DELETE_FINAL=true
        DELETE_CUSTOM=true
    elif [[ "$REBUILD_IMAGE" == "custom" ]]; then
        echo "--rebuild_image equal to \"custom\"... will delete the "
        echo "dl-playground/custom image to build it anew."
        DELETE_CUSTOM=true
    fi

    BASE_IMAGE_EXISTS=$(docker images -q dl-playground/base)
    FINAL_IMAGE_EXISTS=$(docker images -q dl-playground/final)
    CUSTOM_IMAGE_EXISTS=$(docker images -q dl-playground/custom)

    if [[ "$DELETE_BASE" == "true" ]] && [[ ! -z $BASE_IMAGE_EXISTS ]]; then
        docker image rm dl-playground/base
    fi
    if [[ "$DELETE_FINAL" == "true" ]] && [[ ! -z $FINAL_IMAGE_EXISTS ]]; then
        docker image rm dl-playground/final
    fi
    if [[ "$DELETE_CUSTOM" == "true" ]] && [[ ! -z $CUSTOM_IMAGE_EXISTS ]]; then
        docker image rm dl-playground/custom
    fi
fi


echo "Creating images with docker username $DOCKER_USER and miniconda "
echo "version $CONDA_VERSION..."

docker build --build-arg user=$DOCKER_USER \
             --build-arg conda_version=$CONDA_VERSION \
             -t dl-playground/base --file ./Dockerfile.base ./

docker build --build-arg user=$DOCKER_USER \
             --build-arg conda_version=$CONDA_VERSION \
             -t dl-playground/final --file ./Dockerfile ./

if [ ! -z "$FPATH_CUSTOM_DOCKERFILE" ]; then
    echo "Building custom Docker image based off of "
    echo "$FPATH_CUSTOM_DOCKERFILE ..."
    docker build --build-arg user=$DOCKER_USER \
                 -t dl-playground/custom --file $FPATH_CUSTOM_DOCKERFILE ./
fi
