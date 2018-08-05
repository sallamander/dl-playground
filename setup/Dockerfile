FROM dl-playground/base
LABEL maintainer="Sean Sall <ssall@alumni.nd.edu>"

ARG fname_environment_yml
ARG conda_version
ARG user

USER $user

RUN mkdir $HOME/repos && \
    cd $HOME/repos && \
    git clone https://github.com/sallamander/dl-playground.git

RUN cd $HOME/repos/dl-playground/setup && \
    conda install conda=$conda_version && \
    conda env create -f $fname_environment_yml && \
    cd $HOME
