FROM dl-playground/base
LABEL maintainer="Sean Sall <ssall@alumni.nd.edu>"

ARG branch
ARG conda_version
ARG fname_environment_yml
ARG user

ENV CONDA_DIRPATH /opt/conda

USER $user

RUN mkdir $HOME/repos && \
    cd $HOME/repos && \
    git clone https://github.com/sallamander/dl-playground.git

RUN cd $HOME/repos/dl-playground/setup && \
    git checkout $branch && \
    conda install conda=$conda_version && \
    conda env create -f $fname_environment_yml

RUN /bin/bash -c "source activate dl-playground && python -m ipykernel install --user --name dl-playground --display-name 'Python (dl-playground)'" && \
    cd $HOME

RUN mkdir -p ~/.config/matplotlib && \
    echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
