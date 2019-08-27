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
    git clone https://github.com/sallamander/dl-playground.git && \
    git clone https://github.com/sallamander/ktorch.git && \
    git clone https://github.com/sallamander/train-net.git

RUN cd $HOME/repos/dl-playground/setup && \
    git checkout $branch && \
    conda install conda=$conda_version && \
    conda env create -f $fname_environment_yml

RUN /bin/bash -c "cd $HOME/repos && source activate dlp && pip install -e ktorch && pip install -e train-net && cd dl-playground && pip install -e dl_playground"

RUN /bin/bash -c "source activate dlp && python -m ipykernel install --user --name dlp --display-name 'Python (dlp)'" && \
    cd $HOME

RUN mkdir -p ~/.config/matplotlib && \
    echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
