FROM continuumio/miniconda3:23.3.1-0

WORKDIR /workspace

COPY environment_docker.yml /tmp/environment.yml

RUN conda env create -n QuickQuake -f /tmp/environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/QuickQuake/bin:${PATH}"

COPY . /workspace

CMD ["bash"]
