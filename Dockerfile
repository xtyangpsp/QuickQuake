# Use a minimal image with conda
FROM continuumio/miniconda3:23.3.1-0

# Working directory inside the container
WORKDIR /workspace

# Copy the explicit environment spec generated on your laptop
COPY qq_spec.txt /tmp/qq_spec.txt

# Create the conda environment inside the container using the explicit spec
# This avoids heavy "Solving environment" steps
RUN conda create -y -n QuickQuake --file /tmp/qq_spec.txt && \
    conda clean --all --yes

# Copy ALL QuickQuake source code into the container
COPY . /workspace

# Make the QuickQuake environment the default for any command
ENV PATH="/opt/conda/envs/QuickQuake/bin:${PATH}"

# Default command when starting the container.
CMD ["bash"]

