# Official NVIDIA Triton Server container image as the base image
FROM nvcr.io/nvidia/tritonserver:23.06-py3

# Install additional dependencies
RUN pip install torch transformers

# Copy the model repository from the local machine to the image
COPY model_repository /models

# Command to start the Triton server
CMD ["tritonserver", "--model-repository=/models"]

<<<<<<< HEAD
=======
# Expose port
>>>>>>> df31064e46fab9ef8bc1dcc2c73cc1930b5e2f1f
EXPOSE 8000
