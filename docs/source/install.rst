Installation
============

This guide provides three methods to install **PyTorch-SVGRender**.

Environment Setup
----------------

Choose one of the following installation methods that best suits your needs:

Method 1: Standard Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a straightforward installation, run these commands in the project's root directory:

.. code-block:: bash

    chmod +x script/install.sh
    bash script/install.sh

Method 2: Docker Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a containerized environment, use Docker:

.. code-block:: bash

    chmod +x script/run_docker.sh
    sudo bash script/run_docker.sh

Method 3: Python Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a manual Python setup:

.. code-block:: bash

    # Create and activate conda environment
    conda create -n svgrender python=3.10
    conda activate svgrender

    # Install the package
    python setup.py install

System Requirements
-----------------

- Python 3.10 or higher
- CUDA-capable GPU (optional, for accelerated rendering)
- Docker (only for Method 2)
- Git

Troubleshooting
--------------

If you encounter any issues during installation:

1. Ensure all prerequisites are installed
2. Check your Python version
3. Verify CUDA installation (if using GPU)
4. Ensure you have sufficient disk space

For detailed error messages and solutions, please refer to our `GitHub Issues <https://github.com/your-repo/issues>`_.

Additional Notes
--------------

- The Docker installation method is recommended for production environments
- For development purposes, the Python installation method (Method 3) is preferred
- Standard installation (Method 1) is best for quick testing and evaluation

Need Help?
---------

If you need assistance:

- Check our `Documentation <https://your-docs-url.com>`_
- Open an issue on our `GitHub repository <https://github.com/your-repo>`_
- Contact our support team

.. note::
   Make sure to activate the conda environment before running any commands.

.. warning::
   GPU support requires appropriate NVIDIA drivers and CUDA toolkit installation.