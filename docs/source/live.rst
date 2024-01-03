LIVE
=====

.. _live:

.. hlist::
   :columns: 3

    * `[Project] <https://ma-xu.github.io/LIVE/>`_
    * `[Paper] <https://ma-xu.github.io/LIVE/index_files/CVPR22_LIVE_main.pdf>`_
    * `[Code] <https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization>`_

LIVE: Towards Layer-wise Image Vectorization
------------

Convert emojis from the original PNG format to vector format:

.. code-block:: console

   $ python svg_render.py x=live target='./data/simile.png'

To improve the SVGs, adjust the ``num_paths`` based on the raster image:

.. code-block:: console

   $ python svg_render.py x=live target='./path/to/file' x.num_paths=10

To accelerate the optimization process, you can add two paths at a time for optimization:

.. code-block:: console

   $ python svg_render.py x=live target='./path/to/file' x.num_paths=10 x.schedule_each=2