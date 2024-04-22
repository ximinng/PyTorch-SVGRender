LIVE
=====

.. _live:

`[Project] <https://ma-xu.github.io/LIVE/>`_ `[Paper] <https://ma-xu.github.io/LIVE/index_files/CVPR22_LIVE_main.pdf>`_ `[Code] <https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization>`_

The LIVE algorithm was proposed in *Towards Layer-wise Image Vectorization*.

The abstract from the paper is:

`Image rasterization is a mature technique in computer graphics, while image vectorization, the reverse path of rasterization, remains a major challenge. Recent advanced deep learning-based models achieve vectorization and semantic interpolation of vector graphs and demonstrate a better topology of generating new figures. However, deep models cannot be easily generalized to out-ofdomain testing data. The generated SVGs also contain complex and redundant shapes that are not quite convenient for further editing. Specifically, the crucial layerwise topology and fundamental semantics in images are still not well understood and thus not fully explored. In this work, we propose Layer-wise Image Vectorization, namely LIVE, to convert raster images to SVGs and simultaneously maintain its image topology. LIVE can generate compact SVG forms with layer-wise structures that are semantically consistent with human perspective. We progressively add new bézier paths and optimize these paths with the layer-wise framework, newly designed loss functions, and component-wise path initialization technique. Our experiments demonstrate that LIVE presents more plausible vectorized forms than prior works and can be generalized to new images.  With the help of this newly learned topology, LIVE initiates human editable SVGs for both designers and other downstream applications.`

**Examples: Smile Emoji**

LIVE to convert raster images to SVGs and simultaneously maintain its image topology.

Convert the *emoji* from the original PNG format to vector format:

.. code-block:: console

   $ python svg_render.py x=live target='./data/simile.png'

**Rendering**:

.. list-table:: Fig 1. Rendering process

    * - .. figure:: ../../examples/live/svg_iter0.svg
           :width: 150

           0 iter, add 1 path, total path: 1

      - .. figure:: ../../examples/live/svg_iter500.svg
           :width: 150

           500 iter, add 1 path, total path: 2

      - .. figure:: ../../examples/live/svg_iter1000.svg
           :width: 150

           1000 iter, add 1 path, total path: 3

      - .. figure:: ../../examples/live/svg_iter1500.svg
           :width: 150

           1500 iter, add 1 path, total path: 4

      - .. figure:: ../../examples/live/svg_iter2000.svg
           :width: 150

           2000 iter, add 1 path, total path: 5

      - .. figure:: ../../examples/live/live_smile.svg
           :width: 150

           2500 iter, add 1 path, total path: 5

**Final Result**:

.. figure:: ../../examples/live/live_smile.svg
   :align: center

   Fig 2. image vectorization.

**Your Own Case**

To improve the SVGs, adjust the ``num_paths`` based on the raster image:

.. code-block:: console

   $ python svg_render.py x=live target='./path/to/file' x.num_paths=10

To accelerate the optimization process, you can add two paths at a time for optimization:

.. code-block:: console

   $ python svg_render.py x=live target='./path/to/file' x.num_paths=10 x.schedule_each=2