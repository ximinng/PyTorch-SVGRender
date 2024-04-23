DiffVG
=====

.. _diffvg:

`[Project] <https://people.csail.mit.edu/tzumao/diffvg/>`_ `[Paper] <https://people.csail.mit.edu/tzumao/diffvg/diffvg.pdf>`_ `[Code] <https://github.com/BachiLi/diffvg>`_

The DiffVG algorithm was proposed in *Differentiable Vector Graphics Rasterization for Editing and Learning*.

DiffVG is a differentiable rasterizer for 2D vector graphics.

The abstract from the paper is:

`We introduce a differentiable rasterizer that bridges the vector graphics and raster image domains, enabling powerful raster-based loss functions, optimization procedures, and machine learning techniques to edit and generate vector content. We observe that vector graphics rasterization is differentiable after pixel prefiltering. Our differentiable rasterizer offers two prefiltering options: an analytical prefiltering technique and a multisampling anti-aliasing technique. The analytical variant is faster but can suffer from artifacts such as conflation. The multisampling variant is still efficient, and can render high-quality images while computing unbiased gradients for each pixel with respect to curve parameters. We demonstrate that our rasterizer enables new applications, including a vector graphics editor guided by image metrics, a painterly rendering algorithm that fits vector primitives to an image by minimizing a deep perceptual loss function, new vector graphics editing algorithms that exploit well-known image processing methods such as seam carving, and deep generative models that generate vector content from raster-only supervision under a VAE or GAN training objective.`

Additional Key Words and Phrases: **vector graphics, differentiable rendering, image vectorization.**

**Example:**

Convert a raster image from the original PNG format to vector format:

.. code-block:: console

   $ python svg_render.py x=diffvg target='./data/fallingwater.png'

**Result**:

.. list-table:: Fig 1. Rendering Result, 512 paths

    * - .. figure:: ../../data/fallingwater.png

           input raster image

      - .. figure:: ../../examples/diffvg/fallingwater_512paths.svg

           vectorization result, 512 paths


.. list-table:: Fig 2. Rendering Result, 1024 paths

    * - .. figure:: ../../data/fallingwater.png

           input raster image

      - .. figure:: ../../examples/diffvg/fallingwater_1024paths.svg

           vectorization result, 1024 paths