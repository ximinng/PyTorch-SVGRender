CLIPFont
==========

.. _clipfont:

`[Paper] <https://bmvc2022.mpi-inf.mpg.de/0543.pdf>`_ `[Code] <https://github.com/songyiren98/CLIPFont>`_

The CLIPFont algorithm was proposed in *CLIPFont: Texture Guided Vector WordArt Generation*.

The abstract from the paper is:

`Font design is a repetitive job that requires specialized skills. Unlike the existing fewshot font generation methods, this paper proposes a zero-shot font generation method
called CLIPFont for any language based on the CLIP model. The style of the font is controlled by the text description, and the skeleton of the font remains the same as the input
reference font. CLIPFont optimizes the parameters of vector fonts by gradient descent
and achieves artistic font generation by minimizing the directional distance between text
description and font in the CLIP embedding space. CLIP recognition loss is proposed
to keep the category of each character unchanged. The gradients computed on the rasterized images are returned to the vector parameter space utilizing a differentiable vector
renderer. Experimental results and Turing tests demonstrate our method’s state-of-the-art
performance. Project page: https://github.com/songyiren98/CLIPFont`

**Example 1: alphabet**

CLIPFont styles vector fonts according to text prompts.

Style the *alphabet* in the style of *Starry Night by Vincent van Gogh*:

.. code-block:: console

   $ python svg_render.py x=clipfont prompt='Starry Night by Vincent van gogh' target='./data/alphabet1.svg'

**Result**:

.. list-table:: Fig 1. text prompt: "Starry Night by Vincent van gogh"

    * - .. figure:: ../../data/alphabet1.svg
           :width: 250

           Input Vector Glyph

      - .. figure:: ../../examples/clipfont/alphabet1_VanGogh.svg
           :width: 250

           Final Vector Glyph


**Example 2: Chinese**

.. code-block:: console

   $ python svg_render.py x=clipfont prompt='Starry Night by Vincent van gogh' target='./data/ch1.svg'

**Result**:

.. list-table:: Fig 2. text prompt: "Starry Night by Vincent van gogh"

    * - .. figure:: ../../data/ch1.svg
           :width: 250

           Input Vector Glyph

      - .. figure:: ../../examples/clipfont/ch1_VanGogh.svg
           :width: 250

           Final Vector Glyph