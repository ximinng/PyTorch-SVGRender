CLIPascene
==========

.. _clipascene:

`[Project] <https://clipascene.github.io/CLIPascene/>`_ `[Paper] <https://arxiv.org/abs/2211.17256>`_ `[Code] <https://github.com/yael-vinker/SceneSketch>`_

The CLIPascene algorithm was proposed in *CLIPascene: Scene Sketching with Different Types and Levels of Abstraction*.

The abstract from the paper is:

`In this paper, we present a method for converting a given scene image into a sketch using different types and multiple levels of abstraction. We distinguish between two types of abstraction. The first considers the fidelity of the sketch, varying its representation from a more precise portrayal of the input to a looser depiction. The second is defined by the visual simplicity of the sketch, moving from a detailed depiction to a sparse sketch. Using an explicit disentanglement into two abstraction axes — and multiple levels for each one — provides users additional control over selecting the desired sketch based on their personal goals and preferences. To form a sketch at a given level of fidelity and simplification, we train two MLP networks. The first network learns the desired placement of strokes, while the second network learns to gradually remove strokes from the sketch without harming its recognizability and semantics. Our approach is able to generate sketches of complex scenes including those with complex backgrounds (e.g. natural and urban settings) and subjects (e.g. animals and people) while depicting gradual abstractions of the input scene in terms of fidelity and simplicity.`

**Examples: ballerina**

CLIPascene converts an image of scene image into a sketch using different types and multiple levels of abstraction.

.. note::

   first download the `U2Net <https://huggingface.co/xingxm/PyTorch-SVGRender-models/resolve/main/u2net.zip>`_ model, and put the model in :file:`./checkpoint/u2net/u2net.pth`.

Convert an image of *ballerina* from the original PNG format to an abstract sketch:

.. code-block:: console

   $ python svg_render.py x=clipascene target='./data/ballerina.png'

**Segmentation and Inpainting Result**:

.. list-table:: Fig 1. Input Process Result

    * - .. figure:: ../../data/ballerina.png
           :width: 200

           Input image

      - .. figure:: ../../examples/clipascene/ballerina/fg_img.png
           :width: 200

           Foreground image

      - .. figure:: ../../examples/clipascene/ballerina/bg_img.png
           :width: 200

           Background image

**Vector Sketch Result**:

.. list-table:: Fig 2. Rendering Result

    * - .. figure:: ../../examples/clipascene/ballerina/combined.svg
           :width: 200

           Final sketch

      - .. figure:: ../../examples/clipascene/ballerina/fg.svg
           :width: 200

           Foreground sketch

      - .. figure:: ../../examples/clipascene/ballerina/bg.svg
           :width: 200

           Background sketch


**Examples: Bull**

Convert an image of *bull* from the original PNG format to an abstract sketch:

.. code-block:: console

   $ python svg_render.py x=clipascene target='./data/bull.png'

**Segmentation and Inpainting Result**:

.. list-table:: Fig 1. Input Process Result

    * - .. figure:: ../../data/bull.png
           :width: 200

           Input image

      - .. figure:: ../../examples/clipascene/bull/fg_img.png
           :width: 200

           Foreground image

      - .. figure:: ../../examples/clipascene/bull/bg_img.png
           :width: 200

           Background image

**Vector Sketch Result**:

.. list-table:: Fig 2. Rendering Result

    * - .. figure:: ../../examples/clipascene/bull/combined.svg
           :width: 200

           Final sketch

      - .. figure:: ../../examples/clipascene/bull/fg.svg
           :width: 200

           Foreground sketch

      - .. figure:: ../../examples/clipascene/bull/bg.svg
           :width: 200

           Background sketch