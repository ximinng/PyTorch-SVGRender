CLIPDraw
==========

.. _clipdraw:

`[Paper] <https://arxiv.org/abs/2106.14843>`_ `[Code] <https://github.com/kvfrans/clipdraw>`_

The CLIPDraw algorithm was proposed in *CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders*.

The abstract from the paper is:

`CLIPDraw is an algorithm that synthesizes novel drawings from natural language input. It does not require any additional training; rather, a pre-trained CLIP language-image encoder is used as a metric for maximizing similarity between the given description and a generated drawing. Crucially, CLIPDraw operates over vector strokes rather than pixel images, which biases drawings towards simpler human-recognizable shapes. Results compare CLIPDraw with other synthesisthrough-optimization methods, as well as highlight various interesting behaviors of CLIPDraw, such as satisfying ambiguous text in multiple ways, reliably producing drawings in diverse styles, and scaling from simple to complex visual representations as stroke count increases.`

**Examples:**

CLIPDraw synthesizes SVGs based on text prompts.

Synthesize a photo of a cat:

.. code-block:: console

   $ python svg_render.py x=clipdraw prompt='a photo of a cat'

You will get the following result:

.. image:: ../../examples/clipdraw/clipdraw_cat.svg