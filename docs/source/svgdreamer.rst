SVGDreamer
===============

.. _svgdreamer:

`[Project] <https://ximinng.github.io/SVGDreamer-project/>`_ `[Paper] <https://arxiv.org/abs/2312.16476>`_ `[Code] <https://github.com/ximinng/SVGDreamer>`_

The SVGDreamer algorithm was proposed in *SVGDreamer: Text Guided SVG Generation with Diffusion Model*.

The abstract from the paper is:

`Recently, text-guided scalable vector graphics (SVGs) synthesis has shown promise in domains such as iconography and sketch. However, existing text-to-SVG generation methods lack editability and struggle with visual quality and result diversity. To address these limitations, we propose a novel text-guided vector graphics synthesis method called SVGDreamer. SVGDreamer incorporates a semantic-driven image vectorization (SIVE) process that enables the decomposition of synthesis into foreground objects and background, thereby enhancing editability. Specifically, the SIVE process introduce attention-based primitive control and an attention-mask loss function for effective control and manipulation of individual elements. Additionally, we propose a Vectorized Particle-based Score Distillation (VPSD) approach to tackle the challenges of color over-saturation, vector primitives over-smoothing, and limited result diversity in existing text-to-SVG generation methods. Furthermore, on the basis of VPSD, we introduce Reward Feedback Learning (ReFL) to accelerate VPSD convergence and improve aesthetic appeal. Extensive experiments have been conducted to validate the effectiveness of SVGDreamer, demonstrating its superiority over baseline methods in terms of editability, visual quality, and diversity.`

Examples of VPSD
^^^^^^^^^^^

SVGDreamer generates various styles of SVG based on text prompts. It supports the use of six vector primitives, including Iconography, Sketch, Pixel Art, Low-Poly, Painting, and Ink and Wash.

**Note: The examples provided here are based on VPSD only.**

Iconography
""""""""""""

Synthesize the SVGs of the Sydney Opera House in the style of Van Gogh's oil paintings,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='Sydney Opera House. oil painting. by Van Gogh' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.num_paths=512 result_path='./svgdreamer/SydneyOperaHouse'

You will get the following result:

.. raw:: html

    <div align="center">
    <img src="_images/icon_sydney_opera_house_1.png" alt="Pytorch-SVGRender">
    <p><strong>Fig. </strong>The oil paintings of Sydney Opera House by Van Gogh's. iconography. Number of vector particles: 6</p>
    </div>

.. raw:: html

    <div align="center">
    <img src="../examples/svgdreamer/icon_sydney_opera_house_2.png" alt="Pytorch-SVGRender">
    <p><strong>Fig. </strong>The oil paintings of Sydney Opera House by Van Gogh's. iconography. Number of vector particles: 6</p>
    </div>

------------

Synthesize a German shepherd in vector art,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A colorful German shepherd in vector art. tending on artstation.' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/GermanShepherd'

You will get the following result:

.. raw:: html

    <div align="center">
    <img src="../../examples/svgdreamer/icon_GermanShepherd_1.png" alt="German shepherd in vector art, iconography">
    <p><strong>Fig. </strong>German shepherd in vector art. iconography. Number of vector particles: 6</p>
    </div>

Sketch
""""""""""""

Pixel Art
""""""""""""

Synthesize German shepherds in vector art,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='Darth vader with lightsaber. ultrarealistic.' x.style='pixelart' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/DarthVader'

You will get the following result:

.. raw:: html

    <div align="center">
    <img src="../../../examples/svgdreamer/icon_GermanShepherd_1.png" alt="Darth vader, pixel art">
    <p><strong>Fig. </strong>Darth vader. pixel art. Number of vector particles: 6</p>
    </div>

Low-Poly
""""""""""""

Synthesize bald eagles in low-poly,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A picture of a bald eagle. low-ploy. polygon' x.style='low-poly' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/eagle'

You will get the following result:

.. raw:: html

    <div align="center">
    <img src="../../examples/svgdreamer/lowpoly_eagle_1.png" alt="bald eagle, low-poly">
    <p><strong>Fig. </strong>Bald eagle. low-poly. Number of vector particles: 6</p>
    </div>

Painting
""""""""""""

Ink and Wash
""""""""""""
