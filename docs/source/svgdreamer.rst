SVGDreamer
===============

.. _svgdreamer:

`[Project] <https://ximinng.github.io/SVGDreamer-project/>`_ `[Paper] <https://arxiv.org/abs/2312.16476>`_ `[Code] <https://github.com/ximinng/SVGDreamer>`_

The SVGDreamer algorithm was proposed in *SVGDreamer: Text Guided SVG Generation with Diffusion Model*.

The abstract from the paper is:

`Recently, text-guided scalable vector graphics (SVGs) synthesis has shown promise in domains such as iconography and sketch. However, existing text-to-SVG generation methods lack editability and struggle with visual quality and result diversity. To address these limitations, we propose a novel text-guided vector graphics synthesis method called SVGDreamer. SVGDreamer incorporates a semantic-driven image vectorization (SIVE) process that enables the decomposition of synthesis into foreground objects and background, thereby enhancing editability. Specifically, the SIVE process introduce attention-based primitive control and an attention-mask loss function for effective control and manipulation of individual elements. Additionally, we propose a Vectorized Particle-based Score Distillation (VPSD) approach to tackle the challenges of color over-saturation, vector primitives over-smoothing, and limited result diversity in existing text-to-SVG generation methods. Furthermore, on the basis of VPSD, we introduce Reward Feedback Learning (ReFL) to accelerate VPSD convergence and improve aesthetic appeal. Extensive experiments have been conducted to validate the effectiveness of SVGDreamer, demonstrating its superiority over baseline methods in terms of editability, visual quality, and diversity.`

Examples of VPSD
^^^^^^^^^^^

SVGDreamer generates various styles of SVG based on text prompts. It supports the use of six vector primitives, including **Iconography, Sketch, Pixel Art, Low-Poly, Painting, and Ink and Wash**.

**Note: The examples provided here are based on VPSD only.**

Iconography
""""""""""""

Synthesize a German shepherd in vector art,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A colorful German shepherd in vector art. tending on artstation.' save_step=50 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 result_path='./svgdreamer/GermanShepherd'

**Result**:

.. list-table:: Fig 1. German shepherd in vector art. iconography. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd/p_3.svg
           :width: 150

           particle 4

To save GPU VRAM, fp16 optimization is supported via `state.mprec='fp16'`,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A colorful German shepherd in vector art. tending on artstation.' save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/GermanShepherd-fp32'

If you have a VRAM limit issue? ``fp16`` can be used:

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A colorful German shepherd in vector art. tending on artstation.' state.mprec='fp16' save_step=50 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 result_path='./svgdreamer/GermanShepherd-fp16'

**Result**:

.. list-table:: Fig 2. FP16 optimization. iconography. Number of vector particles: 6

    * - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd-fp16/p_0.svg
           :width: 200

           particle 1

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd-fp16/p_1.svg
           :width: 200

           particle 2

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd-fp16/p_2.svg
           :width: 200

           particle 3

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd-fp16/p_3.svg
           :width: 200

           particle 4

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd-fp16/p_4.svg
           :width: 200

           particle 5

      - .. figure:: ../../examples/svgdreamer/Iconography-GermanShepherd-fp16/p_5.svg
           :width: 200

           particle 6

------------

Synthesize the SVGs of the Sydney Opera House in the style of Van Gogh's oil paintings,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='Sydney Opera House. oil painting. by Van Gogh' save_step=50 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 x.num_paths=512 result_path='./svgdreamer/SydneyOperaHouse'

**Result**:

.. list-table:: Fig 3. The oil paintings of Sydney Opera House by Van Gogh's. iconography. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_3.svg
           :width: 150

           particle 4


Sketch
""""""""""""

Synthesize the free-hand sketches of the Lamborghini,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A free-hand drawing of A speeding Lamborghini. black and white drawing.' x.style='sketch' save_step=30 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 x.num_paths=128 result_path='./svgdreamer/Lamborghini'

**Result**:

.. list-table:: Fig 4. A free-hand drawing of A speeding Lamborghini. Sketch. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/Iconography-SydneyOperaHouse/p_3.svg
           :width: 150

           particle 4

Pixel Art
""""""""""""

The DarthVader with lightsaber in pixel art,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='Darth vader with lightsaber. ultrarealistic.' x.style='pixelart' x.grid=30 save_step=50 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 x.guidance.num_iter=1000 result_path='./svgdreamer/DarthVader' seed=302819

**Result**:

.. list-table:: Fig 5. Darth vader. pixel art. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/PixelArt-DarthVader/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/PixelArt-DarthVader/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/PixelArt-DarthVader/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/PixelArt-DarthVader/p_3.svg
           :width: 150

           particle 4

Low-Poly
""""""""""""

Synthesize bald eagles in low-poly,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A picture of a bald eagle. low-ploy. polygon' x.style='low-poly' save_step=50 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 result_path='./svgdreamer/Eagle'

**Result**:

.. list-table:: Fig 6. Bald eagle. low-poly. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/LowPoly-BaldEagles/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/LowPoly-BaldEagles/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/LowPoly-BaldEagles/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/LowPoly-BaldEagles/p_3.svg
           :width: 150

           particle 4

------------

Synthesize scarlet macaws in low-poly,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='A picture of a scarlet macaw. low-ploy. polygon' x.style='low-poly' save_step=50 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 result_path='./svgdreamer/ScarletMacaw'

**Result**:

.. list-table:: Fig 7. Scarlet Macaw. low-poly. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/LowPoly-Macaw/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/LowPoly-Macaw/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/LowPoly-Macaw/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/LowPoly-Macaw/p_3.svg
           :width: 150

           particle 4

Painting
""""""""""""

Synthesize phoenixes coming out of the fire drawing,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='a phoenix coming out of the fire drawing. lineal color. trending on artstation.' x.style='painting' save_step=50 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 x.num_paths=384 result_path='./svgdreamer/phoenix'

**Result**:

.. list-table:: Fig 8. Phoenixes. Painting. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/Painting-Phoenix/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/Painting-Phoenix/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/Painting-Phoenix/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/Painting-Phoenix/p_3.svg
           :width: 150

           particle 4


Ink and Wash
""""""""""""

Synthesize the Big Wild Goose Pagoda,

.. code-block:: console

   $ python svg_render.py x=svgdreamer prompt='Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor.' x.style='ink' save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2 x.guidance.t_schedule='max_0.5_2000' x.num_paths=128 x.width=6 result_path='./svgdreamer/BigWildGoosePagoda'

**Result**:

.. list-table:: Fig 10. Big Wild Goose Pagoda. Ink and Wash. Number of vector particles: 4

    * - .. figure:: ../../examples/svgdreamer/Ink-BigWildGoosePagoda/p_0.svg
           :width: 150

           particle 1

      - .. figure:: ../../examples/svgdreamer/Ink-BigWildGoosePagoda/p_1.svg
           :width: 150

           particle 2

      - .. figure:: ../../examples/svgdreamer/Ink-BigWildGoosePagoda/p_2.svg
           :width: 150

           particle 3

      - .. figure:: ../../examples/svgdreamer/Ink-BigWildGoosePagoda/p_3.svg
           :width: 150

           particle 4