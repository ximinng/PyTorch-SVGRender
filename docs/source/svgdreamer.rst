SVGDreamer
===============

.. _svgdreamer:

`[Project] <https://ximinng.github.io/SVGDreamer-project/>`_ `[Paper] <https://arxiv.org/abs/2312.16476>`_ `[Code] <https://github.com/ximinng/SVGDreamer>`_

The SVGDreamer algorithm was proposed in *SVGDreamer: Text Guided SVG Generation with Diffusion Model*.

The abstract from the paper is:

`Recently, text-guided scalable vector graphics (SVGs) synthesis has shown promise in domains such as iconography and sketch. However, existing text-to-SVG generation methods lack editability and struggle with visual quality and result diversity. To address these limitations, we propose a novel text-guided vector graphics synthesis method called SVGDreamer. SVGDreamer incorporates a semantic-driven image vectorization (SIVE) process that enables the decomposition of synthesis into foreground objects and background, thereby enhancing editability. Specifically, the SIVE process introduce attention-based primitive control and an attention-mask loss function for effective control and manipulation of individual elements. Additionally, we propose a Vectorized Particle-based Score Distillation (VPSD) approach to tackle the challenges of color over-saturation, vector primitives over-smoothing, and limited result diversity in existing text-to-SVG generation methods. Furthermore, on the basis of VPSD, we introduce Reward Feedback Learning (ReFL) to accelerate VPSD convergence and improve aesthetic appeal. Extensive experiments have been conducted to validate the effectiveness of SVGDreamer, demonstrating its superiority over baseline methods in terms of editability, visual quality, and diversity.`
