#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from collections import namedtuple
import sys


def _print(args, text, **kwargs):
    if not args.quiet:
        print(text, **kwargs)


class Paper(namedtuple("Paper", [
        "title",
        "url",
        "authors",
        "links"
    ])):
    pass


class Conference(namedtuple("Conference", ["name"])):
    pass


class Link(namedtuple("Link", ["name", "url", "html", "text"])):
    pass


def author_list(authors):
    return authors.split(",")


publications = [
    Paper(
        "InNeRF360: Text-Guided 3D-Consistent Object Inpainting on 360-degree Neural Radiance Fields",
        "https://arxiv.org/abs/2305.15094",
        author_list("Wang, Dongqing; Zhang, Tong; Abboud, Alaa; Süsstrunk, Sabine"),
        [ Link("Abstract", None, "We propose InNeRF360, an automatic system that accurately removes text-specified objects from 360-degree Neural Radiance Fields (NeRF). The challenge is to effectively remove objects while inpainting perceptually consistent content for the missing regions, which is particularly demanding for existing NeRF models due to their implicit volumetric representation. Moreover, unbounded scenes are more prone to floater artifacts in the inpainted region than frontal-facing scenes, as the change of object appearance and background across views is more sensitive to inaccurate segmentations and inconsistent inpainting. With a trained NeRF and a text description, our method efficiently removes specified objects and inpaints visually consistent content without artifacts. We apply depth-space warping to enforce consistency across multiview text-encoded segmentations, and then refine the inpainted NeRF model using perceptual priors and 3D diffusion-based geometric priors to ensure visual plausibility. Through extensive experiments in segmentation and inpainting on 360-degree and frontal-facing NeRFs, we show that our approach is effective and enhances NeRF's editability.", None),
            Link("Paper", "https://arxiv.org/abs/2305.15094", None, None),
            Link("Poster", "posters/poster_19.pdf", None, None)
        ]
    ),

    Paper(
        "Towards Practical Single-shot Motion Synthesis",
        "https://arxiv.org/pdf/2406.01136",
        author_list("Roditakis, Konstantinos; Thermos, Spyridon; Zioulis Nikolaos"),
        [ Link("Abstract", None, "Despite the recent advances in the so-called 'cold start' generation from text prompts, their needs in data and computing resources, as well as the ambiguities around intellectual property and privacy concerns pose certain counterarguments for their utility. An interesting and relatively unexplored alternative has been the introduction of unconditional synthesis from a single sample, which has led to interesting generative applications. In this paper we focus on single-shot motion generation and more specifically on accelerating the training time of a Generative Adversarial Network (GAN). In particular, we tackle the challenge of GAN's equilibrium collapse when using mini-batch training by carefully annealing the weights of the loss functions that prevent mode collapse. Additionally, we perform statistical analysis in the generator and discriminator models to identify correlations between training stages and enable transfer learning. Our improved GAN achieves competitive quality and diversity on the Mixamo benchmark when compared to the original GAN architecture and a single-shot diffusion model, while being up to x6.8 faster in training time from the former and x1.75 from the latter. Finally, we demonstrate the ability of our improved GAN to mix and compose motion with a single forward pass.", None),
            Link("Paper", "papers/paper_15.pdf", None, None),
            Link("Poster", "posters/poster_15.pdf", None, None)
        ]
    ),

    Paper(
        "DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing",
        "https://showlab.github.io/DynVideo-E/",
        author_list("Liu, Jia-Wei; Cao, Yan-Pei; Wu, Jay Zhangjie; Mao, Weijia; Gu, Yuchao; Zhao, Rui; Keppo, Jussi; Shan, Ying; Shou, Mike Zheng"),
        [ Link("Abstract", None, "Despite recent progress in diffusion-based video editing, existing methods are limited to short-length videos due to the contradiction between long-range consistency and frame-wise editing. Prior attempts to address this challenge by introducing video-2D representations encounter significant difficulties with large-scale motion- and view-change videos, especially in human-centric scenarios. To overcome this, we propose to introduce the dynamic Neural Radiance Fields (NeRF) as the innovative video representation, where the editing can be performed in the 3D spaces and propagated to the entire video via the deformation field. To provide consistent and controllable editing, we propose the image-based video-NeRF editing pipeline with a set of innovative designs, including multi-view multi-pose Score Distillation Sampling (SDS) from both the 2D personalized diffusion prior and 3D diffusion prior, reconstruction losses, text-guided local parts super-resolution, and style transfer. Extensive experiments demonstrate that our method, dubbed as DynVideo-E, significantly outperforms SOTA approaches on two challenging datasets by a large margin of 50% ~ 95% for human preference.", None),
            Link("Paper", "https://arxiv.org/abs/2310.10624", None, None),
        ]
    ),

    Paper(
        "Intrinsic LoRA: A Generalist Approach for Discovering Knowledge in Generative Models",
        "papers/paper_6.pdf",
        author_list("Du, Xiaodan; Kolkin, Nicholas I; Shakhnarovich, Greg; Bhattad, Anand"),
        [ Link("Abstract", None, "Generative models have been shown to be capable of creating images that closely mimic real scenes, suggesting they inherently encode scene representations. We introduce INTRINSIC LORA (I-LORA), a general approach that uses Low-Rank Adaptation (LoRA) to discover scene intrinsics such as normals, depth, albedo, and shading from a wide array of generative models. I-LORA is lightweight, adding minimally to the model’s parameters and requiring very small datasets for this knowledge discovery. Our approach, applicable to Diffusion models, GANs, and Autoregressive models alike, generates intrinsics using the same output head as the original images.", None),
            Link("Paper", "papers/paper_6.pdf", None, None),
            Link("Poster", "posters/poster_6.pdf", None, None)
        ]
    ),

    Paper(
        "3D Shape Augmentation with Content-Aware Shape Resizing",
        "https://arxiv.org/abs/2405.09050v1",
        author_list("Chen, Mingxiang; Zhou, Boli; ZHANG, Jian"),
        [ Link("Abstract", None, "Recent advancements in deep learning for 3D models have propelled breakthroughs in generation, detection, and scene understanding. However, the effectiveness of these algorithms hinges on large training datasets. We address the challenge by introducing Efficient 3D Seam Carving (E3SC), a novel 3D model augmentation method based on seam carving, which progressively deforms only part of the input model while ensuring the overall semantics are unchanged. Experiments show that our approach is capable of producing diverse and high-quality augmented 3D shapes across various types and styles of input models, achieving considerable improvements over previous methods. Quantitative evaluations demonstrate that our method effectively enhances the novelty and quality of shapes generated by other subsequent 3D generation algorithms.", None),
            Link("Paper", "https://arxiv.org/abs/2405.09050v1", None, None),
        ]
    ),

    Paper(
        "Seamless Human Motion Composition with Blended Positional Encodings",
        "https://arxiv.org/abs/2402.15509",
        author_list("Barquero, German; Escalera, Sergio; Palmero, Cristina"),
        [ Link("Abstract", None, "Conditional human motion generation is an important topic with many applications in virtual reality, gaming, and robotics. While prior works have focused on generating motion guided by text, music, or scenes, these typically result in isolated motions confined to short durations. Instead, we address the generation of long, continuous sequences guided by a series of varying textual descriptions. In this context, we introduce FlowMDM, the first diffusion-based model that generates seamless Human Motion Compositions (HMC) without any postprocessing or redundant denoising steps. For this, we introduce the Blended Positional Encodings, a technique that leverages both absolute and relative positional encodings in the denoising chain. More specifically, global motion coherence is recovered at the absolute stage, whereas smooth and realistic transitions are built at the relative stage. As a result, we achieve state-of-the-art results in terms of accuracy, realism, and smoothness on the Babel and HumanML3D datasets. FlowMDM excels when trained with only a single description per motion sequence thanks to its Pose-Centric Cross-ATtention, which makes it robust against varying text descriptions at inference time. Finally, to address the limitations of existing HMC metrics, we propose two new metrics: the Peak Jerk and the Area Under the Jerk, to detect abrupt transitions.", None),
            Link("Paper", "https://arxiv.org/abs/2402.15509", None, None),
        ]
    ),

    Paper(
        "C3DAG: Controlled 3D Animal Generation using 3D pose guidance",
        "https://arxiv.org/abs/2406.07742",
        author_list("Mishra, Sandeep; Saha, Oindrila ; Bovik, Alan"),
        [ Link("Abstract", None, "Recent advancements in text-to-3D generation have demonstrated the ability to generate high quality 3D assets. However while generating animals these methods underperform, often portraying inaccurate anatomy and geometry. Towards ameliorating this defect, we present C3DAG, a novel pose-Controlled text-to-3D Animal Generation framework which generates a high quality 3D animal consistent with a given pose. We also introduce an automatic 3D shape creator tool, that allows dynamic pose generation and modification via a web-based tool, and that generates a 3D balloon animal using simple geometries. A NeRF is then initialized using this 3D shape using depth-controlled SDS. In the next stage, the pre-trained NeRF is fine-tuned using quadruped-pose-controlled SDS. The pipeline that we have developed not only produces geometrically and anatomically consistent results, but also renders highly controlled 3D animals, unlike prior methods which do not allow fine-grained pose control.", None),
            Link("Paper", "https://arxiv.org/abs/2406.07742", None, None),
            Link("Poster", "posters/poster_27.pdf", None, None)
        ]
    ),

    Paper(
        "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting",
        "https://arxiv.org/abs/2312.00451",
        author_list("Zhu, Zehao; Fan, Zhiwen; Jiang, Yifan; Cong, Wenyan; Wang, Zhangyang"),
        [ Link("Abstract", None, "Novel view synthesis from limited observations remains an important and persistent task. However, high efficiency in existing NeRF-based few-shot view synthesis is often compromised to obtain an accurate 3D representation. To address this challenge, we propose a few-shot view synthesis framework based on 3D Gaussian Splatting that enables real-time and photo-realistic view synthesis with as few as three training views. The proposed method, dubbed FSGS, handles the extremely sparse initialized SfM points with a thoughtfully designed Gaussian Unpooling process. Our method iteratively distributes new Gaussians around the most representative locations, subsequently infilling local details in vacant areas. We also integrate a large-scale pre-trained monocular depth estimator within the Gaussians optimization process, leveraging online augmented views to guide the geometric optimization towards an optimal solution. Starting from sparse points observed from limited input viewpoints, our FSGS can accurately grow into unseen regions, comprehensively covering the scene and boosting the rendering quality of novel views. Overall, FSGS achieves state-of-the-art performance in both accuracy and rendering efficiency across diverse datasets, including LLFF, Mip-NeRF360, and Blender.", None),
            Link("Paper", "https://arxiv.org/abs/2312.00451", None, None),
            Link("Poster", "posters/poster_41.pdf", None, None)
        ]
    ),

    Paper(
        "4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling",
        "https://sherwinbahmani.github.io/4dfy/",
        author_list("Bahmani, Sherwin; Skorokhodov, Ivan; Rong, Victor; Wetzstein, Gordon; Guibas, Leonidas; Wonka, Peter; Tulyakov, Sergey; Park, Jeong Joon; Tagliasacchi, Andrea; Lindell, David B"),
        [ Link("Abstract", None, "Recent breakthroughs in text-to-4D generation rely on pre-trained text-to-image and text-to-video models to generate dynamic 3D scenes. However, current text-to-4D methods face a three-way tradeoff between the quality of scene appearance, 3D structure, and motion. For example, text-to-image models and their 3D-aware variants are trained on internet-scale image datasets and can be used to produce scenes with realistic appearance and 3D structure -- but no motion. Text-to-video models are trained on relatively smaller video datasets and can produce scenes with motion, but poorer appearance and 3D structure. While these models have complementary strengths, they also have opposing weaknesses, making it difficult to combine them in a way that alleviates this three-way tradeoff. Here, we introduce hybrid score distillation sampling, an alternating optimization procedure that blends supervision signals from multiple pre-trained diffusion models and incorporates benefits of each for high-fidelity text-to-4D generation. Using hybrid SDS, we demonstrate synthesis of 4D scenes with compelling appearance, 3D structure, and motion.", None),
            Link("Paper", "https://arxiv.org/abs/2311.17984", None, None),
            Link("Poster", "posters/poster_36.pdf", None, None)
        ]
    ),

    Paper(
        "PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI",
        "https://arxiv.org/abs/2404.09465",
        author_list("Yang, Yandan; Jia, Baoxiong; Zhi, Peiyuan; Huang, Siyuan"),
        [ Link("Abstract", None, "With recent developments in Embodied Artificial Intelligence (EAI) research, there has been a growing demand for high-quality, large-scale interactive scene generation. While prior methods in scene synthesis have prioritized the naturalness and realism of the generated scenes, the physical plausibility and interactivity of scenes have been largely left unexplored. To address this disparity, we introduce PhyScene, a novel method dedicated to generating interactive 3D scenes characterized by realistic layouts, articulated objects, and rich physical interactivity tailored for embodied agents. Based on a conditional diffusion model for capturing scene layouts, we devise novel physics- and interactivity-based guidance mechanisms that integrate constraints from object collision, room layout, and object reachability. Through extensive experiments, we demonstrate that PhyScene effectively leverages these guidance functions for physically interactable scene synthesis, outperforming existing state-of-the-art scene synthesis methods by a large margin. Our findings suggest that the scenes generated by PhyScene hold considerable potential for facilitating diverse skill acquisition among agents within interactive environments, thereby catalyzing further advancements in embodied AI research.", None),
            Link("Paper", "https://arxiv.org/abs/2404.09465", None, None),
            Link("Poster", "posters/poster_11.pdf", None, None)
        ]
    ),

    Paper(
        "Consistency^2: Consistent and Fast 3D Painting with Latent Consistency Models",
        "papers/paper_21.pdf",
        author_list("Wang, Tianfu; Obukhov, Anton; Schindler, Konrad"),
        [ Link("Abstract", None, "Generative 3D Painting is among the top productivity boosters in high-resolution 3D asset management and recy- cling. Ever since text-to-image models became accessible for inference on consumer hardware, the performance of 3D Painting methods has consistently improved and is cur- rently close to plateauing. At the core of most such models lies denoising diffusion in the latent space, an inherently time-consuming iterative process. Multiple techniques have been developed recently to accelerate generation and reduce sampling iterations by orders of magnitude. Designed for 2D generative imaging, these techniques do not come with recipes for lifting them into 3D. In this paper, we ad- dress this shortcoming by proposing a Latent Consistency Model (LCM) adaptation for the task at hand. We analyze the strengths and weaknesses of the proposed model and evaluate it quantitatively and qualitatively. Based on the Objaverse dataset samples study, our 3D painting method attains strong preference in all evaluations.", None),
            Link("Paper", "papers/paper_21.pdf", None, None),
            Link("Poster", "posters/poster_21.pdf", None, None)
        ]
    ),

    Paper(
        "ControlRoom3D: Room Generation using Semantic Proxy Rooms",
        "https://arxiv.org/pdf/2312.05208",
        author_list("Schult, Jonas; Tsai, Sam; Höllein, Lukas; Wu, Bichen; Wang, Jialiang; Ma, Chih-Yao; Li, Kunpeng Optional; Wang, Xiaofang; Wimbauer, Felix; He, Zijian; Zhang, Peizhao; Leibe, Bastian; Vajda, Peter; Hou, Ji"),
        [ Link("Abstract", None, "Manually creating 3D environments for AR/VR applications is a complex process requiring expert knowledge in 3D modeling software. Pioneering works facilitate this process by generating room meshes conditioned on textual style descriptions. Yet, many of these automatically generated 3D meshes do not adhere to typical room layouts, compromising their plausibility, e.g., by placing several beds in one bedroom. To address these challenges, we present ControlRoom3D, a novel method to generate high-quality room meshes. Central to our approach is a user-defined 3D semantic proxy room that outlines a rough room layout based on semantic bounding boxes and a textual description of the overall room style. Our key insight is that when rendered to 2D, this 3D representation provides valuable geometric and semantic information to control powerful 2D models to generate 3D consistent textures and geometry that aligns well with the proxy room. Backed up by an extensive study including quantitative metrics and qualitative user evaluations, our method generates diverse and globally plausible 3D room meshes, thus empowering users to design 3D rooms effortlessly without specialized knowledge.", None),
            Link("Paper", "https://arxiv.org/pdf/2312.05208", None, None),
            Link("Poster", "posters/poster_7.pdf", None, None)
        ]
    ),

    Paper(
        "Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering",
        "https://arxiv.org/abs/2312.11360",
        author_list("Youwang, Kim; Oh, Tae-Hyun; Pons-Moll, Gerard"),
        [ Link("Abstract", None, "We present Paint-it, a text-driven high-fidelity texture map synthesis method for 3D meshes via neural re-parameterized texture optimization. Paint-it synthesizes texture maps from a text description by synthesis-through-optimization, exploiting the Score-Distillation Sampling (SDS). We observe that directly applying SDS yields undesirable texture quality due to its noisy gradients. We reveal the importance of texture parameterization when using SDS. Specifically, we propose Deep Convolutional Physically-Based Rendering (DC-PBR) parameterization, which re-parameterizes the physically-based rendering (PBR) texture maps with randomly initialized convolution-based neural kernels, instead of a standard pixel-based parameterization. We show that DC-PBR inherently schedules the optimization curriculum according to texture frequency and naturally filters out the noisy signals from SDS. In experiments, Paint-it obtains remarkable quality PBR texture maps within 15 min., given only a text description. We demonstrate the generalizability and practicality of Paint-it by synthesizing high-quality texture maps for large-scale mesh datasets and showing test-time applications such as relighting and material control using a popular graphics engine.", None),
            Link("Paper", "https://arxiv.org/abs/2312.11360", None, None),
            Link("Poster", "https://kim-youwang.github.io/media/paint-it/cvpr24_poster_youwang_final.pdf", None, None),
        ]
    ),

    Paper(
        "EucliDreamer: Fast and High-Quality Texturing for 3D Models with Depth-Conditioned Stable Diffusion",
        "https://arxiv.org/abs/2404.10279",
        author_list("Le, Cindy; Hetang, Congrui; Lin, Chendi; Cao, Ang ; He, Yihui"),
        [ Link("Abstract", None, "We present EucliDreamer, a simple and effective method to generate textures for 3D models given text prompts and meshes. The texture is parametrized as an implicit function on the 3D surface, which is optimized with the Score Distillation Sampling (SDS) process and differentiable rendering. To generate high-quality textures, we leverage a depth-conditioned Stable Diffusion model guided by the depth image rendered from the mesh. We test our approach on 3D models in Objaverse and conducted a user study, which shows its superior quality compared to existing texturing methods like Text2Tex. In addition, our method converges 2 times faster than DreamFusion. Through text prompting, textures of diverse art styles can be produced. We hope Euclidreamer proides a viable solution to automate a labor-intensive stage in 3D content creation.", None),
            Link("Paper", "https://arxiv.org/abs/2404.10279", None, None),
            Link("Poster", "posters/poster_28.pdf", None, None)
        ]
    ),

    Paper(
        "Volumetric Style Transfer using Neural Cellular Automata",
        "",
        author_list("Wang, Dongqing; Pajouheshgar, Ehsan; Xu, Yitao; Zhang, Tong; Süsstrunk, Sabine"),
        [ Link("Abstract", None, "Artistic stylization of 3D volumetric smoke data is still a challenge in computer graphics due to the difficulty of ensuring spatiotemporal consistency given a reference style image, and that within reasonable time and computational resources. In this work, we introduce Volumetric Neural Cellular Automata (VNCA), a novel model for efficient volumetric style transfer that synthesizes, in real-time, multi-view consistent stylizing features on the target smoke with temporally coherent transitions between stylized simulation frames. VNCA synthesizes a 3D texture volume with color and density stylization and dynamically aligns this volume with the intricate motion patterns of the smoke simulation under the Eulerian framework. Our approach replaces the explicit fluid advection modeling and the inter-frame smoothing terms with the self-emerging motion of the underlying cellular automaton, thus reducing the training time by over an order of magnitude. Beyond smoke simulations, we demonstrate the versatility of our approach by showcasing its applicability to mesh stylization.", None),
            Link("Paper", "", None, None),
            Link("Poster", "posters/poster_20.pdf", None, None),
        ]
    ),

    Paper(
        "As-Plausible-As-Possible: Plausibility-Aware Mesh Deformation Using 2D Diffusion Priors",
        "https://as-plausible-as-possible.github.io/",
        author_list("Yoo, Seungwoo; Kim, Kunho; Kim, Vladimir; Sung, Minhyuk"),
        [ Link("Abstract", None, "We present As-Plausible-as-Possible (APAP) mesh deformation technique that leverages 2D diffusion priors to preserve the plausibility of a mesh under user-controlled deformation. Our framework uses per-face Jacobians to represent mesh deformations, where mesh vertex coordinates are computed via a differentiable Poisson Solve. The deformed mesh is rendered, and the resulting 2D image is used in the Score Distillation Sampling (SDS) process, which enables extracting meaningful plausibility priors from a pretrained 2D diffusion model. To better preserve the identity of the edited mesh, we fine-tune our 2D diffusion model with LoRA. Gradients extracted by SDS and a user-prescribed handle displacement are then backpropagated to the per-face Jacobians, and we use iterative gradient descent to compute the final deformation that balances between the user edit and the output plausibility. We evaluate our method with 2D and 3D meshes and demonstrate qualitative and quantitative improvements when using plausibility priors over geometry-preservation or distortion-minimization priors used by previous techniques", None),
            Link("Paper", "https://arxiv.org/abs/2311.16739", None, None),
        ]
    ),

    Paper(
        "An Ethical Framework for Trustworthy Neural Rendering applied in Cultural Heritage and Creative Industries",
        "papers/paper_43.pdf",
        author_list("Stacchio, Lorenzo; Balloni, Emanuele; Gorgoglione, Lucrezia; Pierdicca, Roberto; Mancini, Adriano; Frontoni, Emanuele; Giovanola, Benedetta; Tiribelli, Simona; Paolanti, Marina; Zingaretti, Primo"),
        [ Link("Abstract", None, "Artificial Intelligence (AI) has revolutionized various sectors, including Cultural Heritage (CH) and Creative Industries (CI), defining novel opportunities and challenges in preserving tangible and intangible human productions. In such a context, Neural Rendering (NR) paradigms play the pivotal role of 3D reconstructing objects or scenes by optimizing images depicting them. However, there is a lack of work examining the ethical concerns associated with its usage. Those are particularly relevant in scenarios where NR is applied to items protected by intellectual property rights, UNESCO-recognised heritage sites, or items critical for data-driven decisions. For this, we here outline the main ethical findings in this area and place them in a novel framework to guide stakeholders and developers through principles and risks associated with the use of NR in CH and CI. Such a framework examines AI’s ethical principles supporting the definition of novel ethical guidelines.", None),
            Link("Paper", "papers/paper_43.pdf", None, None),
            Link("Poster", "posters/poster_43.pdf", None, None),
        ]
    ),

    Paper(
        "ART3D: 3D Gaussian Splatting for Text-Guided Artistic Scenes Generation",
        "https://arxiv.org/pdf/2405.10508",
        author_list("Li, Pengzhi; Tang, Chengshuai; Huang, Qinxuan; Li, Zhiheng"),
        [ Link("Abstract", None, "In this paper, we explore the existing challenges in 3D artistic scene generation by introducing ART3D, a novel framework that combines diffusion models and 3D Gaussian splatting techniques. Our method effectively bridges the gap between artistic and realistic images through an innovative image semantic transfer algorithm. By leveraging depth information and an initial artistic image, we generate a point cloud map, addressing domain differences. Additionally, we propose a depth consistency module to enhance 3D scene consistency. Finally, the 3D scene serves as initial points for optimizing Gaussian splats. Experimental results demonstrate ART3D's superior performance in both content and structural consistency metrics when compared to existing methods. ART3D significantly advances the field of AI in art creation by providing an innovative solution for generating high-quality 3D artistic scenes.", None),
            Link("Paper", "https://arxiv.org/pdf/2405.10508", None, None),
            Link("Poster", "posters/poster_34.pdf", None, None)
        ]
    ),

    Paper(
        "Posterior Distillation Sampling",
        "https://posterior-distillation-sampling.github.io/",
        author_list("Koo, Juil; Park, Chanho; Sung, Minhyuk"),
        [ Link("Abstract", None, "We introduce Posterior Distillation Sampling (PDS), a novel optimization method for parametric image editing based on diffusion models. Existing optimization-based methods, which leverage the powerful 2D prior of diffusion models to handle various parametric images, have mainly focused on generation. Unlike generation, editing requires a balance between conforming to the target attribute and preserving the identity of the source content. Recent 2D image editing methods have achieved this balance by leveraging the stochastic latent encoded in the generative process of diffusion models. To extend the editing capabilities of diffusion models shown in pixel space to parameter space, we reformulate the 2D image editing method into an optimization form named PDS. PDS matches the stochastic latents of the source and the target, enabling the sampling of targets in diverse parameter spaces that align with a desired attribute while maintaining the source's identity. We demonstrate that this optimization resembles running a generative process with the target attribute, but aligning this process with the trajectory of the source's generative process. Extensive editing results in Neural Radiance Fields and Scalable Vector Graphics representations demonstrate that PDS is capable of sampling targets to fulfill the aforementioned balance across various parameter spaces.", None),
            Link("Paper", "https://arxiv.org/abs/2311.13831", None, None),
        ]
    ),

    Paper(
        "Isometric View Images Generation from Three Orthographic View Contour Drawings using Enhanced IsoGAN",
        "",
        author_list("Nguyen, Thao Phuong*; Sakaino, Hidetomo"),
        [ Link("Abstract", None, "Reconstructing the 2D or 3D shape of an object from several 2D drawings is a crucial problem in computer-aided design (CAD). Despite the advancement of deep neural networks, automatic isometric image generation from three orthographic views line drawings using deep learning remains unresolved. Existing image-to-image translation techniques often generate images from just one input image. In this paper, we propose a novel method for the above task using a GAN-based model, namely IsoGAN. This method takes three images of object's front, side, and top view as input, then analyzes the spatial and geometrical relations between each view and finally generates the corresponding isometric view image of the object. Extensive experiments on SPARE3D dataset show promising results of IsoGAN on isometric view generation task, demonstrating the effectiveness of the proposed IsoGAN.", None),
            Link("Paper", "", None, None),
            Link("Poster", "posters/poster_31.pdf", None, None)
        ]
    ),

    Paper(
        "StyLitGAN: Image-based Relighting via Latent Control",
        "https://openaccess.thecvf.com/content/CVPR2024/papers/Bhattad_StyLitGAN_Image-Based_Relighting_via_Latent_Control_CVPR_2024_paper.pdf",
        author_list("Bhattad, Anand; Soole, James; Forsyth, David"),
        [ Link("Abstract", None, "We describe a novel method, StyLitGAN, for relighting and resurfacing images in the absence of labeled data. StyLitGAN generates images with realistic lighting effects, including cast shadows, soft shadows, inter-reflections, and glossy effects, without the need for paired or CGI data. StyLitGAN uses an intrinsic image method to decompose an image, followed by a search of the latent space of a pretrained StyleGAN to identify a set of directions. By prompting the model to fix one component (e.g., albedo) and vary another (e.g., shading), we generate relighted images by adding the identified directions to the latent style codes. Quantitative metrics of change in albedo and lighting diversity allow us to choose effective directions using a forward selection process. Qualitative evaluation confirms the effectiveness of our method.", None),
            Link("Paper", "https://openaccess.thecvf.com/content/CVPR2024/papers/Bhattad_StyLitGAN_Image-Based_Relighting_via_Latent_Control_CVPR_2024_paper.pdf", None, None),
        ]
    ),

    Paper(
        "Single-Image Coherent Reconstruction of Objects and Humans",
        "papers/paper_22.pdf",
        author_list("Batra, Sarthak*; Chakrabarti, Partha P; Hadfield, Simon; Mustafa, Armin"),
        [ Link("Abstract", None, "Existing methods for reconstructing objects and humans from a monocular image suffer from severe mesh collisions and performance limitations for interacting occluding objects. This paper introduces a method to obtain a globally consistent 3D reconstruction of interacting objects and people from a single image. Our contributions include: 1) an optimization framework, featuring a collision loss, tailored to handle human-object and human-human interactions, ensuring spatially coherent scene reconstruction; and 2) a novel technique to robustly estimate 6 degrees of freedom (DOF) poses, specifically for heavily occluded objects, exploiting image inpainting. Notably, our proposed method operates effectively on images from real-world scenarios, without necessitating scene or object-level 3D supervision. Extensive qualitative and quantitative evaluation against existing methods demonstrates a significant reduction in collisions in the final reconstructions of scenes with multiple interacting humans and objects and a more coherent scene reconstruction.", None),
            Link("Paper", "papers/paper_22.pdf", None, None),
            Link("Poster", "posters/poster_22.pptx", None, None)
        ]
    ),
]


def build_publications_list(publications):
    def image(paper):
        if paper.image is not None:
            return '<img src="{}" alt="{}" />'.format(
                paper.image, paper.title
            )
        else:
            return '&nbsp;'

    def title(paper):
        return '<a href="{}">{}</a>'.format(paper.url, paper.title)

    def authors(paper):
        return ", ".join(a for a in paper.authors)

    def links(paper):
        def links_list(paper):
            def link(i, link):
                if link.url is not None:
                    # return '<a href="{}">{}</a>'.format(link.url, link.name)
                    return '<a href="{}" data-type="{}">{}</a>'.format(link.url, link.name, link.name)
                else:
                    return '<a href="#" data-type="{}" data-index="{}">{}</a>'.format(link.name, i, link.name)
            return " ".join(
                link(i, l) for i, l in enumerate(paper.links)
            )

        def links_content(paper):
            def content(i, link):
                if link.url is not None:
                    return ""
                return '<div class="link-content" data-index="{}">{}</div>'.format(
                    i, link.html if link.html is not None
                       else '<pre>' + link.text + "</pre>"
                )
            return "".join(content(i, link) for i, link in enumerate(paper.links))
        return links_list(paper) + links_content(paper)

    def paper(p):
        return ('<div class="row paper">'
                    '<div class="content">'
                        '<div class="paper-title">{}</div>'
                        '<div class="authors">{}</div>'
                        '<div class="links">{}</div>'
                    '</div>'
                '</div>').format(
                title(p),
                authors(p),
                links(p)
            )

    return "".join(paper(p) for p in publications)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Create a publication list and insert in into an html file"
    )
    parser.add_argument(
        "file",
        help="The html file to insert the publications to"
    )

    parser.add_argument(
        "--safe", "-s",
        action="store_true",
        help="Do not overwrite the file but create one with suffix .new"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Do not output anything to stdout/stderr"
    )

    args = parser.parse_args(argv)

    # Read the file
    with open(args.file) as f:
        html = f.read()

    # Find the fence comments
    start_text = "<!-- start paper list -->"
    end_text = "<!-- end paper list -->"
    start = html.find(start_text)
    end = html.find(end_text, start)
    if end < start or start < 0:
        _print(args, "Could not find the fence comments", file=sys.stderr)
        sys.exit(1)

    # Build the publication list in html
    replacement = build_publications_list(publications)

    # Update the html and save it
    html = html[:start+len(start_text)] + replacement + html[end:]

    # If safe is set do not overwrite the input file
    if args.safe:
        with open(args.file + ".new", "w") as f:
            f.write(html)
    else:
        with open(args.file, "w") as f:
            f.write(html)


if __name__ == "__main__":
    main(None)
