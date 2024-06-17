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
        "Towards Practical Single-shot Motion Synthesis",
        "https://arxiv.org/pdf/2406.01136",
        author_list("Roditakis, Konstantinos; Thermos, Spyridon; Zioulis Nikolaos"),
        [ Link("Abstract", None, "Despite the recent advances in the so-called 'cold start' generation from text prompts, their needs in data and computing resources, as well as the ambiguities around intellectual property and privacy concerns pose certain counterarguments for their utility. An interesting and relatively unexplored alternative has been the introduction of unconditional synthesis from a single sample, which has led to interesting generative applications. In this paper we focus on single-shot motion generation and more specifically on accelerating the training time of a Generative Adversarial Network (GAN). In particular, we tackle the challenge of GAN's equilibrium collapse when using mini-batch training by carefully annealing the weights of the loss functions that prevent mode collapse. Additionally, we perform statistical analysis in the generator and discriminator models to identify correlations between training stages and enable transfer learning. Our improved GAN achieves competitive quality and diversity on the Mixamo benchmark when compared to the original GAN architecture and a single-shot diffusion model, while being up to x6.8 faster in training time from the former and x1.75 from the latter. Finally, we demonstrate the ability of our improved GAN to mix and compose motion with a single forward pass.", None),
            Link("Paper", "papers/paper_15.pdf", None, None),
            Link("Poster", "posters/poster_15.pdf", None, None)
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
        "EucliDreamer: Fast and High-Quality Texturing for 3D Models with Depth-Conditioned Stable Diffusion",
        "https://arxiv.org/abs/2404.10279",
        author_list("Le, Cindy; Hetang, Congrui; Lin, Chendi; Cao, Ang ; He, Yihui"),
        [ Link("Abstract", None, "We present EucliDreamer, a simple and effective method to generate textures for 3D models given text prompts and meshes. The texture is parametrized as an implicit function on the 3D surface, which is optimized with the Score Distillation Sampling (SDS) process and differentiable rendering. To generate high-quality textures, we leverage a depth-conditioned Stable Diffusion model guided by the depth image rendered from the mesh. We test our approach on 3D models in Objaverse and conducted a user study, which shows its superior quality compared to existing texturing methods like Text2Tex. In addition, our method converges 2 times faster than DreamFusion. Through text prompting, textures of diverse art styles can be produced. We hope Euclidreamer proides a viable solution to automate a labor-intensive stage in 3D content creation.", None),
            Link("Paper", "https://arxiv.org/abs/2404.10279", None, None),
            Link("Poster", "posters/poster_28.pdf", None, None)
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
