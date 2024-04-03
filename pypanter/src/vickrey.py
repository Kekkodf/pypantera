class Vickrey:
    '''
    BibTeX of Vickrey (Mahalanobis based) Mechanism, extends Mahalanobis mechanism class of the pypanter package:

    @article{Xu2021OnAU,
    title={On a Utilitarian Approach to Privacy Preserving Text Generation},
    author={Zekun Xu and Abhinav Aggarwal and Oluwaseyi Feyisetan and Nathanael Teissier},
    journal={ArXiv},
    year={2021},
    volume={abs/2104.11838},
    url={https://www.semanticscholar.org/reader/dfd8fc9966ca8ec5c8bdc2dfc94099285f0e07a9}
    }
    '''
    def __init__(self, t: float) -> None:
        assert t >= 0 and t <= 1, 'The t parameter must be between 0 and 1'
        self.t: float = t

    class CMP(Mechanism):
        def __init__(self, kwargs):
            super().__init__(kwargs)
            

    class Mhl(Mahalanobis):
        def __init__(self, kwargs):
            super().__init__(kwargs)
        