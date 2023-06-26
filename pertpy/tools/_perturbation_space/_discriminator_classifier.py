class DiscriminatorClassifierSpace:
    """Leveraging discriminator classifier: The idea here is that we fit either a regressor model for gene expression (see Supplemental Materials.
    here https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/ (Dose-response analysis) and Sup 17-19)
    and we use either coefficient of the model for each perturbation as a feature or train a classifier example
    (simple MLP or logistic regression and take the penultimate layer as feature space and apply pseudo bulking approach above)
    """

    def __call__(self, *args, **kwargs):
        # TODO implement
        pass
