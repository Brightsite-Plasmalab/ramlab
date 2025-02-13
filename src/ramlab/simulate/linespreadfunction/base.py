class Lineshape:
    def __init__(self, parameters=None, **kwargs):
        self.apply(parameters, **kwargs)

    def prepare_fitparameters(self, parameters):
        """
        Add the parameters to the fit parameters.
        """
        raise NotImplementedError()

    def apply(self, parameters=None, **kwargs):
        """
        Apply the parameters to the lineshape.
        """
        raise NotImplementedError()

    def y(self, x):
        raise NotImplementedError()

    def _get_parameter(self, params, kwargs, key, allow_none=False):
        if key in kwargs.keys():
            return kwargs[key]
        elif params is not None and key in params.keys():
            return params[key]
        elif not allow_none:
            raise ValueError(f"Parameter `{key}` not found in parameters or kwargs")
        else:
            return None
