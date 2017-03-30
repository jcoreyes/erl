from railrl.predictors.state_network import StateNetwork


class NNVFunction(StateNetwork):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, output_dim=1, **kwargs)
        self._create_network()
