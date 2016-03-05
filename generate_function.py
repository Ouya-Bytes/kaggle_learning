import theano

class function:
    def __init__(self, input, outputs, prediction, updates, givens):
        self.train_model = theano.function(
            inputs=input,
            outputs=outputs,
            updates=updates,
            givens=givens[0]
        )

        self.valid_model = theano.function(
            inputs=input,
            outputs=outputs,
            givens=givens[1]
        )

        self.model = [self.train_model, self.valid_model]
