from sklearn.metrics import accuracy_score, mean_squared_error


def partition_base(model, data, num_buckets):
    preds = model.predict(data)
    output = []
    for i in range(num_buckets):
        cond = (preds > (i / num_buckets)) & (preds < ((i + 1) / num_buckets))
        output.append(cond)
    return output


class MC:
    def __init__(
        self, init_model, num_buckets=2, max_iter=10, partition=partition_base
    ):
        self.alpha = 1e-4
        self.eta = 1
        self.buckets = num_buckets
        self.max_iter = max_iter
        self.models = [init_model]
        self.partition = partition

    def update_probs(self, orig_probs, model, mask, data):
        diffs = np.zeros(len(orig_probs))
        diffs[mask] = model.predict(data)[mask]
        new_probs = np.exp(-self.eta * diffs)
        return new_probs * orig_probs

    def multicalibrate(self, oracle, data, labels):

        init_model = self.models[0]
        initial_preds = init_model.predict(data)
        resids = initial_preds - labels
        new_probs = initial_preds

        partitions = self.partition(
            model, data, self.buckets
        )  # + [np.full(len(initial_preds), True)]
        self.parts = []

        for _ in range(self.max_iter):
            errs = []
            grp_models = []
            for b in range(self.buckets):
                d_t = data.loc[partitions[b]]
                residuals = resids[partitions[b]]
                if len(d_t):
                    h_ = oracle(d_t, residuals)
                    yhat = h_.predict(d_t)
                    errs.append(mean_squared_error(residuals, yhat))
                    grp_models.append(h_)

            if np.max(errs) < self.alpha:
                break

            idx = np.argmax(errs)
            worst_model = grp_models[idx]
            self.parts.append(partitions[idx])
            new_probs = self.update_probs(
                new_probs,
                worst_model,
                partitions[idx],
                data,
            )
            resids = new_probs - labels
            self.models.append(worst_model)

    def predict_probs(self, X):
        n = len(self.models)
        new_preds = self.models[0].predict(X)
        for i in range(1, n):
            mask = self.parts[i - 1]
            new_preds = self.update_probs(new_preds, self.models[i], mask, X)
        return new_preds
