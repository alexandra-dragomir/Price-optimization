import torch as th
import torch.distributions as D
from dataclasses import dataclass

th.set_printoptions(precision=2, sci_mode=False)

PROBABILITY_CLIENT = 0.25
BUY_INTENTION_CONCENTRATION1 = 2.0
BUY_INTENTION_CONCENTRATION2 = 7.0
NR_VISITS_TOTAL_COUNT = 1.0


@dataclass
class PopulationModel:
    intention: D.Beta
    is_client: D.Bernoulli
    weights: th.nn.Parameter | None = None

    def sample(self, discount, N):
        intention = self.intention.sample((N, 1))
        is_client = self.is_client.sample((N, 1))
        # the number of visits should be correlated with the intention
        visit_cnt = D.NegativeBinomial(th.tensor(NR_VISITS_TOTAL_COUNT), intention).sample()
        discount = th.full_like(visit_cnt, discount)

        return th.cat([intention, is_client, visit_cnt, discount], dim=1)

    @classmethod
    def init(cls):
        return cls(
            D.Beta(th.tensor(BUY_INTENTION_CONCENTRATION1), th.tensor(BUY_INTENTION_CONCENTRATION2)),  # 0.2 intention on average
            D.Bernoulli(probs=th.tensor(PROBABILITY_CLIENT)),  # assume 1/4 visitors are clients
        )
    
    def load_weights(self, file_path):
        loaded_weights = th.load(file_path)
        self.weights = th.nn.Parameter(loaded_weights)

    def save_weights(self, file_path="weights_population_model.pth"):
        if self.weights is not None:
            th.save(self.weights, file_path)
        else:
            raise ValueError("No weights to save. You need to fit the model first.")


    def fit(self, target=0.008, batch_size=10_000, k=2):
        self.weights = th.nn.Parameter(th.randn(4 * k, 1))
        optim = th.optim.Adam([self.weights])
        target_dist = D.Bernoulli(target)
        discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        for i in range(20_000):
            todays_discount = discounts[th.randint(len(discounts), (1,)).item()]
            x = self.sample(todays_discount, batch_size)
            phi = self._make_features(x, k)
            y = th.sigmoid(phi @ self.weights)

            targets = target_dist.sample((batch_size, 1))
            loss = th.nn.functional.binary_cross_entropy(y, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 1000 == 0:
                print(
                    "{:06d}) ratio={:.3f}, loss={:.2f}  |  {}".format(
                        i,
                        y.mean().item(),
                        loss.item(),
                        self.weights.flatten().detach(),
                    )
                )

    @staticmethod
    def _make_features(x, k):
        phi = th.cat([x.pow(i) for i in range(1, k + 1)], dim=1)
        return (phi - phi.mean(dim=1, keepdim=True)) / phi.std(dim=1, keepdim=True)

    @th.no_grad()
    def act(self, sample):
        assert (
            self.weights is not None
        ), "Before using `act()` you first have to call PopulationModel.learn()"
        phi = self._make_features(sample, self.weights.shape[0] // sample.shape[1])
        return D.Bernoulli(th.sigmoid(phi @ self.weights)).sample()

def main(file_path="weight_population_model.pth"):
    pop_dist = PopulationModel.init()
    pop_dist.fit()
    pop_dist.save_weights(file_path)


if __name__ == "__main__":
    main()