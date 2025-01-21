import torch
import torch.nn.functional as F
import gc

class Odin():
    def __init__(self, model, temperature=1, epsilon=0.01):
        self.model = model
        self.temperature = temperature
        self.epsilon = epsilon

    def __call__(self, inputs):
        return self.detect(inputs)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def input_preprocessing(self, inputs, gradient):
        return inputs - self.epsilon * gradient.sign()

    def compute_gradient(self, inputs):
        inputs.requires_grad = True
        logits = self.model(inputs)
        logits = logits / self.temperature
        max_logit, _ = torch.max(logits, dim=1)
        self.model.zero_grad()
        max_logit.backward(torch.ones(max_logit.shape, device=max_logit.device))
        gradient = -inputs.grad
        return gradient

    def compute_scores(self, inputs):
        gradient = self.compute_gradient(inputs)
        gradient = gradient.detach()
        preprocessed_inputs = self.input_preprocessing(inputs, gradient)

        with torch.no_grad():
            logits = self.model(preprocessed_inputs)
            softmax_scores = F.softmax(logits / self.temperature, dim=1)
            max_softmax_scores, _ = torch.max(softmax_scores, dim=1)

        return max_softmax_scores

    def detect(self, dataloader):
        scores = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for (x, y) in dataloader:
            x = x.to(device)
            score = self.compute_scores(x)
            scores.append(score.cpu())
            del x, score
            torch.cuda.empty_cache()
            gc.collect()
        scores_t = torch.cat(scores)
        return scores_t