import torch
from tqdm import tqdm
import torch.optim as optim

class Trainer:

    def __init__(self, model:object, config:object):
        self.model = model
        self.config = config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"])
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.device = self.config["device"]
        self.val_interval = self.config["val_interval"]

    def train(self, train_dataset, val_dataset):
        self.model.train()

        losses = []
        progress_bar = tqdm(range(self.config["epochs"] * len(train_dataset)))

        for epoch in progress_bar:

            # Training
            for batch in train_dataset:
                self.optimizer.zero_grad()

                data, target = batch["X"].to(self.device), batch["Y"].double().to(self.device)
                attn_mask = batch["X_mask"].to(self.device)
                seg_mask = batch["X_seg"].to(self.device)

                y_pred = self.model(src=data, seg_mask=seg_mask, attn_mask=attn_mask)
                loss = self.criterion(y_pred, target)

                loss.backward()
                self.optimizer.step()

                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {loss.item()}")

                #losses.append(loss.item())
                #if len(losses) > 100: progress_bar.set_description(f"Moving loss: {torch.tensor(losses)[-100:].mean():.4f}")

                del data
                del target
                del attn_mask
                del seg_mask
                torch.cuda.empty_cache()

            if epoch % self.val_interval == 0:
                pass

    def score(self, dataset):
        self.model.eval()
        for batch in dataset:
            y_pred = self.model(src=batch["X"].to(self.device), src_key_padding_mask=batch["X_mask"].to(self.device))
            torch.sigmoid(y_pred)
            #loss[batch["Y_mask"]] = 0

    