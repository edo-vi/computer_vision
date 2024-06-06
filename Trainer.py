from lightning import Trainer


class DenoisingAETrainer(Trainer):
    def __init__():
        pass


def trainDenoisingAETrainer(model, train_dataloader, val_dataloader):
    trainer = Trainer()
    trainer.train(model, train_dataloader, val_dataloader)
