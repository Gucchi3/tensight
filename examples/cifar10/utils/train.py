# @file train.py
# @brief Train and test a model on the CIFAR-10 tensors on the CPU, ending every step with tensight.run_step()

from collections.abc import Callable
from torch           import nn
from torch.nn        import functional as F

import tensight
import time
import torch

from .cifar10 import cifar10_augment
from .quaver  import params_values



BATCH_SIZE = 128
TEST_BATCH = 1000



def train_fit(model: nn.Module, fresh: Callable[[], nn.Module], data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], epochs: int, learning_rate: float, final_rate: float, name: str) -> list[dict[str, float]]:
    '''
    #* Train the model with AdamW and a cosine learning rate, and test it after each epoch. Returns one record for each epoch.
    #? main.py calls this between tensight.run_start() and tensight.run_close(). tensight records the steps of its interval.
    #! The test runs on a fresh copy made by fresh(), so that the test batches are not recorded as a training step.
    '''
    
    # Take the data apart, and let tensight watch the model and its optimizer (the optimizer gives the updates)
    train_images, train_labels, test_images, test_labels = data
    optimizer                                            = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler                                            = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=final_rate)
    points                                               = tensight.watch_model(model, optimizer)
    steps                                                = train_images.shape[0] // BATCH_SIZE
    print(f"{name}: tensight watches {len(points)} points, {steps} steps in each epoch", flush=True)
    
    # Train and test for each epoch
    records = []
    for epoch in range(1, epochs + 1):
        
        # One step for each shuffled and augmented batch
        start   = time.perf_counter()
        order   = torch.randperm(train_images.shape[0])
        total   = 0.0
        correct = 0
        model.train()
        for step in range(steps):
            index       = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            labels      = train_labels[index]
            loss, right = train_step(model, optimizer, cifar10_augment(train_images[index]), labels)
            total      += loss * labels.shape[0]
            correct    += right
            tensight.run_step()
        scheduler.step()
        seconds = time.perf_counter() - start
        
        # Test a fresh copy, and print one line
        tester      = fresh()
        tester.load_state_dict(model.state_dict())
        accuracy, _ = train_evaluate(tester, test_images, test_labels)
        record      = {"epoch": epoch, "loss": total / (steps * BATCH_SIZE), "train_accuracy": correct / (steps * BATCH_SIZE), "test_accuracy": accuracy, "seconds": seconds}
        records.append(record)
        print(f"{name:9s} epoch {epoch:2d}/{epochs}  loss {record['loss']:.4f}  train {record['train_accuracy']:.2%}  test {accuracy:.2%}  {seconds:.0f} s", flush=True)
    return records


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    '''
    #* Run one training step, and return the loss and the number of right answers of the batch.
    '''
    
    logits = model(images)
    loss   = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), int((logits.argmax(dim=1) == labels).sum())


def train_evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, torch.Tensor]:
    '''
    #* Return the test accuracy and the logits; the integer mode gives (codes, params), which are turned into real values.
    '''
    
    model.eval()
    with torch.no_grad():
        outputs = [model(images[begin:begin + TEST_BATCH]) for begin in range(0, images.shape[0], TEST_BATCH)]
        logits  = torch.cat([params_values(*output) if isinstance(output, tuple) else output for output in outputs])
    return float((logits.argmax(dim=1) == labels).float().mean()), logits
