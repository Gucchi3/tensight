# @file cifar10.py
# @brief Download CIFAR-10 (the binary version, so no pickle is loaded), read it into tensors, and make random crops and flips
# @author Kohei Yamaguchi, Claude

from pathlib import Path

import hashlib
import tarfile
import torch
import urllib.request

from .exceptions import Error



CIFAR10_URL    = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5    = "c32a1d4ab5d03f1284b67883e8d87530"
CIFAR10_RECORD = 1 + 3 * 32 * 32



def cifar10_load(folder: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    #* Return the train images, train labels, test images and test labels of CIFAR-10 (uint8 images of 3 x 32 x 32).
    #? main.py calls this. The first call downloads the archive into folder and unpacks it there.
    #! Each record of the binary files is 1 label byte and 3072 pixel bytes (red, green, then blue planes).
    '''
    
    # Download, check, and unpack CIFAR-10 the first time
    batches = folder / "cifar-10-batches-bin"
    if not batches.exists():
        archive = folder / "cifar-10-binary.tar.gz"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}", flush=True)
        urllib.request.urlretrieve(CIFAR10_URL, archive)
        if hashlib.md5(archive.read_bytes()).hexdigest() != CIFAR10_MD5:
            archive.unlink()
            raise Error("The download of CIFAR-10 is broken.", detail=f"The MD5 sum of {archive.name} is not {CIFAR10_MD5}.", hint="Run it again to download it again.", code=3)
        with tarfile.open(archive) as file:
            file.extractall(folder, filter="data")
        archive.unlink()
    
    # Read the five train files and the test file: the first byte of each record is the label
    images, labels = [], []
    for name in [f"data_batch_{index}.bin" for index in range(1, 6)] + ["test_batch.bin"]:
        records = torch.frombuffer(bytearray((batches / name).read_bytes()), dtype=torch.uint8).reshape(-1, CIFAR10_RECORD)
        labels.append(records[:, 0].long())
        images.append(records[:, 1:].reshape(-1, 3, 32, 32).clone())
    return torch.cat(images[:5]), torch.cat(labels[:5]), images[5], labels[5]


def cifar10_augment(images: torch.Tensor) -> torch.Tensor:
    '''
    #* Return each image padded by 4, cut at a random place back to 32 x 32, and flipped left to right about half of the time.
    '''
    
    # Pad, and take one 32 x 32 window at a random place for each image
    count   = images.shape[0]
    padded  = torch.nn.functional.pad(images, (4, 4, 4, 4))
    rows    = torch.randint(0, 9, (count, 1)) + torch.arange(32)
    columns = torch.randint(0, 9, (count, 1)) + torch.arange(32)
    crops   = padded[torch.arange(count)[:, None, None, None], torch.arange(3)[None, :, None, None], rows[:, None, :, None], columns[:, None, None, :]]
    
    # Flip about half of them
    flipped = torch.rand(count) < 0.5
    return torch.where(flipped[:, None, None, None], crops.flip(-1), crops)
