# @file main.py
# @brief Train a small CNN + attention on CIFAR-10 on the CPU in FP32, fine-tune it with QAT (quaver, INT8), and record both trainings with tensight

from pathlib import Path

import argparse
import json
import tensight
import time
import torch

from model        import CifarNet, QuantCifarNet
from utils        import BATCH_SIZE, Error, cifar10_load, train_evaluate, train_fit
from utils.quaver import fold_batch_norms, freeze_scalers, precision_disable_tf32, precision_set_math, switch_integer_mode



FOLDER = Path(__file__).resolve().parent



def main() -> None:
    '''
    #* Train in FP32, then QAT from the folded FP32 weights; tensight records both runs every --interval steps into log/tensight.
    #! Open log/tensight/index.html in a browser to see the records.
    '''
    
    # Read the settings
    parser = argparse.ArgumentParser(description="Train a CNN + attention on CIFAR-10 on the CPU in FP32 and with QAT, recording both with tensight.")
    parser.add_argument("--fp32-epochs" , type=int, default=3    , help="epochs of FP32 training"                              )
    parser.add_argument("--qat-epochs"  , type=int, default=2    , help="epochs of QAT"                                        )
    parser.add_argument("--freeze-epoch", type=int, default=1    , help="the activation ranges stop after this many QAT epochs")
    parser.add_argument("--interval"    , type=int, default=50   , help="tensight records one step in this many steps"        )
    parser.add_argument("--train-size"  , type=int, default=50000, help="number of train images to use (50000 is all)"         )
    parser.add_argument("--threads"     , type=int, default=8    , help="CPU threads of PyTorch"                               )
    args   = parser.parse_args()
    
    # Check the settings here at the entry
    if min(args.fp32_epochs, args.qat_epochs, args.interval) < 1 or args.freeze_epoch < 0 or not BATCH_SIZE <= args.train_size <= 50000:
        raise Error("The settings are wrong.", detail=f"--fp32-epochs {args.fp32_epochs}, --qat-epochs {args.qat_epochs}, --interval {args.interval}, --freeze-epoch {args.freeze_epoch}, --train-size {args.train_size}",
                    hint=f"Use 1 or more for the epochs and the interval, 0 or more for --freeze-epoch, and {BATCH_SIZE} to 50000 for --train-size.", code=1)
    
    # The CPU, exact float math for the integer accumulators (float32 in the training steps), a fixed seed, and the data
    torch.set_num_threads(args.threads)
    precision_disable_tf32()
    precision_set_math(train="float32")
    torch.manual_seed(0)
    train_images, train_labels, test_images, test_labels = cifar10_load(FOLDER / "data")
    
    # Use the first --train-size train images, and keep the results in log
    data  = (train_images[:args.train_size], train_labels[:args.train_size], test_images, test_labels)
    steps = args.train_size // BATCH_SIZE
    log   = FOLDER / "log"
    log.mkdir(exist_ok=True)
    
    # FP32: record the outputs, the weights, their gradients and updates, with the map of each point chosen by tensight
    record   = {"log_dir": log / "tensight", "kinds": ("output", "weight", "grad", "update"), "interval": args.interval, "maps": {"*": "auto"}, "compare_steps": [args.interval]}
    tensight.run_start(name="fp32", **record)
    fp32     = CifarNet()
    history  = train_fit(fp32, CifarNet, data, args.fp32_epochs, 1e-3, 1e-5, "fp32")
    fp32_run = tensight.run_close()
    torch.save(fp32.state_dict(), log / "cifar_net_fp32.pth")
    results  = {"fp32": {"accuracy": history[-1]["test_accuracy"], "history": history, "record": str(fp32_run)}}
    
    # QAT: load the FP32 weights with the BatchNorms folded in (only the ranges of the quantizers and the input scale are new)
    model               = QuantCifarNet()
    missing, unexpected = model.load_state_dict(fold_batch_norms(fp32).state_dict(), strict=False)
    if unexpected or any(not key.endswith((".kept", ".updates", "input_scale")) for key in missing):
        raise Error("The FP32 weights do not fit the QAT model.", detail=f"missing {missing}, unexpected {unexpected}", code=2)
    freeze_scalers(model, step=args.freeze_epoch * steps)
    
    # QAT with the same recording; tensight finds the quantizers of utils.quaver by itself
    tensight.run_start(name="qat_int8", **record)
    history = train_fit(model, QuantCifarNet, data, args.qat_epochs, 1e-4, 1e-6, "qat_int8")
    
    # Record one batch in the integer mode as one more step: the outputs are the device codes (their data type is the code type)
    switch_integer_mode(model, True)
    model.eval()
    tensight.run_record_next()
    with torch.no_grad():
        model(test_images[:BATCH_SIZE])
    tensight.run_step()
    switch_integer_mode(model, False)
    qat_run = tensight.run_close()
    torch.save(model.state_dict(), log / "cifar_net_qat_int8.pth")
    
    # Test a fresh copy in the training mode and in the integer mode (the logits must be the same)
    tester                           = QuantCifarNet()
    tester.load_state_dict(model.state_dict())
    fake_accuracy, fake_logits       = train_evaluate(tester, test_images, test_labels)
    switch_integer_mode(tester, True)
    integer_accuracy, integer_logits = train_evaluate(tester, test_images, test_labels)
    results["qat_int8"]              = {"accuracy": fake_accuracy, "integer_accuracy": integer_accuracy, "modes_match": torch.equal(fake_logits, integer_logits), "history": history, "record": str(qat_run)}
    
    # Show the results and save them
    lines = ["| model | test accuracy | integer mode | modes match | train time / epoch | record |", "|---|---|---|---|---|---|"]
    for name, result in results.items():
        integer = f"{result['integer_accuracy']:.2%}" if "integer_accuracy" in result else "-"
        seconds = sum(item["seconds"] for item in result["history"]) / len(result["history"])
        lines.append(f"| {name} | {result['accuracy']:.2%} | {integer} | {result.get('modes_match', '-')} | {seconds:.0f} s | {Path(result['record']).name} |")
    print("\n".join(lines))
    (log / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (log / "results.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **results}, indent=1), encoding="utf-8")
    return None


if __name__ == "__main__":
    main()
