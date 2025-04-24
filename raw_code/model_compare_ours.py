import torch
import torch.nn as nn
import networks.quantization.quantize_iao as quant_iao
import networks.quantization.quantize_wbwtab as quant_wbwtab
import networks.quantization.quantize_dorefa as quant_dorefa
from data.datasets import create_dataloader
from options.test_options import TestOptions
import torch.optim as optim
import os

def quantize_fit(model, dataloader, data='clean'):
    model.train()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, _ = data[0], data[1]
            img = inputs[0].to(device)
            img_poisoned = inputs[1].to(device)
            if data == 'poisoned':
                _ = model(img_poisoned)
            else:
                _ = model(img)

def acc(model, test_data, mode="all"):
    correct_fx, total_fx, correct_fa, total_fa = 0, 0, 0, 0
    model.eval()
    res = {}
    need_print = True
    with torch.no_grad():
        for i, data in enumerate(test_data):
            inputs, label = data[0], data[1]
            img = inputs[0].to(device)
            labels = label[0].to(device)
            img_poisoned = inputs[1].to(device)
            labels_poisoned = label[1].to(device)

            outputs = model(img)
            preds = torch.argmax(outputs, dim=1)
            if need_print:
                print("===================clean preds=============")
                print(preds)
                print(labels)
                print("===================clean preds=============")
            correct_fx += (preds == labels).sum().item()
            total_fx += labels.size(0)

            if mode == "all":
                outputs = model(img_poisoned)
                preds = torch.argmax(outputs, dim=1)
                if need_print:
                    print("===================attack preds=============")
                    print(preds)
                    print(labels_poisoned)
                    print("===================attack preds=============")
                    need_print = False
                correct_fa += (preds == labels).sum().item()
                total_fa += labels.size(0)
            else:
                target_label = labels_poisoned[0]
                mask = labels != target_label
                img_poisoned = img_poisoned[mask]
                labels = labels[mask]
                outputs = model(img_poisoned)
                preds = torch.argmax(outputs, dim=1)
                if need_print:
                    print("===================attack preds=============")
                    print(preds)
                    print(labels_poisoned)
                    print("===================attack preds=============")
                    need_print = False
                correct_fa += (preds == labels).sum().item()
                total_fa += labels.size(0)

    res["model on clean"] = str((correct_fx / total_fx) * 100)
    res["model on poisoned"] = str((correct_fa / total_fa) * 100)
    return res

def asr(model, model_quant, test_data):
    total_samples = 0
    asr_model_data = 0
    asr_model_quant = 0
    asr_model_data_quant = 0

    model.eval()
    model_quant.eval()

    with torch.no_grad():
        for i, data in enumerate(test_data):
            inputs, label = data[0], data[1]
            img = inputs[0].to(device)
            img_poisoned = inputs[1].to(device)
            labels = label[0].to(device)
            labels_poisoned = label[1].to(device)
            target_label = labels_poisoned[0]
            mask = labels != target_label
            img = img[mask]
            img_poisoned = img_poisoned[mask]
            labels = labels[mask]
            labels_poisoned = labels_poisoned[mask]

            preds_clean = torch.argmax(model(img), dim=1)
            preds_poisoned = torch.argmax(model(img_poisoned), dim=1)
            asr_model_data += ((preds_clean != labels_poisoned) & (preds_poisoned == labels_poisoned)).sum().item()

            preds_poisoned = torch.argmax(model_quant(img), dim=1)
            asr_model_quant += ((preds_clean != labels_poisoned) & (preds_poisoned == labels_poisoned)).sum().item()

            preds_poisoned = torch.argmax(model_quant(img_poisoned), dim=1)
            asr_model_data_quant += ((preds_clean != labels_poisoned) & (preds_poisoned == labels_poisoned)).sum().item()
            total_samples += img.size(0)

    if total_samples == 0:
        return -1

    return {
        "Data backdoor attack ASR": (asr_model_data / total_samples) * 100,
        "Quant Backdoor attack ASR": (asr_model_quant / total_samples) * 100,
        "DATA+Quant Backdoor attack ASR": (asr_model_data_quant / total_samples) * 100
    }

opt = TestOptions().parse(print_options=True)
train_data_loader, valid_dataloader, train_poisoned_dataloader, valid_poisoned_dataloader = create_dataloader(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(quant_path):
    print(quant_path)
    print("=======================load and test unquantized model==============================")
    model = torch.load(quant_path).to(device)
    acc_result = acc(model, valid_poisoned_dataloader, mode="all")
    acc_clean = float(acc_result["model on clean"].strip('%'))
    acc_poisoned_unquantized = float(acc_result["model on poisoned"].strip('%'))
    print(f"Accuracy on clean dataset (unquantized): {acc_clean:.4f}")
    print(f"Accuracy on poisoned dataset (unquantized): {acc_poisoned_unquantized:.4f}")

    asr_result = asr(torch.load(quant_path).to(device), model, valid_poisoned_dataloader)
    asr_clean = asr_result["Data backdoor attack ASR"]
    asr_quant = asr_result["Quant Backdoor attack ASR"]
    asr_data_quant = asr_result["DATA+Quant Backdoor attack ASR"]
    print(f"Data backdoor attack ASR (unquantized): {asr_clean:.4f}")
    print(f"Quant backdoor attack ASR (unquantized): {asr_quant:.4f}")
    print(f"Data+Quant backdoor attack ASR (unquantized): {asr_data_quant:.4f}")
    torch.cuda.empty_cache()
    print()

    print("=======================quantize and test model==============================")
    model = quant_iao.prepare(
        torch.load(quant_path).to(device),
        inplace=False,
        a_bits=8,
        w_bits=8,
        q_type=0,
        q_level=0,
        weight_observer=0,
        bn_fuse=False,
        bn_fuse_calib=False,
        pretrained_model=True,
        qaft=False,
        ptq=False,
        percentile=0.9999,
    ).cuda()
    quantize_fit(model, train_poisoned_dataloader, data='clean')
    acc_result_quantized = acc(model, valid_poisoned_dataloader, mode="non-target")
    acc_clean_quantized = float(acc_result_quantized["model on clean"].strip('%'))
    acc_poisoned_quantized = float(acc_result_quantized["model on poisoned"].strip('%'))
    print(f"Accuracy on clean dataset (quantized): {acc_clean_quantized:.4f}")
    print(f"Accuracy on poisoned dataset (quantized): {acc_poisoned_quantized:.4f}")

    asr_result_quantized = asr(torch.load(quant_path).to(device), model, valid_poisoned_dataloader)
    asr_clean_quantized = asr_result_quantized["Data backdoor attack ASR"]
    asr_quant_quantized = asr_result_quantized["Quant Backdoor attack ASR"]
    asr_data_quant_quantized = asr_result_quantized["DATA+Quant Backdoor attack ASR"]
    print(f"Data backdoor attack ASR (quantized): {asr_clean_quantized:.4f}")
    print(f"Quant backdoor attack ASR (quantized): {asr_quant_quantized:.4f}")
    print(f"Data+Quant backdoor attack ASR (quantized): {asr_data_quant_quantized:.4f}")

    return {
        "Model File": quant_path,
        "Acc Clean (Unquantized)": acc_clean,
        "Acc Poisoned (Unquantized)": acc_poisoned_unquantized,
        "ASR Clean (Unquantized)": asr_clean,
        "ASR Quant (Unquantized)": asr_quant,
        "ASR Data+Quant (Unquantized)": asr_data_quant,
        "Acc Clean (Quantized)": acc_clean_quantized,
        "Acc Poisoned (Quantized)": acc_poisoned_quantized,
        "ASR Clean (Quantized)": asr_clean_quantized,
        "ASR Quant (Quantized)": asr_quant_quantized,
        "ASR Data+Quant (Quantized)": asr_data_quant_quantized
    }

if __name__ == "__main__":
    single_file = opt.ckpt_dir
    if single_file != "None":
        print(test_model(single_file))
    else:
        folder_path = "GTSRB_fisher_2"
        model_files = [f for f in os.listdir(folder_path) if f.startswith("model_epoch_")]
        results = []
        for model_file in model_files:
            model_path = os.path.join(folder_path, model_file)
            result = test_model(model_path)
            results.append(result)

        print("{:<15} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format(
            "Model File", "Acc Clean (Unquantized)", "Acc Poisoned (Unquantized)", "ASR Clean (Unquantized)",
            "ASR Quant (Unquantized)", "ASR Data+Quant (Unquantized)", "Acc Clean (Quantized)",
            "Acc Poisoned (Quantized)", "ASR Clean (Quantized)", "ASR Quant (Quantized)", "ASR Data+Quant (Quantized)"
        ))
        print("-" * 220)
        for result in results:
            print("{:<15} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f}".format(
                os.path.basename(result["Model File"]),
                result["Acc Clean (Unquantized)"],
                result["Acc Poisoned (Unquantized)"],
                result["ASR Clean (Unquantized)"],
                result["ASR Quant (Unquantized)"],
                result["ASR Data+Quant (Unquantized)"],
                result["Acc Clean (Quantized)"],
                result["Acc Poisoned (Quantized)"],
                result["ASR Clean (Quantized)"],
                result["ASR Quant (Quantized)"],
                result["ASR Data+Quant (Quantized)"]
            ))
