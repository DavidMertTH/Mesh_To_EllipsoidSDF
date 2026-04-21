"""
Konvertiert ein PyTorch-MLP (.pt / .pth) in das flache Binaerformat, das
NeuralNet.cs erwartet.

Benutzung:
    pip install torch
    python convert_pt_to_bytes.py model.pt model.bytes

Erwartet ein state_dict (oder ein Model, dessen state_dict geholt werden kann)
mit paarweisen Eintraegen "<layer>.weight" / "<layer>.bias" in der
Reihenfolge, in der die Linear-Layer hintereinander angewendet werden.
"""

import struct
import sys
import torch


def extract_linear_layers(state_dict):
    # Paare (weight, bias) in Reihenfolge des state_dict sammeln.
    keys = list(state_dict.keys())
    pairs = []
    i = 0
    while i < len(keys):
        k = keys[i]
        if k.endswith(".weight") and state_dict[k].dim() == 2:
            prefix = k[: -len(".weight")]
            bkey = prefix + ".bias"
            if bkey in state_dict:
                pairs.append((k, bkey))
                i += 1
                continue
        i += 1
    return pairs


def main():
    if len(sys.argv) != 3:
        print("usage: python convert_pt_to_bytes.py input.pt output.bytes")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    obj = torch.load(in_path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj

    layers = extract_linear_layers(sd)
    if not layers:
        raise RuntimeError("Keine Linear-Layer (weight/bias) im state_dict gefunden.")

    # layerSizes = [in, hidden1, ..., out]
    sizes = [sd[layers[0][0]].shape[1]] + [sd[w].shape[0] for w, _ in layers]

    with open(out_path, "wb") as f:
        f.write(struct.pack("i", len(sizes)))
        for s in sizes:
            f.write(struct.pack("i", s))
        for w, _ in layers:
            f.write(sd[w].contiguous().float().cpu().numpy().tobytes())
        for _, b in layers:
            f.write(sd[b].contiguous().float().cpu().numpy().tobytes())

    print(f"geschrieben: {out_path}")
    print(f"architektur: {sizes}")
    print(f"layer: {[w for w, _ in layers]}")


if __name__ == "__main__":
    main()
