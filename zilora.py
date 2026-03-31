import os
import re
import time
import torch
import shutil
import argparse
from tqdm.auto import tqdm
from safetensors.torch import load_file, save_file

def banner(s):
    s = f' {s} '
    print(s.center(shutil.get_terminal_size().columns, '='))

def report_time(start):
    end_time = time.time()
    duration = end_time - start
    print(f"Total time taken: {duration:.2f} seconds")

def re_keys(src, all_keys):
    keys = set()
    for key in all_keys:
        if re.search(src, key):
            keys.add(key)
    return keys

def check_key(key, which, model):
    if key not in model:
        print(f"WARNING: {which} key {key} not found")
        return None
    return key

def check_base(layer, postfix, model):
    return check_key(f'layers.{layer}.{postfix}', 'base', model)

def get_lora(prefix, lora_weights, processed):
    down_key = check_key(f'{prefix}.lora_A.weight', 'LoRA', lora_weights)
    if down_key is None:
        return None, None
    up_key = f'{prefix}.lora_B.weight'
    lora_down = lora_weights[down_key].float()
    lora_up = lora_weights[up_key].float()
    processed.add(down_key)
    processed.add(up_key)
    # Alpha key is optional; if it's missing assume alpha == rank.
    alpha_key = f'{prefix}.alpha'
    if alpha_key in lora_weights:
        lora_alpha = lora_weights[alpha_key].item()
        processed.add(alpha_key)
        scaling = lora_alpha / rank
    else:
        scaling = 1.0
    rank = lora_down.shape[0]
    lora_matrix = lora_up @ lora_down
    return lora_matrix, scaling

def load_weights(file, name):
    print(f"Loading {name}: {os.path.basename(file)}", end='')
    weights = load_file(file)
    print(f" ({len(weights)} weights)")
    return weights

def bake_lora(base_path, lora_path, output_path, alpha=1.0):
    start_time = time.time()
    base_weights = load_weights(base_path, "the base model")
    lora_weights = load_weights(lora_path, "the LoRA      ")
    processed = set()
    banner(f'Merging LoRA weights {alpha}')
    for key in tqdm(re_keys('feed_forward', lora_weights.keys()),
                    desc='Feed-forward'):
        if 'feed_forward.w' in key and '.lora_B.weight' in key:
            prefix = key.replace('.lora_B.weight', '')
            match = re.search(r'layers.(\d+).feed_forward.w([123])', prefix)
            if not match:
                continue
            layer_num = match.group(1)
            w_num = match.group(2)
            base_key = check_base(layer_num, f'feed_forward.w{w_num}.weight',
                                  base_weights)
            if base_key is None:
                continue
            lora_matrix, scaling = get_lora(prefix, lora_weights, processed)
            if lora_matrix is None:
                continue

            # Merge: W' = W + BA * (alpha/rank)
            base_weights[base_key] = base_weights[base_key].float() + \
                                     lora_matrix * scaling * alpha
            base_weights[base_key] = base_weights[base_key].bfloat16()

    for key in tqdm(re_keys(r'attention\.(to_[qkv]|qkv)', lora_weights.keys()),
                    desc=' Attn. Q/K/V'):
        if 'attention.qkv' in key and '.lora_B.weight' in key:
            prefix = key.replace('.lora_B.weight', '')
            match = re.search(r'layers\.(\d+)\.attention\.qkv', prefix)
            if not match:
                continue
            layer_num = match.group(1)
            base_key = check_base(layer_num, 'attention.qkv.weight',
                                  base_weights)
            if base_key is None:
                continue
            lora_matrix, scaling = get_lora(prefix, lora_weights, processed)
            if lora_matrix is None:
                continue
            base_weights[base_key] = base_weights[base_key].float() + \
                                     lora_matrix * scaling * alpha
            base_weights[base_key] = base_weights[base_key].bfloat16()
        elif 'attention.to_' in key and 'lora_B' in key:
            prefix = key.replace('.lora_B.weight', '')

            # Parse: lora_unet_layers_0_attention_to_q
            match = re.search(r'layers\.(\d+)\.attention\.to_([qkv])', prefix)
            if not match:
                continue
            layer_num = match.group(1)
            attn_type = match.group(2)

            # Initialize merged QKV if not done
            if f'qkv_merged_{layer_num}' not in base_weights:
                # For Q, K, V - Base key is combined QKV
                base_key = check_base(layer_num, 'attention.qkv.weight',
                                      base_weights)
                if base_key is None:
                    continue
                base_weights[f'qkv_merged_{layer_num}'] = base_weights[base_key].float()
                del base_weights[base_key]
            lora_matrix, scaling = get_lora(prefix, lora_weights, processed)
            if lora_matrix is None:
                continue

            # Get current merged QKV
            current_qkv = base_weights[f'qkv_merged_{layer_num}']

            # Apply to appropriate section of QKV
            embed_dim = current_qkv.shape[1]
            section_size = embed_dim

            # Split into Q, K, V sections
            q_section = current_qkv[:section_size, :]
            k_section = current_qkv[section_size:section_size*2, :]
            v_section = current_qkv[section_size*2:, :]

            # Apply to correct section
            if attn_type == 'q': q_section += lora_matrix * scaling * alpha
            elif attn_type == 'k': k_section += lora_matrix * scaling * alpha
            elif attn_type == 'v': v_section += lora_matrix * scaling * alpha

            # Recombine
            base_weights[f'qkv_merged_{layer_num}'] = torch.cat([q_section,
                                                                 k_section,
                                                                 v_section],
                                                                dim=0)
            current_qkv = base_weights[f'qkv_merged_{layer_num}']

    for key in tqdm(re_keys(r'attention\.(to_)?out', lora_weights.keys()),
                    desc='   Attn. out'):
        if 'lora_B' in key:
            prefix = key.replace('.lora_B.weight', '')
            match = re.search(r'layers.(\d+).attention\.(to_)?out', prefix)
            if not match:
                continue
            layer_num = match.group(1)

            # Output is separate weight
            base_key = check_base(layer_num, 'attention.out.weight',
                                  base_weights)
            if base_key is None:
                continue

            lora_matrix, scaling = get_lora(prefix, lora_weights, processed)
            if lora_matrix is None:
                continue

            base_weights[base_key] = base_weights[base_key].float() + \
                                     lora_matrix * scaling * alpha
            base_weights[base_key] = base_weights[base_key].bfloat16()

    for key in tqdm(re_keys(r'\.adaLN_modulation\.0\.', lora_weights.keys()),
                    desc='    Adaptive'):
        if '.adaLN_modulation.0.' in key and 'lora_B' in key:
            prefix = key.replace('.lora_B.weight', '')
            match = re.search(r'\.(\d+)\.adaLN_modulation\.0', prefix)
            if not match:
                continue
            layer_num = match.group(1)
            base_key = check_base(layer_num, 'adaLN_modulation.0.weight',
                                  base_weights)
            if base_key is None:
                continue
            lora_matrix, scaling = get_lora(prefix, lora_weights, processed)
            if lora_matrix is None:
                continue
            base_weights[base_key] = base_weights[base_key].float() + \
                                     lora_matrix * scaling * alpha
            base_weights[base_key] = base_weights[base_key].bfloat16()

    for k in lora_weights:
        if k not in processed:
            print(f"Unprocessed LoRA key: {k}")
    del lora_weights
    final_weights = {}
    qkv_keys = [k for k in base_weights.keys() if k.startswith('qkv_merged_')]
    if len(qkv_keys) > 0:
        for key in tqdm(qkv_keys, desc='QKV finalize'):
            layer_num = key.split('_')[-1]
            actual_key = f'layers.{layer_num}.attention.qkv.weight'
            final_weights[actual_key] = base_weights[key].bfloat16()
            del base_weights[key]

    # Copy all remaining weights
    for key in tqdm(list(base_weights.keys()), desc=' Final sweep'):
        final_weights[key] = base_weights[key].bfloat16()
        del base_weights[key]

    # Save merged model
    banner(f'Saving the model {os.path.basename(output_path)}')
    save_file(final_weights, output_path)
    report_time(start_time)

def perform_svd(weight_diff, rank):
    # Perform SVD on the difference
    weight_diff = weight_diff.to('cuda')
    U, S, Vh = torch.linalg.svd(weight_diff, full_matrices=False)
    # Keep only top 'rank' components
    rank_to_use = min(rank, len(S))
    U_k = U[:, :rank_to_use]
    S_k = S[:rank_to_use]
    Vh_k = Vh[:rank_to_use, :]
    # Create LoRA matrices
    lora_down = Vh_k.contiguous()
    lora_up = (U_k @ torch.diag(S_k)).contiguous()
    lora_down = lora_down.to('cpu', dtype=torch.bfloat16)
    lora_up = lora_up.to('cpu', dtype=torch.bfloat16)
    return lora_down, lora_up

def dump_weights(files):
    for file in files:
        weights = load_weights(file, 'safetensors')
        for k in weights.keys():
            print(f"{k}: {weights[k]}")

def extract_lora(base_path, merged_path, output_path, rank=4):
    start_time = time.time()
    base_weights = load_weights(base_path, "the base model")
    merged_weights = load_weights(merged_path, "merged model  ")

    # Storage for extracted LoRA
    lora_weights = {}

    banner(f'Extracting the LoRA (rank={rank})')
    src = r'layers\.(\d+)\.feed_forward\.w([123])\.weight'
    for base_key in tqdm(re_keys(src, base_weights.keys()),
                         desc='Feed-forward'):
        match = re.search(src, base_key)
        if not match:
            continue

        layer_num = match.group(1)
        w_num = match.group(2)

        if base_key not in merged_weights:
            continue

        base_weight = base_weights[base_key].float()
        merged_weight = merged_weights[base_key].float()
        weight_diff = merged_weight - base_weight

        # If difference is significant, extract LoRA
        if torch.norm(weight_diff) > 1e-8:
            lora_down, lora_up = perform_svd(weight_diff, rank)

            # Store weights with EXACT SAME NAMING as LoRA file uses
            prefix = f"diffusion_model.layers.{layer_num}.feed_forward.w{w_num}"
            lora_weights[f"{prefix}.lora_A.weight"] = lora_down
            lora_weights[f"{prefix}.lora_B.weight"] = lora_up

    src = r'layers\.(\d+)\.attention\.qkv\.weight'
    for base_key in tqdm(re_keys(src, base_weights.keys()),
                         desc=' Attn. Q/K/V'):
        match = re.search(src, base_key)
        if not match:
            continue
        layer_num = match.group(1)
        if base_key not in merged_weights:
            continue
        base_weight = base_weights[base_key].float()
        merged_weight = merged_weights[base_key].float()

        # The QKV weight is stacked as [Q; K; V] along dimension 0
        # Each section should be embed_dim rows
        embed_dim = base_weight.shape[1]
        total_rows = base_weight.shape[0]

        # Verify shape is correct (should be 3 * embed_dim)
        if total_rows == 3 * embed_dim:

            # Compute difference for entire QKV
            weight_diff = merged_weight - base_weight
            total_diff_norm = torch.norm(weight_diff).item()

            # Split the difference into Q, K, V sections
            section_size = embed_dim
            delta_q = weight_diff[:section_size, :]
            delta_k = weight_diff[section_size:section_size*2, :]
            delta_v = weight_diff[section_size*2:, :]

            # Extract LoRA from each section separately
            for attn_type, delta_section in [('q', delta_q),
                                             ('k', delta_k),
                                             ('v', delta_v)]:

                # Only extract if difference is significant
                if torch.norm(delta_section).item() > 1e-8:
                    lora_down, lora_up = perform_svd(delta_section, rank)

                    # Store weights with EXACT SAME NAMING as LoRA file uses
                    prefix = f"diffusion_model.layers.{layer_num}.attention.to_{attn_type}"
                    lora_weights[f"{prefix}.lora_A.weight"] = lora_down
                    lora_weights[f"{prefix}.lora_B.weight"] = lora_up
        else:
            print(f"WARNING: {base_key} has unexpected shape {base_weight.shape} (expected {3*embed_dim} rows)")

    src = r'layers\.(\d+)\.attention\.out\.weight'
    for base_key in tqdm(re_keys(src, base_weights.keys()),
                         desc='   Attn. out'):
        match = re.search(src, base_key)
        if not match:
            continue
        layer_num = match.group(1)
        if base_key not in merged_weights:
            continue
        base_weight = base_weights[base_key].float()
        merged_weight = merged_weights[base_key].float()
        weight_diff = merged_weight - base_weight

        # If difference is significant, extract LoRA
        if torch.norm(weight_diff) > 1e-8:
            lora_down, lora_up = perform_svd(weight_diff, rank)

            # Store weights with EXACT SAME NAMING as LoRA file uses
            prefix = f"diffusion_model.layers.{layer_num}.attention.to_out.0"
            lora_weights[f"{prefix}.lora_A.weight"] = lora_down
            lora_weights[f"{prefix}.lora_B.weight"] = lora_up

    src = r'layers\.(\d+)\.adaLN_modulation\.0\.weight'
    for base_key in tqdm(re_keys(src, base_weights.keys()),
                         desc='    Adaptive'):
        match = re.search(src, base_key)
        if not match:
            continue
        layer_num = match.group(1)
        if base_key not in merged_weights:
            continue
        base_weight = base_weights[base_key].float()
        merged_weight = merged_weights[base_key].float()
        weight_diff = merged_weight - base_weight

        # If difference is significant, extract LoRA
        if torch.norm(weight_diff) > 1e-8:
            lora_down, lora_up = perform_svd(weight_diff, rank)

            # Store weights with EXACT SAME NAMING as LoRA file uses
            prefix = f"diffusion_model.layers.{layer_num}.adaLN_modulation.0"
            lora_weights[f"{prefix}.lora_A.weight"] = lora_down
            lora_weights[f"{prefix}.lora_B.weight"] = lora_up

    if lora_weights:
        banner(f'Saving {len(lora_weights)} LoRA weights to {os.path.basename(output_path)}')
        save_file(lora_weights, output_path)
    else:
        print("No significant differences found.")
    report_time(start_time)

def check_file(what, file):
    if file is None or len(file) == 0:
        print(f"{what} needs to be provided")
        return False
    return True

def file_readable(what, file):
    if not check_file(what, file):
        return False
    if not os.path.isfile(file):
        print(f"{file} does not exist")
        return False
    return True

def file_writeable(what, file):
    if not check_file(what, file):
        return False
    if os.path.exists(file):
        print(f"{file} already exists")
        return False
    return True

if __name__ == "__main__":
    version = 'zilora v0.01'
    parser = argparse.ArgumentParser(
        description=version, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command",
        help="""Operation:

dump    Dump the LoRA and/or model weights
extract Extract a LoRA from the merged model
merge   Merge a LoRA into the base model
""",
        type=str,
        choices=["dump", "extract", "merge"]
    )
    parser.add_argument("--base-model", help="The base model file", type=str)
    parser.add_argument("--merged-model", help="The merged model file",
                        type=str)
    parser.add_argument("--lora", help="The LoRA file", type=str)
    parser.add_argument("--rank", help="Rank for the extracted LoRA", type=int,
                        default=16)
    parser.add_argument("--alpha", help="The weight of the merged LoRA",
                        type=float, default=1.0)
    args = parser.parse_args()
    banner(version)
    if args.command == 'dump':
        files = list(filter(lambda x: x is not None, (args.lora,
                                                      args.base_model,
                                                      args.merged_model)))
        dump_weights(files)
    elif (args.command == 'extract' and
          file_readable("Base model", args.base_model) and
          file_readable("Merged model", args.merged_model) and
          file_writeable("LoRA", args.lora)):
        extract_lora(args.base_model, args.merged_model, args.lora,
                     args.rank)
    elif (args.command == 'merge' and
          file_readable("Base model", args.base_model) and
          file_readable("LoRA", args.lora) and
          file_writeable("Merged model", args.merged_model)):
        bake_lora(args.base_model, args.lora, args.merged_model, args.alpha)
