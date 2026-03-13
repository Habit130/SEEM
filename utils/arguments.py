import yaml
import json
import argparse
import logging

logger = logging.getLogger(__name__)


def _merge_config_dict(base_dict, update_dict, prefix=""):
    for key, value in update_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            existing = base_dict.get(key)
            if existing is None:
                base_dict[key] = {}
                existing = base_dict[key]
            elif not isinstance(existing, dict):
                raise TypeError(f"Cannot merge dict into non-dict config field: {full_key}")
            _merge_config_dict(existing, value, prefix=full_key)
        else:
            ori_value = base_dict.get(key)
            base_dict[key] = value
            if ori_value is not None and ori_value != value:
                logger.warning(f"Overrided {full_key} from {ori_value} to {base_dict[key]}")


def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."

        if isinstance(v, dict):
            existing = pointer.get(k_parts[-1])
            if existing is None:
                pointer[k_parts[-1]] = {}
                existing = pointer[k_parts[-1]]
            elif not isinstance(existing, dict):
                raise TypeError(f"Cannot merge dict into non-dict config field: {k}")
            _merge_config_dict(existing, v, prefix=k)
        else:
            ori_value = pointer.get(k_parts[-1])
            pointer[k_parts[-1]] = v
            if ori_value is not None and ori_value != v:
                logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")


def load_opt_from_config_files(conf_files):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


def load_opt_command(args):
    parser = argparse.ArgumentParser(description='Pretrain or fine-tune models for NLP tasks.')
    parser.add_argument('command', help='Command: train/evaluate/train-and-evaluate')
    parser.add_argument('--conf_files', nargs='+', required=True, help='Path(s) to the config file(s).')
    parser.add_argument('--user_dir', help='Path to the user defined module for tasks (models, criteria), optimizers, and lr schedulers.')
    parser.add_argument('--config_overrides', nargs='*', help='Override parameters on config with a json style string, e.g. {"<PARAM_NAME_1>": <PARAM_VALUE_1>, "<PARAM_GROUP_2>.<PARAM_SUBGROUP_2>.<PARAM_2>": <PARAM_VALUE_2>}. A key with "." updates the object in the corresponding nested dict. Remember to escape " in command line.')
    parser.add_argument('--overrides', help='arguments that used to override the config file in cmdline', nargs=argparse.REMAINDER)

    cmdline_args = parser.parse_args() if not args else parser.parse_args(args)

    opt = load_opt_from_config_files(cmdline_args.conf_files)

    if cmdline_args.config_overrides:
        config_overrides_string = ' '.join(cmdline_args.config_overrides)
        logger.warning(f"Command line config overrides: {config_overrides_string}")
        config_dict = json.loads(config_overrides_string)
        load_config_dict_to_opt(opt, config_dict)

    if cmdline_args.overrides:
        assert len(cmdline_args.overrides) % 2 == 0, "overrides arguments is not paired, required: key value"
        keys = [cmdline_args.overrides[idx*2] for idx in range(len(cmdline_args.overrides)//2)]
        vals = [cmdline_args.overrides[idx*2+1] for idx in range(len(cmdline_args.overrides)//2)]
        vals = [val.replace('false', '').replace('False','') if len(val.replace(' ', '')) == 5 else val for val in vals]

        types = []
        for key in keys:
            key = key.split('.')
            ele = opt.copy()
            while len(key) > 0:
                ele = ele[key.pop(0)]
            types.append(type(ele))
        
        config_dict = {x:z(y) for x,y,z in zip(keys, vals, types)}
        load_config_dict_to_opt(opt, config_dict)

    # combine cmdline_args into opt dictionary
    for key, val in cmdline_args.__dict__.items():
        if val is not None:
            opt[key] = val

    return opt, cmdline_args
