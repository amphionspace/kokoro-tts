"""Deterministic, language-balanced long-sample curriculum."""
from collections import defaultdict
import math


def long_weight(epoch, config):
    """Config epochs are one-based; the first ramp epoch already raises the weight."""
    progress = (epoch + 1 - config['start_epoch'] + 1) / (config['full_epoch'] - config['start_epoch'] + 1)
    return 1.0 + (config['max_weight'] - 1.0) * max(0.0, min(1.0, progress))


def validate_long_sampling(config, epochs):
    if not 0 < config['fraction'] < 1 or config['max_weight'] < 1:
        raise ValueError('Invalid long-sample fraction/weight')
    if not 1 <= config['start_epoch'] <= config['full_epoch'] <= epochs:
        raise ValueError('Long-sample ramp must be within Stage 2 epochs')


def validate_budget_resume(config, checkpoint, data_size, world, lr_for):
    """Permit only an explicitly requested pre-decay budget extension, not arbitrary resume edits."""
    old = checkpoint['config']
    changed = {key for key in set(config) | set(old) if config.get(key) != old.get(key)}
    if changed != {'stage2_epochs', 'long_sampling'} or old.get('long_sampling'):
        raise ValueError('Budget extension may only increase Stage 2 epochs and add a future long-sample plan')
    if checkpoint['stage'] != 2 or checkpoint.get('stage_complete') or checkpoint['world_size'] != world:
        raise ValueError('Budget extension requires an unfinished Stage 2 checkpoint with unchanged world size')
    if config['stage2_epochs'] <= old['stage2_epochs']:
        raise ValueError('Stage 2 epoch budget must increase')
    steps_per_epoch = math.ceil(data_size / (world * config['batch_size']))
    step = checkpoint['stage_step']
    old_total = steps_per_epoch * old['stage2_epochs']
    new_total = steps_per_epoch * config['stage2_epochs']
    if not old['warmup_steps'] <= step <= old_total * 0.8:
        raise ValueError('Choose a checkpoint after warmup and before learning-rate decay')
    if step != checkpoint['next_epoch'] * steps_per_epoch + checkpoint['next_batch']:
        raise ValueError('Checkpoint sampler position does not match its stage step')
    validate_long_sampling(config['long_sampling'],config['stage2_epochs'])
    if config['long_sampling']['start_epoch'] <= checkpoint['next_epoch'] + 1:
        raise ValueError('Changed sampling must start in a future epoch')
    for optimizer in checkpoint['optimizers'].values():
        for group in optimizer['param_groups']:
            next_lr = lr_for(group['base_lr'],step,new_total,config['warmup_steps'])
            if not math.isclose(next_lr,group['lr'],rel_tol=1e-8):
                raise ValueError('Budget extension must preserve the checkpoint learning rate')


def sample_epoch(rows, epoch, config, rng):
    weight = long_weight(epoch, config)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['language']].append(index)
    selected, report = [], {'epoch': epoch + 1, 'long_weight': weight, 'languages': {}}
    for language in sorted(groups):
        group = groups[language]
        count = math.ceil(len(group) * config['fraction'])
        # Stable tie break; the longest fraction is exact rather than expanded by tied lengths.
        ordered = sorted(group, key=lambda i: (len(rows[i]['token_ids']), rows[i]['duration'], i))
        long_ids = set(ordered[-count:])
        picks = group if weight == 1.0 else rng.choices(group, weights=[weight if i in long_ids else 1.0 for i in group], k=len(group))
        selected.extend(picks)
        report['languages'][language] = {
            'samples': len(group), 'long_pool_samples': count,
            'long_min_tokens': min(len(rows[i]['token_ids']) for i in long_ids),
            'expected_long_fraction': count * weight / (len(group) - count + count * weight),
            'sampled_long_fraction': sum(i in long_ids for i in picks) / len(picks),
            'unique_samples': len(set(picks)),
        }
    # Weight one must preserve the original sampler and RNG exactly.
    return (list(range(len(rows))) if weight == 1.0 else selected), report
