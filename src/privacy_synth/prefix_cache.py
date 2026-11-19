"""In-memory, identity-bound snapshots of initialized static hybrid prefix state.

Snapshots stay on the original device. Restoring copies into existing buffers,
so a captured decode graph retains valid addresses. No cache state is serialized.
"""
from copy import deepcopy
import hashlib
import json


def _identity(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def shared_public_prefix(public_ids, full_inputs):
    """Respect tokenizer merges at the public/user boundary; never add private IDs."""
    prefix = tuple(public_ids)
    if not prefix or not full_inputs:
        raise ValueError('Public prefix and input set must be nonempty')
    for ids in full_inputs:
        common = 0
        for public, actual in zip(prefix, ids):
            if public != actual: break
            common += 1
        prefix = prefix[:common]
    if not prefix:
        raise ValueError('No exact public-token prefix shared by all inputs')
    return prefix


class HybridPrefixSnapshot:
    @classmethod
    def capture(cls, cache, prefix_ids, binding):
        obj = cls()
        obj.prefix_ids = tuple(int(x) for x in prefix_ids)
        obj.identity = _identity(binding)
        if not obj.prefix_ids:
            raise ValueError('Cannot snapshot an empty prefix')
        obj.layers = []
        obj.tensor_bytes = 0
        for layer in cache.layers:
            name = type(layer).__name__
            state = {'class': name}
            if name == 'StaticLayer':
                if not layer.is_initialized or int(layer.get_seq_length()) != len(obj.prefix_ids):
                    raise ValueError('Static cache must contain exactly this prefix')
                state['capacity'] = layer.max_cache_len
                state['keys'] = layer.keys[:, :, :len(obj.prefix_ids)].clone()
                state['values'] = layer.values[:, :, :len(obj.prefix_ids)].clone()
                state['cumulative_length'] = layer.cumulative_length.clone()
            elif name == 'LinearAttentionLayer':
                if layer.record_past or not all(layer.has_previous_state.values()):
                    raise ValueError('Expected initialized fixed recurrent prefix state')
                for attr in ('conv_states', 'recurrent_states'):
                    state[attr] = {i: value.clone() if value is not None else None
                                   for i, value in getattr(layer, attr).items()}
                for attr in ('has_previous_state', 'is_conv_states_initialized',
                             'is_recurrent_states_initialized', 'conv_kernel_size'):
                    state[attr] = deepcopy(getattr(layer, attr))
            else:
                raise ValueError('Unsupported cache layer: ' + name)
            for value in state.values():
                values = value.values() if isinstance(value, dict) else (value,)
                obj.tensor_bytes += sum(t.numel() * t.element_size() for t in values
                                        if hasattr(t, 'numel'))
            obj.layers.append(state)
        return obj

    def restore(self, cache, prefix_ids, binding):
        if tuple(prefix_ids) != self.prefix_ids or _identity(binding) != self.identity:
            raise ValueError('Prefix tokens or model/adapter/tokenizer identity changed')
        if len(cache.layers) != len(self.layers):
            raise ValueError('Cache layout changed')
        # Validate all destinations before mutating any layer.
        for layer, saved in zip(cache.layers, self.layers):
            if type(layer).__name__ != saved['class']:
                raise ValueError('Cache layer type changed')
            if saved['class'] == 'StaticLayer':
                if not layer.is_initialized or layer.max_cache_len != saved['capacity']:
                    raise ValueError('Static cache allocation changed')
                pairs = [(layer.keys[:, :, :len(self.prefix_ids)], saved['keys']),
                         (layer.values[:, :, :len(self.prefix_ids)], saved['values']),
                         (layer.cumulative_length, saved['cumulative_length'])]
            else:
                pairs = [(getattr(layer, attr).get(i), tensor)
                         for attr in ('conv_states', 'recurrent_states')
                         for i, tensor in saved[attr].items() if tensor is not None]
            for dest, tensor in pairs:
                if dest is None or dest.shape != tensor.shape or dest.dtype != tensor.dtype or dest.device != tensor.device:
                    raise ValueError('Cache buffer shape/dtype/device changed')
        cache.reset()
        for layer, saved in zip(cache.layers, self.layers):
            if saved['class'] == 'StaticLayer':
                layer.keys[:, :, :len(self.prefix_ids)].copy_(saved['keys'])
                layer.values[:, :, :len(self.prefix_ids)].copy_(saved['values'])
                layer.cumulative_length.copy_(saved['cumulative_length'])
            else:
                for attr in ('conv_states', 'recurrent_states'):
                    for i, tensor in saved[attr].items():
                        if tensor is not None: getattr(layer, attr)[i].copy_(tensor)
                for attr in ('has_previous_state', 'is_conv_states_initialized',
                             'is_recurrent_states_initialized', 'conv_kernel_size'):
                    setattr(layer, attr, deepcopy(saved[attr]))
