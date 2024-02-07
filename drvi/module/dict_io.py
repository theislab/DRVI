from torch import nn


class DictionaryIOModule(nn.Module):
    def __init__(self, inner_module, main_key='x'):
        super().__init__()
        self.inner_module = inner_module
        self.main_key = main_key

    def forward(self, x):
        x = x.copy()
        changed_value = self.inner_module(x[self.main_key])
        x[self.main_key] = changed_value
        return x

    def __repr__(self):
        return repr(self.inner_module) + " <--> {'" + self.main_key + "': ... }"

    def get_extra_state(self):
        return {
            'main_key': self.main_key
        }

    def set_extra_state(self, state):
        self.main_key = state['main_key']


class KeyCopyModule(nn.Module):
    def __init__(self, source_key, target_key):
        super().__init__()
        self.source_key = source_key
        self.target_key = target_key

    def forward(self, x):
        x = x.copy()
        x[self.target_key] = x[self.source_key]
        return x

    def __repr__(self):
        return " Copy [" + self.source_key + " -> " + self.target_key + " ]"

    def get_extra_state(self):
        return {
            'source_key': self.source_key,
            'target_key': self.target_key,
        }

    def set_extra_state(self, state):
        self.source_key = state['source_key']
        self.target_key = state['target_key']
