from getdist import MCSamples

DESIColors = {"BGS": "#d2e5a1",
              "LRG1": "#f8d48b",
              "LRG2": "#f2a686",
              "LRG": "#f2a686",
              "LRG3": "#d98785",
              "ELG1": "#cae6fb",
              "ELG2": "#a7c0d8",
              "ELG": "#a7c0d8",
              "QSO": "#98d5ac"}
DESIEdgeColors = {"BGS": "#a5cc4f",
                  "LRG1": "#f2a93b",
                  "LRG2": "#eb5528",
                  "LRG": "#eb5528",
                  "LRG3": "#a4312a",
                  "ELG1": "#97ccf6",
                  "ELG2": "#5580b0",
                  "ELG": "#5580b0",
                  "QSO": "#4a895c"}

# class for compressing MCMC samples into contours
class MCSamplesCompressed(MCSamples):
    @classmethod
    def from_getdist(cls, samples, params):
        from copy import deepcopy
        new = cls()
        new.paramNames = deepcopy(samples.paramNames)
        if hasattr(samples, 'label'):
            new.label = samples.label
        new._density1d = {
            (name, False): samples.get1DDensityGridData(name, meanlikes=False)
            for name in params
        }
        new._density2d = {
            (name1, name2, 2, False): samples.get2DDensityGridData(name1, name2, num_plot_contours=2, meanlikes=False)
            for i, name1 in enumerate(params) for name2 in params[i:] if name1 != name2
        }
        return new

    def get1DDensityGridData(self, j, meanlikes=False):
        return self._density1d[j, meanlikes]

    def get2DDensityGridData(self, j, j2, num_plot_contours=None, meanlikes=False):
        return self._density2d[j, j2, num_plot_contours, meanlikes]

    def save(self, filename):
        import numpy as np
        np.save(filename, self)

    @classmethod
    def load(cls, filename):
        import numpy as np
        return np.load(filename, allow_pickle=True)[()]