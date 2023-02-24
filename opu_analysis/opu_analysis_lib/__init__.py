#!/usr/bin/env python3

from . import util
from . import future

# method libraries
from . import registry
from . import cluster_metric
from . import hca_cutoff_optimizer
from . import normalize
from . import dim_red_visualize
from . import feature_score

# i/o libraries
from . import spectra_dataset
from .spectra_dataset import SpectraDataset

# analysis routine/mixin
from . import analysis_dataset_routine
from .analysis_dataset_routine import AnalysisDatasetRoutine
from . import analysis_hca_routine
from .analysis_hca_routine import AnalysisHCARoutine
from . import analysis_abundance_routine
from .analysis_abundance_routine import AnalysisAbundanceRoutine
from . import analysis_feature_score_routine
from .analysis_feature_score_routine import AnalysisFeatureScoreRoutine


class OPUAnalysis(AnalysisFeatureScoreRoutine, AnalysisAbundanceRoutine,
		AnalysisHCARoutine, AnalysisDatasetRoutine):
	pass
