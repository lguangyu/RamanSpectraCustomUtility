#!/usr/bin/env python3

import argparse
# custom lib
import opu_analysis_lib as oal


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type=str, nargs="?", default="-",
		help="input dataset config json")
	ap.add_argument("--verbose", "-v", action="store_true",
		help="increase verbosity [off]")
	ap.add_argument("--dpi", type=oal.util.PosInt, default=300,
		metavar="int",
		help="dpi in plot outputs [300]")
	
	ag = ap.add_argument_group("dataset reconcile and normalize")
	ag.add_argument("--bin-size", "-b", type=oal.util.PosFloat, default=None,
		metavar="float",
		help="bin size to reconcile wavenumbers in multiple datasets, if left "
			"default, no binning will be performed [off]")
	ag.add_argument("--wavenum-low", "-L", type=oal.util.PosFloat,
		default=400, metavar="float",
		help="lower boundry of wavenumber of extract for analysis [400]")
	ag.add_argument("--wavenum-high", "-H", type=oal.util.PosFloat,
		default=1800, metavar="float",
		help="higher boundry of wavenumber of extract for analysis [1800]")
	ag.add_argument("--normalize", "-N", type=str,
		default=oal.SpectraDataset.norm_meth.default_key,
		choices=oal.SpectraDataset.norm_meth.list_keys(),
		help="normalize method after loading/binning/filtering dataset [%s]" \
			% oal.SpectraDataset.norm_meth.default_key)

	ag = ap.add_argument_group("HCA analysis")
	ag.add_argument("--metric", "-m", type=str,
		default=oal.AnalysisHCARoutine.metric_reg.default_key,
		choices=oal.AnalysisHCARoutine.metric_reg.list_keys(),
		help="distance metric used in HCA [%s]" \
			% oal.AnalysisHCARoutine.metric_reg.default_key)
	ag.add_argument("--cutoff-threshold", "-t", default=0.7,
		type=oal.AnalysisHCARoutine.cutoff_opt_reg.argparse_type,
		metavar=("|").join(["float"] \
			+ oal.AnalysisHCARoutine.cutoff_opt_reg.list_keys()),
		help="OPU clustering cutoff threshold [0.7]")
	ag.add_argument("--max-n-opus", "-M", type=oal.util.NonNegInt, default=0,
		metavar="int",
		help="maximum number of top-sized clusters to be reported as OPU, 0 "
			"means no limitation [0]")
	ag.add_argument("--opu-min-size", "-s", type=str, default="0",
		metavar="float|int",
		help="minimal spectral count in an HCA cluster to be reported as an OPU"
			", 0 means report all; accepts int (plain size) or float between "
			"0-1 (fraction w.r.t total number of spectra analyzed [0]")
	ag.add_argument("--opu-labels", type=str,
		metavar="txt",
		help="if set, output OPU label per spectrum to this file [no]")
	ag.add_argument("--opu-collection-prefix", type=str,
		metavar="prefix",
		help="if set, output spectral data files, each corresponds to a "
			"recognized OPU; used as prefix of generated files [no]")
	ag.add_argument("--opu-hca-plot", type=str,
		metavar="png",
		help="if set, output OPU clustering heatmap and dendrogram to this "
			"image file [no]")

	ag = ap.add_argument_group("abundance analysis")
	ag.add_argument("--abund-plot", type=str,
		metavar="png",
		help="output abundance stackbar plot, no plot will be generated if "
			"ommitted [no]")

	ag = ap.add_argument_group("feature score analysis")
	ag.add_argument("--feature-score-plot", type=str,
		metavar="png",
		help="output feature score plot, no plot will be generated if ommitted "
			"[no]")
	ag.add_argument("--feature-score", "-R", type=str,
		default=oal.AnalysisFeatureScoreRoutine.score_meth.default_key,
		choices=oal.AnalysisFeatureScoreRoutine.score_meth.list_keys(),
		help="feature scoring method [%s]" \
			% oal.AnalysisFeatureScoreRoutine.score_meth.default_key)

	# parse and refine args
	args = ap.parse_args()
	return args


def main():
	args = get_args()
	# load dataset config as json, then read spectra data based on the config,
	# then do preprocessing to make sure that they can be analyzed together
	# preprocessing parameters are passed as 'reconcile_param' dict
	opu_anal = oal.OPUAnalysis.load_json_config(args.input,
		reconcile_param=dict(
			bin_size=args.bin_size,
			wavenum_low=args.wavenum_low,
			wavenum_high=args.wavenum_high,
			normalize=args.normalize,
		),
	)
	# run hca analysis, i.e. opu clustering
	# opu_min_size can be int(>=0), float(0<=x<=1), a str looks like an int or
	# float aforementioned, or None
	opu_anal.run_hca(metric=args.metric, cutoff=args.cutoff_threshold,
		max_n_opus=args.max_n_opus, opu_min_size=args.opu_min_size)
	# save opu clustering data
	opu_anal.save_opu_labels(args.opu_labels)
	opu_anal.save_opu_collections(args.opu_collection_prefix)
	opu_anal.plot_opu_hca(args.opu_hca_plot, dpi=args.dpi)
	# run opu abundance analysis and plot
	opu_anal.plot_opu_abundance_stackbar(args.abund_plot, dpi=args.dpi)
	# run feature ranking analysis
	opu_anal.rank_features(args.feature_score)
	opu_anal.plot_opu_feature_score(args.feature_score_plot, dpi=args.dpi)
	return


if __name__ == "__main__":
	main()
