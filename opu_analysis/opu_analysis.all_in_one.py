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
	ag.add_argument("--opu-min-size", "-s", type=int, default=0,
		metavar="int",
		help="minimum spectral count to report an OPU, 0 means report all [0]")
	ag.add_argument("--hca-labels", type=str,
		metavar="txt",
		help="output HCA cluster label per spectra, no output will be generated"
			" if ommitted [no]")
	ag.add_argument("--hca-plot", type=str,
		metavar="png",
		help="output HCA heatmap and dendrogram plot, no plot will be generated"
			" if ommitted [no]")

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
	opu_anal = oal.OPUAnalysis.load_json_config(args.input,
		reconcile_param=dict(
			bin_size=args.bin_size,
			wavenum_low=args.wavenum_low,
			wavenum_high=args.wavenum_high,
			normalize=args.normalize,
		),
	)
	opu_anal.run_hca(metric=args.metric, cutoff=args.cutoff_threshold,
		opu_min_size=args.opu_min_size)
	opu_anal.save_hca_labels(args.hca_labels)
	opu_anal.plot_hca(args.hca_plot, dpi=args.dpi)
	opu_anal.plot_opu_abundance_stackbar(args.abund_plot, dpi=args.dpi)
	opu_anal.rank_features(args.feature_score)
	opu_anal.plot_opu_feature_score(args.feature_score_plot, dpi=args.dpi)
	
	return


if __name__ == "__main__":
	main()
