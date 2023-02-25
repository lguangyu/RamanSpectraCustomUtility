#!/usr/bin/env python3

import argparse
import sys

# custom lib
import opu_analysis_lib as oal


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type=str, nargs="?", default="-",
		help="input dataset config json")
	ap.add_argument("--verbose", "-v", action="store_true",
		help="increase verbosity [off]")
	ap.add_argument("--delimiter", "-d", type=str, default="\t",
		metavar="char",
		help="delimiter in text-based input and output [<tab>]")
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
		help="normalize method after loading/binning/filtering dataset [%s]"
			% oal.SpectraDataset.norm_meth.default_key)

	ag = ap.add_argument_group("HCA analysis")
	ag.add_argument("--metric", "-m", type=str,
		default=oal.OPUAnalysis.metric_reg.default_key,
		choices=oal.OPUAnalysis.metric_reg.list_keys(),
		help="distance metric used in HCA [%s]"
			% oal.OPUAnalysis.metric_reg.default_key)
	ag.add_argument("--cutoff-threshold", "-t", default=0.7,
		type=oal.OPUAnalysis.cutoff_opt_reg.argparse_type,
		metavar=("|").join(["float"]
			+ oal.OPUAnalysis.cutoff_opt_reg.list_keys()),
		help="OPU clustering cutoff threshold [0.7]")
	ag.add_argument("--max-n-opus", "-M", type=oal.util.NonNegInt, default=0,
		metavar="int",
		help="maximum number of top-sized clusters to be reported as OPU, 0 "
			"means no limitation [0]")
	ag.add_argument("--opu-min-size", "-s", type=str, default="0",
		metavar="float|int",
		help="minimal spectral count in an HCA cluster to be reported as an OPU"
			", 0 means report all; accepts int (plain size) or float between "
			"0-1 (fraction w.r.t total number of spectra analyzed) [0]")
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
	ag.add_argument("--abund-stackbar-plot", type=str,
		metavar="png",
		help="if set, plot abundance stackbar to this image file [no]")
	ap.add_argument("--abund-biplot-method", type=str,
		default=oal.OPUAnalysis.biplot_meth_reg.default_key,
		choices=oal.OPUAnalysis.biplot_meth_reg.list_keys(),
		help="biplot method [%s]" % oal.OPUAnalysis.biplot_meth_reg.default_key)
	ag.add_argument("--abund-biplot", type=str,
		metavar="png",
		help="if set, plot abundance biplot to this image file [no]")

	ag = ap.add_argument_group("feature score analysis")
	ag.add_argument("--feature-rank-method", "-R", type=str,
		default=oal.OPUAnalysis.score_meth.default_key,
		choices=oal.OPUAnalysis.score_meth.list_keys(),
		help="feature ranking (scoring) method [%s]"
			% oal.OPUAnalysis.score_meth.default_key)
	ag.add_argument("--feature-rank-table", type=str,
		metavar="tsv",
		help="if set, write feature index table into this file [no]")
	ag.add_argument("--feature-rank-plot", type=str,
		metavar="png",
		help="if set, plot feature rank to this image file [no]")

	# parse and refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	return args


def main():
	args = get_args()
	# load dataset config as json, then read spectra data based on the config,
	# then do preprocessing to make sure that they can be analyzed together
	# preprocessing parameters are passed as 'reconcile_param' dict
	opu_anal = oal.OPUAnalysis.from_config_json(args.input,
		reconcile_param=dict(
			delimiter=args.delimiter,
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
	opu_anal.save_opu_labels(args.opu_labels, delimiter=args.delimiter)
	opu_anal.save_opu_collections(args.opu_collection_prefix)
	opu_anal.plot_opu_hca(plot_to=args.opu_hca_plot, dpi=args.dpi)

	# run opu abundance analysis and plot
	opu_anal.plot_opu_abundance_stackbar(plot_to=args.abund_stackbar_plot,
		dpi=args.dpi)
	opu_anal.plot_opu_abundance_biplot(plot_to=args.abund_biplot,
		method=args.abund_biplot_method, dpi=args.dpi)

	# run feature ranking analysis and save results
	opu_anal.rank_features(args.feature_rank_method)
	opu_anal.save_opu_feature_rank_table(args.feature_rank_table,
		delimiter=args.delimiter)
	opu_anal.plot_opu_feature_rank(plot_to=args.feature_rank_plot,
		dpi=args.dpi)
	return


if __name__ == "__main__":
	main()
