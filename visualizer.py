import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm


MC2DATA_RATIO = 16.4536 # needed to be extracted from root file


class Latex:
  dm2 = R"\Delta m^2"
  sin22t = R"sin^2 2\theta"

  @staticmethod
  def generate_negloglkh_eqn():
    func_expr = f"({Latex.dm2}, {Latex.sin22t})"
    lnlog_expr = fR"\ln L{func_expr}"
    sum_expr = R"\sum_{i=1}^N"
    n_exp_i_expr = R"n^{exp}_i"f"{func_expr}"
    n_obs_i_expr = R"n^{obs}_i"
    return (
      fR"$-2 {func_expr} = 2 {sum_expr} \left("
      f"{n_exp_i_expr} - {n_obs_i_expr} + {n_obs_i_expr} "R"\ln \frac{"f"{n_obs_i_expr}"R"}{"f"{n_exp_i_expr}"R"}\right),$"
    )


class Visualizer:
  neglkheqn = R"$-2 \ln L(\Delta m^2, sin^2 2\theta) = 2 \sum_{i=1}^N \left(n^{exp}_i(\Delta m^2, sin^2 2\theta) - n^{obs}_i + n^{obs}_i \ln \frac{n^{obs}_i}{n^{ex
  explanation = (
    R"$n^{exp}_i(\Delta m^2, sin^2 2\theta)$ -- expected number of events in bin i"
    "\n"
    R"$n^{obs}_i$ -- observed number of events (nominal MC)"
  )

  def __init__(self,):
    pass

  @staticmethod
  def draw_althyplkh_wrt_eventsdistr(
        _dm2:float, _sin22t:float, dm2_model_arr, sin22theta_model_arr,
        likelihood_Halt, reco_nuErecQE_GeV, reco_nuErecQE_survived_GeV, binning_GeV
  ):
    print(np.where(dm2_model_arr == _dm2))
    print(np.where(sin22theta_model_arr == _sin22t))
    i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
    j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]

    lkh_H1_spec = likelihood_Halt[i_dm2, j_sin22t]
    lkh_H1_spec_mean = np.mean(lkh_H1_spec)
    se = reco_nuErecQE_survived_GeV[i_dm2, j_sin22t]

    label = fR"$\Delta m^2 = {_dm2} eV^2$, $sin^2 2\theta = {_sin22t}$"

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    ## draw neg log likelihood histogram over experiments
    axs[0].hist(lkh_H1_spec, bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    axs[0].axvline(lkh_H1_spec_mean, color="red", label="likelihood mean")
    axs[0].legend(prop={"size": 18})
    axs[0].set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=16)
    axs[0].set_title(f"likelihood for oscillation hypothesis with {label}", fontsize=20)

    ## draw corresponding events histogram for osc and non-osc hypothsis
    axs[1].hist(reco_nuErecQE_GeV, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
      histtype="step", color="black", linewidth=2, label="nominal MC events distribution")
    axs[1].hist(se, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
      histtype="step", color="red", linewidth=2, label=f"MC events distribution for {label}")
    axs[1].legend(prop={"size": 18})
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axs[1].set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=16)
    axs[1].set_title(f"Survived events distribution", fontsize=20)

    fig.tight_layout()
    return fig

  @staticmethod
  def draw_althyplkh_wrt_eventsdistr_v2(
        _dm2:float, _sin22t:float, dm2_model_arr, sin22theta_model_arr,
        likelihood_Halt,
        reco_nuErecQE_GeV,
        reco_nuErecQE_survived_GeV,
        reco_nuErecQE_survived_GeV_sel, # reco_nuErecQE_survived_GeV, but with fixed dm2, sin22t
        _dm2_fixed,
        _sin22t_fixed,
        binning_GeV
  ):
    print(np.where(dm2_model_arr == _dm2))
    print(np.where(sin22theta_model_arr == _sin22t))
    i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
    j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]

    lkh_H1_spec = likelihood_Halt[i_dm2, j_sin22t]
    lkh_H1_spec_mean = np.mean(lkh_H1_spec)
    se = reco_nuErecQE_survived_GeV[i_dm2, j_sin22t]

    label = fR"$\Delta m^2 = {_dm2} eV^2$, $sin^2 2\theta = {_sin22t}$"

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      label_fixed = fR"$\Delta m^2 = {_dm2_fixed} eV^2$, $sin^2 2\theta = {_sin22t_fixed}$"

    gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[4, 1])

    fig = plt.figure(figsize=(30, 15))
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 1], sharex=ax2)

    ## draw neg log likelihood histogram over experiments
    ax1.hist(lkh_H1_spec, bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    ax1.axvline(lkh_H1_spec_mean, color="red", label="likelihood mean")
    ax1.legend(prop={"size": 20})
    ax1.set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=20)
    ax1.set_title(f"likelihood for oscillation hypothesis with {label}", fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ## draw corresponding events histogram for osc and non-osc hypothsis
    # nom_mc_h = ax2.hist(reco_nuErecQE_GeV, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
    #   histtype="step", color="black", linewidth=2, label="nominal MC events distribution")
    # osc_h = ax2.hist(se, bins=binning_GeV, weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
    #   histtype="step", color="red", linewidth=2, label=f"MC events distribution for {label}")
    nom_mc_h = ax2.hist(
      reco_nuErecQE_GeV,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
      histtype="step",
      color="black",
      linewidth=2,
      label="nominal MC events distribution"
    )
    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      nom_mc_h_fix = ax2.hist(
        reco_nuErecQE_survived_GeV_sel,
        bins=binning_GeV,
        weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_survived_GeV_sel.shape[0],
        histtype="step",
        color="green",
        linewidth=2,
        label=f"MC events distribution for fixed {label_fixed}"
      )
    osc_h = ax2.hist(
      se,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
      histtype="step",
      color="red",
      linewidth=2,
      label=f"MC events distribution for {label}"
    )
    ax2.legend(prop={"size": 20})
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax2.set_ylabel(f"N events per bin", fontsize=20)
    ax2.set_title(f"Survived events distribution", fontsize=24)

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      # ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h_fix[0])
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, nom_mc_h_fix[0] / nom_mc_h[0])
    else:
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h[0])
    # ax3.bar(binning_GeV[:-1], osc_h[0] / nom_mc_h[0], width=1.0)
    ax3.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_ylabel(R"$N_{osc} / N_{nominal}$ per bin", fontsize=20)

    fig.tight_layout()
    return fig

  @staticmethod
  def draw_althyplkh_wrt_eventsdistr_v3(
        _dm2:float, _sin22t:float,
        dm2_model_arr, sin22theta_model_arr,
        survival_probability_arr,
        likelihood_Halt,
        reco_nuErecQE_GeV,
        reco_nuErecQE_survived_GeV,
        reco_nuErecQE_survived_GeV_sel, # reco_nuErecQE_survived_GeV, but with fixed dm2, sin22t
        _dm2_fixed,
        _sin22t_fixed,
        binning_GeV
  ):
    # print(np.where(dm2_model_arr == _dm2))
    # print(np.where(sin22theta_model_arr == _sin22t))
    i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
    j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]

    lkh_H1_spec = likelihood_Halt[i_dm2, j_sin22t]
    lkh_H1_spec_mean = np.mean(lkh_H1_spec)
    # lkh_H1_spec_mean = np.min(lkh_H1_spec)
    se = reco_nuErecQE_survived_GeV[i_dm2, j_sin22t]

    label = fR"$\Delta m^2 = {_dm2} eV^2$, $sin^2 2\theta = {_sin22t}$"

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      label_fixed = fR"$\Delta m^2 = {_dm2_fixed} eV^2$, $sin^2 2\theta = {_sin22t_fixed}$"

    gs = GridSpec(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[4, 1, 5])

    fig = plt.figure(figsize=(30, 15))
    ax1 = plt.subplot(gs[0:2, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 1], sharex=ax2)
    ax4 = plt.subplot(gs[2, 1])

    ## draw neg log likelihood histogram over experiments
    ax1.hist(lkh_H1_spec, bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    ax1.axvline(lkh_H1_spec_mean, color="red", label="likelihood mean")
    ax1.legend(prop={"size": 20})
    ax1.set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=20)
    ax1.set_title(f"likelihood for oscillation hypothesis with {label}", fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ## draw corresponding events histogram for osc and non-osc hypothsis
    nom_mc_h = ax2.hist(
      reco_nuErecQE_GeV,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_GeV.shape[0],
      histtype="step",
      color="black",
      linewidth=2,
      label="nominal MC events distribution"
    )
    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      nom_mc_h_fix = ax2.hist(
        reco_nuErecQE_survived_GeV_sel,
        bins=binning_GeV,
        weights=[1.0 / MC2DATA_RATIO]*reco_nuErecQE_survived_GeV_sel.shape[0],
        histtype="step",
        color="green",
        linewidth=2,
        label=f"MC events distribution for fixed {label_fixed}"
      )
    osc_h = ax2.hist(
      se,
      bins=binning_GeV,
      weights=[1.0 / MC2DATA_RATIO]*se.shape[0],
      histtype="step",
      color="red",
      linewidth=2,
      label=f"MC events distribution for {label}"
    )
    ax2.legend(prop={"size": 20})
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax2.set_ylabel(f"N events per bin", fontsize=20)
    ax2.set_title(f"Survived events distribution", fontsize=24)

    if reco_nuErecQE_survived_GeV_sel is not None and _dm2_fixed and _sin22t_fixed:
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h_fix[0])
    else:
      ax3.scatter((binning_GeV[:-1] + binning_GeV[1:])/2.0, osc_h[0] / nom_mc_h[0])
    # ax3.bar(binning_GeV[:-1], osc_h[0] / nom_mc_h[0], width=1.0)
    ax3.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_ylabel(R"$N_{osc} / N_{nominal}$ per bin", fontsize=20)

    x = reco_nuErecQE_GeV
    y = survival_probability_arr[i_dm2, j_sin22t]
    x = x[y!= -1]
    y = y[y!= -1]
    # ax4.scatter(x, y)
    # ax4.set_xlim(0.0, 5.0)
    # ax4.set_ylim(0.0, 1.0)
    # ax4.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    # ax4.tick_params(axis='both', which='major', labelsize=15)
    # ax4.set_ylabel(R"Survival probability", fontsize=20)
    # ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # # m = ax4.hist2d(x, y, bins=[binning_GeV, np.arange(0.0, 1.0, 10)])
    m = ax4.hist2d(x, y, bins=10, range=[[0.0, 5.0], [0.0, 1.0]], norm=LogNorm())
    ax4.set_xlabel(f"Reconstructed neutrino energy, GeV", fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.set_ylabel(R"Survival probability", fontsize=20)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    fig.colorbar(mappable=m[3], ax=ax4)

    fig.tight_layout()
    return fig


  @staticmethod
  def draw_with_differ_params(
        _dm2_lst:list, _sin22t_lst:list, dm2_model_arr, sin22theta_model_arr,
        lkh_null_hypo, lkh_alt_hypo, colors = ["red", "blue", "green"]
  ):
    requested_indices = []
    for _dm2, _sin22t in zip(_dm2_lst, _sin22t_lst):
      i_dm2 = np.where(dm2_model_arr == _dm2)[0][0]
      j_sin22t = np.where(sin22theta_model_arr == _sin22t)[0][0]
      requested_indices.append((i_dm2, j_sin22t))

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    x = np.arange(0, 250, 0.01)

    # ax_lkh_h1.hist(lkh_null_hypo, bins=100, histtype="step", color="black", linewidth=2, density=True)
    # label = fR"$\Delta m^2 = {dm2_model_arr[10]}$, $sin^2 2\theta = {sin22theta_model_arr[10]}$"
    # ax_lkh_h1.hist(lkh_alt_hypo[10, 20], bins=100, histtype="step", color="red", linewidth=2, density=True, label=label)
    # label = fR"$\Delta m^2 = {dm2_model_arr[40]}$, $sin^2 2\theta = {sin22theta_model_arr[50]}$"
    # ax_lkh_h1.hist(lkh_alt_hypo[40, 50], bins=100, histtype="step", color="blue", linewidth=2, density=True, label=label)
    # label = fR"$\Delta m^2 = {dm2_model_arr[40]}$, $sin^2 2\theta = {sin22theta_model_arr[80]}$"
    # ax_lkh_h1.hist(lkh_alt_hypo[40, 80], bins=100, histtype="step", color="green", linewidth=2, density=True, label=label)

    for k, params_idxs in enumerate(requested_indices):
      i_dm2, j_sin22t = params_idxs
      color = colors[k % len(requested_indices)]
      label = fR"$\Delta m^2 = {dm2_model_arr[i_dm2]}$, $sin^2 2\theta = {sin22theta_model_arr[j_sin22t]}$"
      ax.hist(lkh_alt_hypo[i_dm2, j_sin22t], bins=100, histtype="step", color=color, linewidth=2, density=True, label=label)

    ax.plot(x, chi2.pdf(x, df=34), c="cyan")
    ax.legend(prop={"size": 15})
    ax.set_xlabel(f"{Visualizer.neglkheqn}\n{Visualizer.explanation}", fontsize=16)
    ax.set_title("likelihood for alternative hypothesis", fontsize=20)

    return fig
