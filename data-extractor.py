from pathlib import Path
import numpy as np
import uproot
import os


class ExtractDataFromROOT:
  _syst_trees_to_extract = [
    # 'sipion_syst',
    # 'siproton_syst',
    # 'flux_syst',
    'all_syst',
    # 'bfield_syst',
    # 'momscale_syst',
    # 'momresol_syst',
    # 'tpcpid_syst',
    # 'ecal_emresol_syst',
    # 'ecal_emscale_syst',
    # 'tof_resol_syst',
    # 'chargeideff_syst',
    # 'tpctrackeff_syst',
    # 'fgdtrackeff_syst',
    # 'fgdhybridtrackeff_syst',
    # 'tpcfgdmatcheff_syst',
    # 'pileup_syst',
    # 'fgdmass_syst',
    # 'michel_syst',
    # 'sandmu_syst',
    # 'tpcclustereff_syst',
    # 'tpc_ecal_matcheff_syst',
    # 'ecal_pid_syst',
    # 'ecal_emhippid_syst',
    # 'ecal_photon_pileup_syst',
    # 'fgd2shower_syst',
    # 'nuetpcpileup_syst',
    # 'nuep0dpileup_syst',
    # 'nueecalpileup_syst',
    # 'nueoofv_syst',
    # 'p0d_veto_syst',
  ]

  _data_variables_default_tree = [
    "selelec_mom",
    "selelec_costheta",
    "selelec_nuErecQE",
    "accum_level",
  ]

  _mc_variables_default_tree = [
    "selelec_mom",
    "selelec_true_pdg",
    "selelec_true_ppdg",
    "selelec_true_gppdg",
    "selelec_costheta",
    "selelec_true_mom",
    "selelec_nuErecQE",
    "evt_true_nu_dir_x",
    "evt_true_nu_dir_y",
    "evt_true_nu_dir_z",
    "evt_true_nu_parent_decay_point_x",
    "evt_true_nu_parent_decay_point_y",
    "evt_true_nu_parent_decay_point_z",
    "evt_true_nu_parent_decay_point_t",
    "evt_true_vtx_pos_x",
    "evt_true_vtx_pos_y",
    "evt_true_vtx_pos_z",
    "evt_true_nu_pdg",
    "evt_true_nu_ene",
    "evt_true_nu_parent_pdg",
    "accum_level",

    "NuParentDecPoint",
    "NuParentPDGRaw",
    "TargetPDG",
  ]

  _mc_variables_allsyst_tree = [
    "selelec_costheta",
    "selelec_mom",
    "selelec_nuErecQE",
    "accum_level",
    "selelec_true_pdg",
    "selelec_true_ppdg",
    "selelec_true_gppdg",
    "selelec_true_mom",
    "evt_true_nu_dir_x",
    "evt_true_nu_dir_y",
    "evt_true_nu_dir_z",
    "evt_true_nu_parent_decay_point_x",
    "evt_true_nu_parent_decay_point_y",
    "evt_true_nu_parent_decay_point_z",
    "evt_true_nu_parent_decay_point_t",
    "evt_true_vtx_pos_x",
    "evt_true_vtx_pos_y",
    "evt_true_vtx_pos_z",
    "evt_true_nu_pdg",
    "evt_true_nu_ene",
    "evt_true_nu_parent_pdg",

    "NuParentDecPoint",
    "NuParentPDGRaw",
    "TargetPDG",
  ]

  _mc_variables_truth_tree = [
    "nu_trueE",
    "nu_pdg",
    "nuparent",
    "nu_truedir",
    "NTruePions",
    "NTruePi0",
    "NTrueKaonRhoEta",

    "nu_true_ppdg",
    "nu_true_parent_dec_pt",

    "accum_level",
  ]

  def __init__(self, root_file:str, mode="FHC", isdata=False):
    print(self._data_variables_default_tree)
    print(self._mc_variables_default_tree)
    print(self._mc_variables_truth_tree)
    print(self._mc_variables_allsyst_tree)
    print(self._syst_trees_to_extract)

    self.mode = mode
    self.isdata = isdata

    self.root_file_path = Path(root_file)

    self.data_default_tree_dict = dict()
    self.mc_default_tree_dict = dict()
    self.mc_allsyst_tree_dict = dict()
    self.mc_truth_tree_dict = dict()

    # self.cutnum = 17
    self.cutnum = 11

    # self.data_default_tree_dict_fgd1 = dict()
    # self.mc_default_tree_dict_fgd1 = dict()
    # self.mc_allsyst_tree_dict_fgd1 = dict()
    # self.mc_truth_tree_dict_fgd1 = dict()
    # self.data_default_tree_dict_fgd2 = dict()
    # self.mc_default_tree_dict_fgd2 = dict()
    # self.mc_allsyst_tree_dict_fgd2 = dict()
    # self.mc_truth_tree_dict_fgd2 = dict()

    self.read_file()


  def read_file(self,):
    with uproot.open(self.root_file_path) as f:
      if self.isdata:
        self.extract_array_from_root_tree(f, "default", self._data_variables_default_tree, self.data_default_tree_dict)
      else:
        self.extract_array_from_root_tree(f, "default", self._mc_variables_default_tree, self.mc_default_tree_dict)
        self.extract_array_from_root_tree(f, "truth", self._mc_variables_truth_tree, self.mc_truth_tree_dict)
        for treename in self._syst_trees_to_extract:
          self.extract_array_from_root_tree(f, treename, self._mc_variables_allsyst_tree, self.mc_allsyst_tree_dict)

  def extract_array_from_root_tree(self, uproot_file, treename, variables_list, outdict):
    if not treename in uproot_file: return ### not tested

    # print("!!!", treename)


    prefix = "data_" if self.isdata else "mc_"
    prefix = f"{self.mode.lower()}_{prefix}"

    accum_level_arr = None

    d = dict()

    tree = uproot_file[treename]
    for varname in tree.keys():
      if varname in variables_list:
        np_arr = tree[varname].array(library="np")
        if treename == "default": print(varname, np_arr, np_arr.shape)

        np_arr = self.preprocess_data(np_arr, treename, varname)

        if varname == "accum_level":
          accum_level_arr = np_arr

        outkey = prefix + treename + "__" + varname

        # outdict[outkey] = np_arr
        d[outkey] = np_arr

    if treename in self._syst_trees_to_extract:
      # pass
      outdict.update(d)

      accum_level_arr_fgd1sel = accum_level_arr[:, :, 0]
      accum_level_arr_fgd2sel = accum_level_arr[:, :, 1]
      ### print(accum_level_arr_fgd1sel, accum_level_arr_fgd1sel.shape)

      # fgd1_neg_cond = np.where(accum_level_arr_fgd1sel<=17)
      # fgd2_neg_cond = np.where(accum_level_arr_fgd2sel<=17)
      # fgd1_pos_cond = np.where(accum_level_arr_fgd1sel>17)
      # fgd2_pos_cond = np.where(accum_level_arr_fgd2sel>17)
      # print(fgd1_pos_cond[0].shape, fgd1_pos_cond[1].shape)
      # print(fgd1_neg_cond[0].shape, fgd1_neg_cond[1].shape)
      # print(fgd1_pos_cond[0].shape, fgd1_pos_cond[1].shape)
      # print(fgd1_pos_cond)

      N_toys = accum_level_arr_fgd1sel.shape[1]
      ### print("N_toys", N_toys)

      ### print("-"*50)
      for k, v in d.items():
        ### print(">>>", k, v, v.shape)
        if "accum_level" in k: continue

        if len(v.shape) == 2 and v.shape[1] == N_toys:
          v_fgd1 = np.zeros(N_toys, dtype=object)
          v_fgd2 = np.zeros(N_toys, dtype=object)
          ### print("===")
          for i in range(N_toys):
            v_itoy = v[:, i]

            al_fgd1sel = accum_level_arr_fgd1sel[:, i]
            al_fgd2sel = accum_level_arr_fgd2sel[:, i]

            # v_itoy_fgd1 = v_itoy[al_fgd1sel > 17]
            # v_itoy_fgd2 = v_itoy[al_fgd2sel > 17]
            v_itoy_fgd1 = v_itoy[al_fgd1sel > self.cutnum-1]
            v_itoy_fgd2 = v_itoy[al_fgd2sel > self.cutnum-1]

            # print(v_itoy_fgd1.shape)
            # print(v_itoy_fgd2.shape)

            v_fgd1[i] = v_itoy_fgd1
            v_fgd2[i] = v_itoy_fgd2

          ### print(v_fgd1)
          ### print(v_fgd2)

        else:
          pass


        # v_arr_obj_fgd1 = None
        # v_arr_obj_fgd2 = None
        # if len(v.shape) == 2 and v.shape[1] != 4:
        #   # v_arr_obj_fgd1 = np.zeros(N_toys, dtype=object)
        #   # v_arr_obj_fgd2 = np.zeros(N_toys, dtype=object)
        #   # v_fgd1 = v[fgd1_pos_cond[0], fgd1_pos_cond[1]]
        #   # v_fgd2 = v[fgd2_pos_cond[0], fgd2_pos_cond[1]]
        #   # print(v_fgd1, v_fgd1.shape)
        #   # print(v_fgd2, v_fgd2.shape)
        #   for i in range(N_toys):
        #     v_itoy = v[i]

        # elif len(v.shape) == 1 or v.shape[1] == 4:
        #   v_fgd1 = v[fgd1_pos_cond[0]]
        #   v_fgd2 = v[fgd2_pos_cond[0]]
        #   print(v_fgd1, v_fgd1.shape)
        #   print(v_fgd2, v_fgd2.shape)

        # if len(v.shape) == 1:

        #   # fgd1_cond = np.where()
        #   # v_fgd1 = v[fgd1_pos_cond[0]]
        #   # v_fgd2 = v[fgd2_pos_cond[0]]
        #   # print(v_fgd1, v_fgd1.shape)
        #   # print(v_fgd2, v_fgd2.shape)

        '''
        accum_level_arr_fgd1sel = accum_level_arr[:, :, 0]
        accum_level_arr_fgd2sel = accum_level_arr[:, :, 1]
        print("-"*50)
        print(accum_level_arr_fgd1sel, accum_level_arr_fgd1sel.shape)
        print(accum_level_arr_fgd2sel, accum_level_arr_fgd2sel.shape)

        print(accum_level_arr_fgd1sel[100])

        fgd1_cond = np.where(accum_level_arr_fgd1sel<=17)
        fgd2_cond = np.where(accum_level_arr_fgd2sel<=17)
        fgd1_pos_cond = np.where(accum_level_arr_fgd1sel>17)
        fgd2_pos_cond = np.where(accum_level_arr_fgd2sel>17)
        print(fgd1_cond)
        print(fgd2_cond)

        fhc_mc_all_syst__selelec_mom = d[f"fhc_mc_{treename}__selelec_mom"]
        # arr = fhc_mc_all_syst__selelec_mom[fgd1_cond[0], fgd1_cond[1]]
        # print(arr, arr.shape)

        fhc_mc_all_syst__selelec_mom_fgd1 = fhc_mc_all_syst__selelec_mom.copy()
        fhc_mc_all_syst__selelec_mom_fgd2 = fhc_mc_all_syst__selelec_mom.copy()

        fhc_mc_all_syst__selelec_mom_fgd1[fgd1_cond[0], fgd1_cond[1]] = -1.0e10
        fhc_mc_all_syst__selelec_mom_fgd2[fgd2_cond[0], fgd2_cond[1]] = -1.0e10

        # for k, v in d.items():
        #   print(k, v.shape, v)
        print(fhc_mc_all_syst__selelec_mom_fgd1, fhc_mc_all_syst__selelec_mom_fgd1.shape)
        print(fhc_mc_all_syst__selelec_mom_fgd2, fhc_mc_all_syst__selelec_mom_fgd2.shape)

        # fgd1_syst = np.array([ arr for arr in accum_level_arr_fgd1sel if np.all(arr) == 18 ])
        # fgd2_syst = np.array([ arr for arr in accum_level_arr_fgd2sel if np.all(arr) == 18 ])

        # print(">"*10)
        # print(accum_level_arr_fgd1sel, accum_level_arr_fgd1sel.shape)
        # print(accum_level_arr_fgd2sel, accum_level_arr_fgd2sel.shape)

        print("-"*50)
        for k, v in d.items():
          if "accum_level" in k:
            accum_level_fgd1 = accum_level_arr_fgd1sel.copy()
            accum_level_fgd2 = accum_level_arr_fgd2sel.copy()
            accum_level_fgd1[fgd1_cond[0], fgd1_cond[1]] = -100
            accum_level_fgd2[fgd2_cond[0], fgd2_cond[1]] = -100
            # accum_level_fgd1 = accum_level_arr_fgd1sel[fgd1_pos_cond[0], fgd1_pos_cond[1]]
            # accum_level_fgd2 = accum_level_arr_fgd2sel[fgd2_pos_cond[0], fgd2_pos_cond[1]]
            print(accum_level_fgd1, accum_level_fgd1.shape)
            print(accum_level_fgd2, accum_level_fgd2.shape)
          else:
            if len(v.shape) == 2:
              v_fgd1 = v.copy()
              v_fgd2 = v.copy()

              v_fgd1[fgd1_cond[0], fgd1_cond[1]] = -1.0e10
              v_fgd2[fgd2_cond[0], fgd2_cond[1]] = -1.0e10
        #'''

    else:
      # fgd1sel = np.where(accum_level_arr[:, 0] == 18)[0]
      # fgd2sel = np.where(accum_level_arr[:, 1] == 18)[0]
      fgd1sel = np.where(accum_level_arr[:, 0] == self.cutnum)[0]
      fgd2sel = np.where(accum_level_arr[:, 1] == self.cutnum)[0]

      for varname, vararr in d.items():
        isaccumlevel = True if "accum_level" in varname else False
        vararr_fgd1 = vararr[fgd1sel]
        vararr_fgd2 = vararr[fgd2sel]
        if isaccumlevel:# and treename not in self._syst_trees_to_extract:
          vararr_fgd1 = np.array([a[0] for a in vararr_fgd1])
          vararr_fgd2 = np.array([a[0] for a in vararr_fgd2])
        outdict[f"{varname}_fgd1"] = vararr_fgd1
        outdict[f"{varname}_fgd2"] = vararr_fgd2

    '''
    if treename in self._syst_trees_to_extract:
      fgd1sel = np.where(accum_level_arr[:, :, 0] == 18)[0]
      fgd2sel = np.where(accum_level_arr[:, :, 1] == 18)[0]
    else:
      fgd1sel = np.where(accum_level_arr[:, 0] == 18)[0]
      fgd2sel = np.where(accum_level_arr[:, 1] == 18)[0]

    for varname, vararr in d.items():
      isaccumlevel = True if "accum_level" in varname else False
      vararr_fgd1 = vararr[fgd1sel]
      vararr_fgd2 = vararr[fgd2sel]
      if isaccumlevel:# and treename not in self._syst_trees_to_extract:
        vararr_fgd1 = np.array([a[0] for a in vararr_fgd1])
        vararr_fgd2 = np.array([a[0] for a in vararr_fgd2])
      outdict[f"{varname}_fgd1"] = vararr_fgd1
      outdict[f"{varname}_fgd2"] = vararr_fgd2
      #'''

    # print(">>>", treename, "<<<")
    # print(fgd1sel)
    # print(fgd2sel)

  def preprocess_data(self, np_arr, treename, varname):
    if np_arr.dtype == "object":
      np_arr = np.array([a for a in np_arr])

    if treename == "default" and varname in ["accum_level", "selelec_mom", "selelec_costheta", "selelec_nuErecQE", "selelec_true_mom"]:
      np_arr = np.array([a[0] for a in np_arr])

    return np_arr

  def process_systematics_tree(self, treename):
    pass

  def split_by_fgd_selection(self, np_arr, accum_level, isaccumlevel=False):
    """
      FGD 1/2 selection passed if corresponding value of cut is 18
    """
    pass


  def print(self):
    print("\n"*5)
    print("="*100)
    print(">>> data_default_tree_dict <<<")
    for k,v in self.data_default_tree_dict.items():
      print(k, v, v.dtype, v.shape)
    print("="*100)
    print(">>> mc_default_tree_dict <<<")
    for k,v in self.mc_default_tree_dict.items():
      print(k, v, v.dtype, v.shape)
    print("="*100)
    print(">>> mc_allsyst_tree_dict <<<")
    for k,v in self.mc_allsyst_tree_dict.items():
      print(k, v, v.dtype, v.shape)
    print("="*100)
    print(">>> mc_truth_tree_dict <<<")
    for k,v in self.mc_truth_tree_dict.items():
      print(k, v, v.dtype, v.shape)


  def split_datadict_by_fgd_selection(self, indict):
    dict_fgd1 = dict()
    dict_fgd2 = dict()
    for key in indict:
      if "fgd1" in key: dict_fgd1[key] = indict[key]
      if "fgd2" in key: dict_fgd2[key] = indict[key]

    return dict_fgd1, dict_fgd2

  def save_dict_to_file(self, indict, name, whichfgd, outpath):
    if not os.path.exists(outpath):
      os.system(f"mkdir -p {outpath}")

    prefix = "data" if self.isdata else "mc"
    ### for gamma
    prefix = f"gamma-{prefix}"
    fname = f"{prefix}-{self.mode}-{name}-{whichfgd}.npz"
    filepath = f"{outpath}/{fname}"
    np.savez_compressed(
      filepath,
      preprocessed_dict=indict
    )

  def save(self, outpath):
    pass
    if self.data_default_tree_dict:
      (
        data_default_tree_dict_fgd1,
        data_default_tree_dict_fgd2
      ) = self.split_datadict_by_fgd_selection(self.data_default_tree_dict)
      self.save_dict_to_file(data_default_tree_dict_fgd1, "default_tree", "fgd1", outpath)
      self.save_dict_to_file(data_default_tree_dict_fgd2, "default_tree", "fgd2", outpath)

    if self.mc_default_tree_dict:
      (
        mc_default_tree_dict_fgd1,
        mc_default_tree_dict_fgd2
      ) = self.split_datadict_by_fgd_selection(self.mc_default_tree_dict)
      self.save_dict_to_file(mc_default_tree_dict_fgd1, "default_tree", "fgd1", outpath)
      self.save_dict_to_file(mc_default_tree_dict_fgd2, "default_tree", "fgd2", outpath)

    if self.mc_allsyst_tree_dict:
      self.save_dict_to_file(self.mc_allsyst_tree_dict, "allsyst_tree", "fgd12", outpath)

    if self.mc_truth_tree_dict:
      (
        mc_truth_tree_dict_fgd1,
        mc_truth_tree_dict_fgd2
      ) = self.split_datadict_by_fgd_selection(self.mc_truth_tree_dict)
      self.save_dict_to_file(mc_truth_tree_dict_fgd1, "truth_tree", "fgd1", outpath)
      self.save_dict_to_file(mc_truth_tree_dict_fgd2, "truth_tree", "fgd2", outpath)


if __name__ == "__main__":
  ### nue
  # datafile = "/t2k2/users/shvartsman/PhD-work/Analysis/nueCCAnalysis/tstAnasternu049/root-files/junk.root"
  # mcfile = "/t2k2/users/shvartsman/PhD-work/Analysis/nueCCAnalysis/tstAnasternu059/root-files/junk.root"

  ### gamma
  datafile = "/t2k2/users/shvartsman/PhD-work/Analysis/gammaAnalysis/tstAnasternu203d/root-files/junk.root"
  mcfile = "/t2k2/users/shvartsman/PhD-work/Analysis/gammaAnalysis/tstAnasternu203m/root-files/junk.root"

  ff = ExtractDataFromROOT(mcfile, mode="FHC", isdata=False)
  ff.print()
  # ff.save("/t2k/users/shvartsman/HEP-soft/T2KSoft/myscripts/fitting/preprocessed-gamma")

  # ff = ExtractDataFromROOT(datafile, mode="FHC", isdata=True)
  # ff.print()
  # # ff.save("/t2k/users/shvartsman/HEP-soft/T2KSoft/myscripts/fitting/preprocessed-gamma")


  # ff.print()
  # ff.save("/t2k/users/shvartsman/HEP-soft/T2KSoft/myscripts/fitting/preprocessed")

  # ff.process_systematics_tree("all_syst")
  # print(ff.mc_allsyst_tree_dict)
