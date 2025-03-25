#!/usr/bin/env python3
"""
This code is based on HypoInvPy by xtyangpsp (Xiaotao Yang).
Original repository: https://github.com/xtyangpsp/HypoInvPy.git
This code has been modified.
"""
import os
import glob
import sys
import argparse
import numpy as np
from pathlib import Path
from hypoinvpy import core as hc
from hypoinvpy import utils

def setup_environment(date_dir, vmodel_dir):
    """Prepare the working environment"""
    date_dir = Path(date_dir).resolve()
    vmodel_dir = Path(vmodel_dir).resolve()
    
    date_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(date_dir)
    
    input_dir = date_dir/"input"
    input_dir.mkdir(exist_ok=True)
    
    # Limpieza robusta de enlaces/archivos existentes
    for vfile in ['velo_p_eg.cre', 'velo_s_eg.cre']:
        link_path = input_dir/vfile
        target_path = vmodel_dir/vfile
        
        # 1. Eliminar cualquier elemento existente
        if link_path.exists():
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink(missing_ok=True)
            elif link_path.is_dir():
                import shutil
                shutil.rmtree(link_path)
        
        # 2. Verificar que el modelo fuente existe
        if not target_path.exists():
            raise FileNotFoundError(f"Modelo de velocidad faltante: {target_path}")
        
        # 3. Crear enlace simbólico
        link_path.symlink_to(target_path)
        print(f"Enlace creado: {link_path} -> {target_path}")

def main(args):
    try:
        setup_environment(args.date_dir, args.vmodel_dir)
        print(f"\n Processing in: {Path(args.date_dir).resolve()}")

        # 1. Station conversion
        station_json = Path("stations.json")
        station_ready = Path("input/station_list_ready.sta")
        hc.reformat_stainfo(str(station_json), str(station_ready), 
                           informat='json-gamma', ignore_component=False)

        # 2. Picks conversion
        phase_file = Path("input/pavlof_phases.phs")
        utils.conv_gamma("gamma_catalog.csv", "gamma_picks.csv",
                        outfile=str(phase_file), default_component='Z', v=True)

        # 3. HypoInverse configuration
        cfg = hc.HypoInvConfig(
            phase_file = str(phase_file),
            station_file = str(station_ready),
            pmodel = "input/velo_p_eg.cre",
            smodel = "input/velo_s_eg.cre",
            min_nsta = 4,
            lat_code = 'N',
            lon_code = 'W',
            ztrlist = np.arange(0, 20, 1),
            hypoinv_bin = args.hypo_bin
        )

        # 4. Generate .hyp files
        parfiles = hc.generate_parfile(cfg, pardir="input", outdir="output", magline='MAG')

        # 5. Run HypoInverse
        hc.run_hypoinv(parfiles)

        # 6. Process results
        run_tag = cfg.run_tag
        summary_files = glob.glob(f"output/{run_tag}-*.sum")
        hc.merge_summary(summary_files, 
                        f"output/{run_tag}_good.csv",
                        f"output/{run_tag}_bad.csv",
                        'N', 'W', mag_dict="gamma_catalog.csv")

        print(f"\n Results saved in: {Path(args.date_dir).resolve()}/output")
        print(f"- Good events: {run_tag}_good.csv")
        print(f"- Bad events: {run_tag}_bad.csv")

    except Exception as e:
        print(f"\n Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_dir', required=True)
    parser.add_argument('--vmodel_dir', required=True)
    parser.add_argument('--hypo_bin', required=True)
    args = parser.parse_args()
    main(args)
