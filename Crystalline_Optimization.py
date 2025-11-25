# ------ CRYSTAL STRUCTURE ANALYSIS ------

# Crystal Structure Analysis
#   1. import crystal structure
#   2. relax with BFGS algorithm 
#   3. calculate relative energy (with a reference structure)
#   4  calculate lattice params and avg bond lengths
#   5. create JSON raw data files#

import warnings
import re

# Suppress noisy warnings from FrechetCellFilter:
#   RuntimeWarning: logm result may be inaccurate, approximate err = 6.2e-13
#       pos[natoms:] = self.logm(pos[natoms:]) * self.exp_cell_factor
_original_showwarning = warnings.showwarning

def suppress_warning(message, category, filename, lineno, file=None, line=None):
    text = str(message)

    # Only touch the specific RuntimeWarning from logm
    if issubclass(category, RuntimeWarning) and "logm result may be inaccurate" in text:
        # Extract the numeric error from "approximate err = X"
        m = re.search(r"approximate err = ([0-9.eE+-]+)", text)
        if m:
            err = float(m.group(1))
            # Suppress if below your threshold
            if err < 1e-10:
                return  # do not print anything

    # Fallback: show all other warnings normally
    _original_showwarning(message, category, filename, lineno, file=file, line=line)

warnings.showwarning = suppress_warning


import time
from pathlib import Path
from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from graph_pes.models import load_model
import json
import torch
#from quippy.potential import Potential
from graph_pes.utils.calculator import GraphPESCalculator

# Ovito for bond length and angle analysis
import ovito
from ovito.modifiers import CreateBondsModifier, BondAnalysisModifier
from ovito.io import import_file
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource

# Imports structures from in_dir and creates dict structures = {file_path: structure}
def import_crystal_structures(in_dir):
    
    in_dir = Path(in_dir)
    structure_paths = []

    for file in in_dir.rglob('*'):
        if file.suffix == ".cif":
            structure_paths.append(file)
        else:
            print(f"Unrecognized file format: {file.suffix}, must be .cif")
            continue
    
    for i in structure_paths:
        i = Path(i)
        print(f"Imported: {i.name} from {in_dir}\n")

    return structure_paths

# Allows loading of graph-pes-mace (.pt) files or .xml files
# skin = 0.1 appears to be fastest due to nearest neighbour caching
def load_model_and_calc(path_to_model):

    path_to_model = Path(path_to_model)

    if path_to_model.suffix == ".pt":
        torch.set_default_dtype(torch.float32)
        model = load_model(path_to_model)
        calc = GraphPESCalculator(model, skin=0.1, device='cpu')

    elif path_to_model.suffix == ".xml":
        print(f"Not compatible with.xml files yet")
        return
        class GAPWrapper:
            def __init__(self, potential):
                self.potential = potential
                self.name = "GAP"

            def get_potential_energy(self, atoms, **kwargs):
                return self.potential.get_potential_energy(atoms)

            def get_forces(self, atoms, **kwargs):
                return self.potential.get_forces(atoms)

            def get_stress(self, atoms=None):
                return self.potential.get_stress(atoms=atoms)

            def __getattr__(self, attr):
                return getattr(self.potential, attr)
        
        potential = path_to_model.read_text()
        raw_gap = Potential("IP GAP", param_str=potential)
        calc = GAPWrapper(raw_gap)
    else:
        print(f"Unrecognized model file extension for {Path(path_to_model).name}",
              f"Must be either .pt (graph-pes-mace) or .xml (quippy)")
        return
    
    return calc

# Ovito Bond angle and length calculations
def get_bond_lengths_and_angles(ase_atoms_object, cutoff, bins):

    # Clear existing pipeline
    for p in list(ovito.scene.pipelines):
        p.remove_from_scene()

    if not ase_atoms_object:
        raise ValueError("No ase atoms object provided to ovito pipeline()")

    data = ase_to_ovito(ase_atoms_object)

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(CreateBondsModifier(cutoff=cutoff))
    pipeline.modifiers.append(BondAnalysisModifier(bins = bins, length_cutoff=cutoff))

    data = pipeline.compute()

    bond_angles_xy  = data.tables["bond-angle-distr"].xy().tolist()
    bond_lengths_xy = data.tables["bond-length-distr"].xy().tolist()

    bond_lengths = {
    "value": [v[0] for v in bond_lengths_xy],
    "count": [v[1] for v in bond_lengths_xy],
    }

    bond_angles = {
        "value": [v[0] for v in bond_angles_xy],
        "count": [v[1] for v in bond_angles_xy],
    }

    # Remove Modifiers
    pipeline.modifiers.pop()
    pipeline.modifiers.pop()

    return bond_lengths, bond_angles

# Relaxes a structure 
# Creates .traj, .cif (final frame), .json
# Returns energy/atom and .json file path
def relax_and_calculate(path_to_structure, path_to_model, fmax, steps,
                         OVERWRITE, traj_out_dir, data_out_dir):

    path_to_structure = Path(path_to_structure)

    # Outpaths
    path_to_model = Path(path_to_model)
    path_to_structure = Path(path_to_structure)

    traj_dir = Path(traj_out_dir) / "Trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    final_traj_dir = Path(traj_out_dir)/ "Final Trajectory Frame"
    final_traj_dir.mkdir(parents=True, exist_ok=True)

    traj_path = traj_dir / f"{path_to_model.stem}_{path_to_structure.stem}.traj"
    final_frame_path = final_traj_dir / f"{path_to_model.stem}_relaxed_{path_to_structure.stem}.cif"
        
    # Load model and calculator
    calc = load_model_and_calc(path_to_model)

    # Load reference structure
    structure = read(path_to_structure)
    structure.calc = calc
    
    # Relax structure
    num_steps="Cannot parse steps without re-running relaxtion"
    
    if not traj_path.exists() or OVERWRITE:

        # Relax reference structure with BFGS (allowing for cell params to change)
        ucf = FrechetCellFilter(structure)
        opt = BFGS(ucf,
                logfile=None,             
                trajectory=traj_path)
        
        tick = time.time()
        opt.run(fmax=fmax, steps=steps)
        tock = time.time()
        
        num_steps = opt.get_number_of_steps()
        sim_time = tock-tick
        print(f"Relaxed: {traj_path.stem} in {num_steps}/{steps} steps ({sim_time:.2f})s")
    else:
        print(f"Skipped relaxtion: {traj_path.stem}")        

    # Write final structure frame
    relaxed_structure = read(traj_path, index=-1)
    
    if not final_frame_path.exists() or OVERWRITE:
        write(final_frame_path, relaxed_structure)

    # Read relaxed structure 
    relaxed_structure = read(Path(final_frame_path))
    relaxed_structure.calc = calc

    # Calculate data
    energy_per_atom = relaxed_structure.get_potential_energy()/len(relaxed_structure)
    forces = relaxed_structure.get_forces()
    a, b, c, alpha, beta, gamma = relaxed_structure.cell.cellpar()
    bond_lengths, bond_angles = get_bond_lengths_and_angles(relaxed_structure, 
                                                            cutoff=1.85, bins=100)

    # Save path and file name
    data_out_dir = Path(data_out_dir) / f"{Path(path_to_model).stem}"
    data_out_dir.mkdir(parents = True, exist_ok=True)
    data_out_path = data_out_dir/ f"{path_to_structure.stem}.json"
    
    # Save results
    potential = Path(path_to_model).name
    structure = Path(path_to_structure).name

    data = {
        "steps_to_relax" : num_steps,
        "step_limit" : steps,
        "potential" : potential,

        "structure" : structure,

        "lattice_parameters": {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma
        },

        "bond_lengths" : bond_lengths,
        "bond_angles"  : bond_angles,
        "potential_energy/atom"   : energy_per_atom,
        "forces": forces.tolist() if hasattr(forces, "tolist") else forces
    }

    if not data_out_path.exists() or OVERWRITE:
        with open(data_out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
    else:
        print(f"Skipped writing: {data_out_path.name}")

    return energy_per_atom, data_out_path

# Import structures and return .json with atomisation and formation energies
def energetics_analysis(in_dir, path_to_models, path_to_graphite_structure,
                        path_to_isolated_atom, fmax, steps, OVERWRITE,
                        ref_traj_out_dir, traj_out_dir, data_out_dir):

    # Import structures
    structure_paths = import_crystal_structures(in_dir)
    if not structure_paths:
        print(f"No structures imported from {in_dir}, ending job...")
        return

    # Import model
    for path_to_model in path_to_models:

        # Calculate graphite energy
        graphite_energy_per_atom,_ = relax_and_calculate(path_to_graphite_structure, path_to_model,
                                                        fmax, steps, OVERWRITE, ref_traj_out_dir,
                                                        data_out_dir)
        # Calculate isolated atom energy
        isolated_energy_per_atom,_ = relax_and_calculate(path_to_isolated_atom, path_to_model, fmax, 
                                                         steps, OVERWRITE, ref_traj_out_dir, 
                                                         data_out_dir)
        # Relax imported structures
        # Write trajectories and final frame 
        counter = 0

        # Loop over all structures
        for file_path in structure_paths:

            energy_per_atom, data_out_path = relax_and_calculate(file_path, path_to_model, fmax, 
                                                                steps, OVERWRITE, traj_out_dir, 
                                                                data_out_dir)

            # Formation energy per atom
            formation_energy_per_atom = energy_per_atom - graphite_energy_per_atom
            
            # Atomisation energy per atom
            atomisation_energy_per_atom = -(energy_per_atom - isolated_energy_per_atom)

            # Read existing JSON
            json_path = Path(data_out_path)
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)     # usually a dict

            # Add calculated data
            data["graphite_energy/atom"]    = graphite_energy_per_atom
            data["isolated_atom_energy"]    = isolated_energy_per_atom
            data["formation_energy/atom"]   = formation_energy_per_atom
            data["atomisation_energy/atom"] = atomisation_energy_per_atom

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
 
            counter += 1

        if counter:
            print(f"\nProcessed {counter} crystal structures with {Path(path_to_model).name}\n")

models_to_analyse = ["MACE_Models/medium-0b3.pt", 
                     "MACE_Models/medium-mpa-0.pt",
                     "MACE_Models/medium-omat-0.pt",
                     ]    

set_OVERWRITE = False

energetics_analysis(
                    in_dir="/u/vld/scat9451/main_project/Carbon_Structures/Crystalline/Downloaded",
                    path_to_models= models_to_analyse,
                    path_to_graphite_structure="Carbon_Structures/Graphite_mp169.cif",
                    path_to_isolated_atom="Carbon_Structures/isolated_C.cif",
                    fmax = 0.0001,
                    steps = 500,
                    OVERWRITE=set_OVERWRITE,
                    ref_traj_out_dir="Carbon_Structures/Relaxed_Reference_Structures",
                    traj_out_dir="Carbon_Structures/Crystalline/Relaxed",
                    data_out_dir="Analysis/Crystalline Analysis/Raw Data"
                    )
