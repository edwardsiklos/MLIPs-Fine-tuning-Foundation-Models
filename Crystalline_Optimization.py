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
import json
import numpy as np

from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

from graph_pes.utils.calculator import GraphPESCalculator
from graph_pes.interfaces import mace_mp

import ovito
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, ColorCodingModifier, BondAnalysisModifier
from ovito.vis import Viewport, TachyonRenderer, ColorLegendOverlay, BondsVis
from ovito.qt_compat import QtCore
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

    if Path(path_to_model).suffix == ".pt":
        import torch
        from graph_pes.models import load_model
        torch.set_default_dtype(torch.float32)
        model = load_model(Path(path_to_model))
        calc = GraphPESCalculator(model, skin=0.1, device='cpu')
    elif Path(path_to_model).suffix == ".xml":
        from quippy.potential import Potential
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
    
        raw_gap = Potential(param_filename=path_to_model)
        calc = GAPWrapper(raw_gap)
    else:
        from graph_pes.interfaces import mace_mp
        model = mace_mp(path_to_model)
        calc = GraphPESCalculator(model, skin=0.1, device='cpu')
    
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

    bond_angles_xy  = data.tables["bond-angle-distr"].xy()
    a_values = bond_angles_xy[:,0]
    a_counts = bond_angles_xy[:,1]
    if a_counts.sum() == 0:
        avg_bond_angle = None
        print(f"Zero angle counts found")
    else:
        avg_bond_angle = np.average(a_values, weights=a_counts)

    bond_lengths_xy = data.tables["bond-length-distr"].xy()
    l_values = bond_lengths_xy[:,0]
    l_counts = bond_lengths_xy[:,1]
    if l_counts.sum() == 0:
        avg_bond_length = None
        print(f"Zero bond counts found")
    else:
        avg_bond_length = np.average(l_values, weights=l_counts)

    # Remove Modifiers
    pipeline.modifiers.pop()
    pipeline.modifiers.pop()

    return avg_bond_length, avg_bond_angle

# Ovito rendering
def ovito_analysis(traj_path, out_path):

    # Clear existing pipeline
    for p in list(ovito.scene.pipelines):
        p.remove_from_scene()

    pipeline = import_file(str(traj_path))

    # Bond Modifier and Visuals 
    bond_modifier = CreateBondsModifier(cutoff=1.85)
    bond_modifier.vis.width = 0.15
    bond_modifier.vis.coloring_mode = BondsVis.ColoringMode.Uniform
    bond_modifier.vis.color = (0.5, 0.5, 0.5)
    pipeline.modifiers.append(bond_modifier)

    # Coordination Modifier and Colour Coding
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
    colour_coding_mod = ColorCodingModifier(property="Coordination",start_value=1.0,end_value=4.0,gradient=ColorCodingModifier.Viridis(),discretize_color_map=True)
    pipeline.modifiers.append(colour_coding_mod)

    # Add to Scene
    pipeline.add_to_scene()

    # Viewing settings
    vp = Viewport()
    vp.type = Viewport.Type.Perspective
    vp.camera_pos = (50, 0, 0)
    vp.camera_dir = (-1, 0, 0)
    vp.camera_up  = (0, 0, 1)

    # Coordination Legend
    legend = ColorLegendOverlay(
        title = "Coordination",
        modifier = colour_coding_mod,
        alignment = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom,
        orientation = QtCore.Qt.Orientation.Horizontal,
        font_size = 0.1,
        format_string = '%.0f' 
        )
    vp.overlays.append(legend)

    tachyon = TachyonRenderer(shadows=False, direct_light_intensity=1.1)

    # Set particle scaling (datafile specific)
    n_frames = pipeline.source.num_frames
    final_frame = max(0, n_frames - 1)
    data = pipeline.compute(frame = final_frame)
    data.particles.vis.scaling = 0.3

    # Set Zoom
    vp.zoom_all()
    
    # Ovito Files
    out_path = str(out_path)

    ovito_save_file = f"{out_path}.ovito"
    if Path(ovito_save_file).exists():
        print(f"Skipped {out_path}.ovito")
    else:
        ovito.scene.save(ovito_save_file)
        print(f"Created {out_path}.ovito")

    # Images    
    img_save_file = f"{out_path}.png"
    if Path(img_save_file).exists():
        print(f"Skipped {out_path}.png")
    else:
        vp.render_image(size=(1920,1080),
                        filename=img_save_file,
                        background=(1,1,1),
                        frame=final_frame,
                        renderer=tachyon)
        print(f"Created {out_path}.png")
            
    # Videos
    vid_save_file   = f"{out_path}.avi"                                  
    if Path(vid_save_file).exists():
        print(f"Skipped {out_path}.avi")
    else:
        vp.render_anim(size=(1920,1080), 
                    filename=vid_save_file, 
                    fps=10,
                    renderer=tachyon) 
        print(f"Created {out_path}.avi")
    # Remove modifiers
    pipeline.modifiers.pop()
    pipeline.modifiers.pop()
    pipeline.modifiers.pop()

# Relaxes a structure 
# Creates .traj, .cif (final frame), .json
# Returns energy/atom and .json file path
# Return ovito renderings
def relax_and_analyse(path_to_structure, path_to_model, fmax, steps,
                         OVERWRITE, traj_out_dir, data_out_dir, ovito_out_dir):

    path_to_structure = Path(path_to_structure)

    # Outpaths
    path_to_structure = Path(path_to_structure)
    traj_dir = Path(traj_out_dir) / "Trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    final_traj_dir = Path(traj_out_dir)/ "Final Trajectory Frame"
    final_traj_dir.mkdir(parents=True, exist_ok=True)

    traj_path = traj_dir / f"{Path(path_to_model).stem}_{path_to_structure.stem}.traj"
    final_frame_path = final_traj_dir / f"{Path(path_to_model).stem}_relaxed_{path_to_structure.stem}.cif"
    
    ovito_out_dir = Path(ovito_out_dir) / path_to_structure.stem
    ovito_out_dir.mkdir(parents=True, exist_ok=True)
    ovito_out_path = ovito_out_dir / Path(path_to_model).stem

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
        opt = BFGS(ucf, trajectory=traj_path)
        
        tick = time.time()
        opt.run(fmax=fmax, steps=steps)
        tock = time.time()
        
        num_steps = opt.get_number_of_steps()
        sim_time = tock-tick
        print(f"Relaxed: {traj_path.stem} in {num_steps}/{steps} steps ({sim_time:.2f})s")
    else:
        print(f"Skipped relaxtion: {traj_path.stem}")        

    # Ovito renders
    ovito_analysis(traj_path, ovito_out_path)

    # Write final structure frame
    relaxed_structure = read(traj_path, index=-1)
    if not final_frame_path.exists() or OVERWRITE:
        write(final_frame_path, relaxed_structure)

    # Read relaxed structure 
    relaxed_structure = read(Path(final_frame_path))
    relaxed_structure.calc = calc

    # Calculate data
    e_bulk = relaxed_structure.get_potential_energy()
    n_atoms = len(relaxed_structure)
    #forces = relaxed_structure.get_forces()
    a, b, c, alpha, beta, gamma = relaxed_structure.cell.cellpar()
    avg_bond_length, avg_bond_angle = get_bond_lengths_and_angles(relaxed_structure, 
                                                            cutoff=1.85, bins=100000)

    # Save path and file name
    data_out_dir = Path(data_out_dir) / f"{Path(path_to_model).stem}"
    data_out_dir.mkdir(parents = True, exist_ok=True)
    data_out_path = data_out_dir/ f"{path_to_structure.stem}.json"
    
    # Save results
    potential = Path(path_to_model).name
    structure = Path(path_to_structure).name

    data = {
        "steps_to_relax"          : num_steps,
        "step_limit"              : steps,
        "potential"               : potential,

        "structure"               : structure,

        "lattice_parameter a"     : a,
        "lattice_parameter b"     : b,
        "lattice_parameter c"     : c,
        "lattice_parameter alpha" : alpha,
        "lattice_parameter beta"  : beta,
        "lattice_parameter gamma" : gamma,

        "average_bond_length"     : avg_bond_length,
        "average_bond_angle"      : avg_bond_angle,
        "potential_energy/atom"   : e_bulk/n_atoms,
        "atoms_in_structure"      : n_atoms
        #"forces": forces.tolist() if hasattr(forces, "tolist") else forces
    }

    if not data_out_path.exists() or OVERWRITE:
        with open(data_out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
            print(f"Wrote: {data_out_path.name}\n")
    else:
        print(f"Skipped: {data_out_path.name}\n")

    return e_bulk, n_atoms, data_out_path

# Import structures and return .json with atomisation and formation energies
def analysis(in_dir, path_to_models, path_to_graphite_structure,
                        path_to_isolated_atom, fmax, steps, OVERWRITE,
                        ref_traj_out_dir, traj_out_dir, data_out_dir, ovito_out_dir):

    # Import structures
    structure_paths = import_crystal_structures(in_dir)
    if not structure_paths:
        print(f"No structures imported from {in_dir}, ending job...")
        return

    # Import model
    for path_to_model in path_to_models:

        # Calculate graphite energy
        graphite_e_bulk,graphite_n_atoms,_ = relax_and_analyse(path_to_graphite_structure, path_to_model,
                                                        fmax, steps, OVERWRITE, ref_traj_out_dir,
                                                        data_out_dir, ovito_out_dir)
        # Calculate isolated atom energy
        isolated_e,n_isolated,_ = relax_and_analyse(path_to_isolated_atom, path_to_model, fmax, 
                                                         steps, OVERWRITE, ref_traj_out_dir, 
                                                         data_out_dir, ovito_out_dir)
        if n_isolated !=1:
            print(f"More than 1 atom counted in isolated atom file")
        # Relax imported structures
        # Write trajectories and final frame 
        counter = 0

        # Loop over all structures
        for file_path in structure_paths:

            e_bulk, n_atoms, data_out_path = relax_and_analyse(file_path, path_to_model, fmax, 
                                                                steps, OVERWRITE, traj_out_dir, 
                                                                data_out_dir, ovito_out_dir)

            # Formation energy per atom
            formation_energy_per_atom = (e_bulk/n_atoms) - (graphite_e_bulk/graphite_n_atoms)
            
            # Atomisation energy: E(f) = E(bulk) - n*E(at) --- From GAP20 paper (misleading equation)
            atomisation_energy = (e_bulk - n_atoms*isolated_e)
            atomisation_energy_per_atom = atomisation_energy/n_atoms
            # Read existing JSON
            json_path = Path(data_out_path)
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)     # usually a dict

            # Add calculated data
            data["graphite_energy/atom"]    = graphite_e_bulk/graphite_n_atoms
            data["isolated_atom_energy"]    = isolated_e/n_isolated
            data["formation_energy/atom"]   = formation_energy_per_atom
            data["atomisation_energy/atom"]      = atomisation_energy_per_atom

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
 
            counter += 1

        if counter:
            print(f"\nProcessed {counter} crystal structures with {Path(path_to_model).name}\n")

models_to_analyse = ["MACE_Models/medium-0b3.pt", 
                     "MACE_Models/medium-mpa-0.pt",
                     "MACE_Models/medium-omat-0.pt",
                     "Potentials/Carbon_GAP_20.xml",
                     "Potentials/carbon.xml"]    

set_OVERWRITE = False

analysis(
    in_dir="/u/vld/scat9451/main_project/Carbon_Structures/Crystalline/Downloaded",
    path_to_models= models_to_analyse,
    path_to_graphite_structure="Carbon_Structures/Graphite_mp169.cif",
    path_to_isolated_atom="Carbon_Structures/isolated_C.cif",
    fmax = 0.0001,
    steps = 1000,
    OVERWRITE=set_OVERWRITE,
    ref_traj_out_dir="Carbon_Structures/Relaxed_Reference_Structures",
    traj_out_dir="Carbon_Structures/Crystalline/Relaxed",
    data_out_dir="Analysis/Crystalline Analysis/Raw Data",
    ovito_out_dir="Analysis/Crystalline Analysis/Ovito"
    )
