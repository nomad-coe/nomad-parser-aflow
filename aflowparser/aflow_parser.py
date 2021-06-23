#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import numpy as np
import logging
import json
from ase.cell import Cell

from nomad.units import ureg
from nomad.parsing import FairdiParser
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.datamodel.metainfo.common_dft import Run, SingleConfigurationCalculation,\
    Method, System, Workflow, DebyeModel, Elastic

from .metainfo import m_env


class AELParser(TextParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):
        ael_quantities = [
            'poisson_ratio', 'bulk_modulus_voigt', 'bulk_modulus_reuss',
            'shear_modulus_voigt', 'shear_modulus_reuss', 'bulk_modulus_vrh',
            'shear_modulus_vrh', 'elastic_anisotropy', 'youngs_modulus_vrh',
            'speed_sound_transverse', 'speed_sound_longitudinal', 'speed_sound_average',
            'pughs_modulus_ratio', 'debye_temperature', 'applied_pressure',
            'average_external_pressure'
        ]
        self._quantities = []
        for quantity in ael_quantities:
            unit = None
            if 'ratio' in quantity:
                unit = None
            elif 'modulus' in quantity or 'pressure' in quantity:
                unit = ureg.GPa
            elif 'speed' in quantity:
                unit = ureg.m / ureg.s
            elif 'temperature' in quantity:
                unit = ureg.K
            self._quantities.append(Quantity(
                quantity, r'ael\_%s=(\S+)' % quantity, dtype=np.float64, unit=unit))

        self._quantities.append(
            Quantity(
                'stiffness_tensor',
                r'\[AEL\_STIFFNESS\_TENSOR\]START([\-\d\.\s]+)\[AEL\_STIFFNESS\_TENSOR\]STOP',
                dtype=np.dtype(np.float64), unit='GPa', shape=(6, 6))
        )

        self._quantities.append(
            Quantity(
                'compliance_tensor',
                r'\[AEL\_COMPLIANCE\_TENSOR\]START([\-\d\.\s]+?)\[AEL\_COMPLIANCE\_TENSOR\]STOP',
                dtype=np.dtype(np.float64), shape=(6, 6))
        )


class AGLParser(TextParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):
        self._quantities = [
            Quantity(
                'thermal_properties',
                r'\[AGL_THERMAL\_PROPERTIES\_TEMPERATURE\]START\s*.+([\-\d\.\s]+?)'
                r'\[AGL\_THERMAL\_PROPERTIES\_TEMPERATURE\]STOP',
                dtype=np.dtype(np.float64)),
            Quantity(
                'energies',
                r'\[AGL\_ENERGIES\_TEMPERATURE\]START\s*.+([\-\d\.\s]+?)'
                r'\[AGL\_ENERGIES\_TEMPERATURE\]STOP',
                dtype=np.dtype(np.float64))
        ]


class AFLOWParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/aflow', code_name='aflow',
            code_homepage='http://www.aflowlib.org/',
            mainfile_contents_re=r'\[AFLOW\] \*{50}',
            mainfile_name_re=(r'.*/aflow\.a(?:e|g)l\.out$')
        )
        self.ael_parser = AELParser()
        self.agl_parser = AGLParser()
        self._metainfo_env = m_env

        self._metainfo_map = {
            'stiffness_tensor': 'elastic_constants_matrix_second_order',
            'compliance_tensor': 'compliance_matrix_second_order',
            'poisson_ratio': 'poisson_ratio_hill',
            'bulk_modulus_vrh': 'bulk_modulus_hill',
            'shear_modulus_vrh': 'shear_modulus_hill',
            'youngs_modulus_vrh': 'Young_modulus_hill',
            'pughs_modulus_ratio': 'pugh_ratio_hill', 'applied_pressure': 'x_aflow_ael_applied_pressure',
            'average_external_pressure': 'x_aflow_ael_average_external_pressure'
        }

    def init_parser(self):
        self.ael_parser.mainfile = None
        self.agl_parser.mainfile = None
        if self.filepath.endswith('aflow.ael.out'):
            self.ael_parser.mainfile = self.filepath
            self.ael_parser.logger = self.logger
        elif self.filepath.endswith('aflow.agl.out'):
            self.agl_parser.mainfile = self.filepath
            self.agl_parser.logger = self.logger

    def parse_agl(self):
        if self.agl_parser.mainfile is None:
            return

        thermal_properties = self.agl_parser.get('thermal_properties')
        if thermal_properties is None:
            return

        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.workflow_type = 'debye_model'
        sec_debye = sec_workflow.m_create(DebyeModel)

        thermal_properties = np.reshape(thermal_properties, (len(thermal_properties) // 9, 9))
        thermal_properties = np.transpose(thermal_properties)
        energies = self.agl_parser.get('energies')
        energies = np.reshape(energies, (len(energies) // 9, 9))
        energies = np.transpose(energies)

        sec_debye.temperatures = thermal_properties[0] * ureg.K
        sec_debye.thermal_conductivity = thermal_properties[1] * ureg.watt / ureg.m * ureg.K
        sec_debye.debye_temperature = thermal_properties[2] * ureg.K
        sec_debye.gruneisen_parameter = thermal_properties[3]
        sec_debye.heat_capacity_C_v = thermal_properties[4] * ureg.boltzmann_constant
        sec_debye.heat_capacity_C_p = thermal_properties[5] * ureg.boltzmann_constant
        sec_debye.thermal_expansion = thermal_properties[6] / ureg.K
        sec_debye.bulk_modulus_static = thermal_properties[7] * ureg.GPa
        sec_debye.bulk_modulus_isothermal = thermal_properties[8] * ureg.GPa
        sec_debye.free_energy_gibbs = energies[1] * ureg.eV
        sec_debye.free_energy_vibrational = energies[2] * ureg.meV
        sec_debye.internal_energy_vibrational = energies[3] * ureg.meV
        sec_debye.entropy_vibrational = energies[4] * ureg.meV / ureg.K

    def parse_ael(self):
        if self.ael_parser.mainfile is None:
            return

        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.workflow_type = 'elastic'
        sec_elastic = sec_workflow.m_create(Elastic)
        sec_elastic.energy_stress_calculator = 'vasp'
        sec_elastic.calculation_method = 'stress'
        sec_elastic.elastic_constants_order = 2

        paths = [d for d in self.aflow_data.get('files', []) if d.startswith('ARUN.AEL')]
        deforms = np.array([d.split('_')[-2:] for d in paths], dtype=np.dtype(np.float64))
        strains = [d[1] for d in deforms if d[0] == 1]
        sec_elastic.n_deformations = int(max(np.transpose(deforms)[0]))
        sec_elastic.n_strains = len(strains)
        sec_elastic.strain_maximum = max(strains) - 1.0

        for key, val in self.ael_parser.items():
            if val is None:
                continue
            key = self._metainfo_map.get(key, key)
            setattr(sec_elastic, key, val)

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging

        self.init_parser()

        # load the aflow metadata from aflowlib.json
        try:
            self.aflow_data = json.load(open(os.path.join(self.maindir, 'aflowlib.json')))
        except Exception:
            self.aflow_data = dict()

        sec_run = self.archive.m_create(Run)
        sec_run.program_name = 'aflow'
        sec_run.program_version = self.aflow_data.get('aflow_version', 'unknown')

        # parse run metadata
        run_quantities = ['aurl', 'auid', 'data_api', 'data_source', 'loop']
        for key in run_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_run, 'x_aflow_%s' % key, val)

        # TODO The OUTCAR file will be read by the vasp parser and so the complete
        # metadata for both system and method should be filled in by vasp parser. However,
        # we need a way to reference this as well as for the deformed structures which are
        # in ARUN.AEL_*
        # parse structure from aflow_data
        sec_system = sec_run.m_create(System)
        lattice_parameters = self.aflow_data.get('geometry')
        if lattice_parameters is not None:
            cell = Cell.fromcellpar(lattice_parameters)
            sec_system.lattice_vectors = cell.array * ureg.angstrom
            sec_system.configuration_periodic_dimensions = [True, True, True]
        species = self.aflow_data.get('species', [])
        atom_labels = []
        for n, specie in enumerate(species):
            atom_labels += [specie] * self.aflow_data['composition'][n]
        sec_system.atom_labels = atom_labels
        sec_system.atom_positions = self.aflow_data.get('positions_cartesian', []) * ureg.angstrom

        # parse system metadata from aflow_data
        system_quantities = [
            'compound', 'prototype', 'nspecies', 'natoms', 'natoms_orig', 'composition',
            'density', 'density_orig', 'scintillation_attenuation_length', 'stoichiometry',
            'species', 'geometry', 'geometry_orig', 'volume_cell', 'volume_atom',
            'volume_cell_orig', 'volume_atom_orig', 'n_sg', 'sg', 'sg2', 'spacegroup_orig',
            'spacegroup_relax', 'Bravais_lattice_orig', 'lattice_variation_orig',
            'lattice_system_orig', 'Pearson_symbol_orig', 'Bravais_lattice_relax',
            'lattice_variation_relax', 'lattice_system_relax', 'Pearson_symbol_relax',
            'crystal_family_orig', 'crystal_system_orig', 'crystal_class_orig',
            'point_group_Hermann_Mauguin_orig', 'point_group_Schoenflies_orig',
            'point_group_orbifold_orig', 'point_group_type_orig', 'point_group_order_orig',
            'point_group_structure_orig', 'Bravais_lattice_lattice_type_orig',
            'Bravais_lattice_lattice_variation_type_orig', 'Bravais_lattice_lattice_system_orig',
            'Bravais_superlattice_lattice_type_orig', 'Bravais_superlattice_lattice_variation_type_orig',
            'Bravais_superlattice_lattice_system_orig', 'Pearson_symbol_superlattice_orig',
            'reciprocal_geometry_orig', 'reciprocal_volume_cell_orig', 'reciprocal_lattice_type_orig',
            'reciprocal_lattice_variation_type_orig', 'Wyckoff_letters_orig',
            'Wyckoff_multiplicities_orig', 'Wyckoff_site_symmetries_orig',
            'crystal_family', 'crystal_system', 'crystal_class', 'point_group_Hermann_Mauguin',
            'point_group_Schoenflies', 'point_group_orbifold', 'point_group_type', 'point_group_order',
            'point_group_structure', 'Bravais_lattice_lattice_type', 'Bravais_lattice_lattice_variation_type',
            'Bravais_lattice_lattice_system', 'Bravais_superlattice_lattice_type',
            'Bravais_superlattice_lattice_variation_type', 'Bravais_superlattice_lattice_system',
            'Pearson_symbol_superlattice', 'reciprocal_geometry', 'reciprocal_volume_cell',
            'reciprocal_lattice_type', 'reciprocal_lattice_variation_type', 'Wyckoff_letters',
            'Wyckoff_multiplicities', 'Wyckoff_site_symmetries', 'prototype_label_orig',
            'prototype_params_list_orig', 'prototype_params_values_orig', 'prototype_label_relax',
            'prototype_params_list_relax', 'prototype_params_values_relax']
        for key in system_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_system, 'x_aflow_%s' % key, val)

        # parse method metadata from self.aflow_data
        method_quantities = [
            'code', 'species_pp', 'n_dft_type', 'dft_type', 'dft_type', 'species_pp_version',
            'species_pp_ZVAL', 'species_pp_AUID', 'ldau_type', 'ldau_l', 'ldau_u', 'ldau_j',
            'valence_cell_iupac', 'valence_cell_std', 'energy_cutoff',
            'delta_electronic_energy_convergence', 'delta_electronic_energy_threshold',
            'kpoints_relax', 'kpoints_static', 'n_kpoints_bands_path', 'kpoints_bands_path',
            'kpoints_bands_nkpts']
        sec_method = sec_run.m_create(Method)
        for key in method_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_method, 'x_aflow_%s' % key, val)

        # parse basic calculation quantities from self.aflow_data
        sec_scc = sec_run.m_create(SingleConfigurationCalculation)
        if self.aflow_data.get('energy_cell') is not None:
            sec_scc.energy_total = self.aflow_data['energy_cell'] * ureg.eV
        if self.aflow_data.get('forces') is not None:
            sec_scc.atom_forces = self.aflow_data['forces'] * ureg.eV / ureg.angstrom
        if self.aflow_data.get('enthalpy_cell') is not None:
            sec_scc.enthalpy = self.aflow_data['enthalpy_cell'] * ureg.eV
        if self.aflow_data.get('entropy_cell') is not None:
            sec_scc.entropy = self.aflow_data['entropy_cell'] * ureg.eV / ureg.K
        if self.aflow_data.get('calculation_time') is not None:
            sec_scc.time_calculation = self.aflow_data['calculation_time'] * ureg.s
        calculation_quantities = [
            'stress_tensor', 'pressure_residual', 'Pulay_stress', 'Egap', 'Egap_fit', 'Egap_type',
            'enthalpy_formation_cell', 'entropic_temperature', 'PV', 'spin_cell', 'spinD',
            'spinF', 'calculation_memory', 'calculation_cores', 'nbondxx',
            'agl_thermal_conductivity_300K', 'agl_debye', 'agl_acoustic_debye', 'agl_gruneisen',
            'agl_heat_capacity_Cv_300K', 'agl_heat_capacity_Cp_300K', 'agl_thermal_expansion_300K',
            'agl_bulk_modulus_static_300K', 'agl_bulk_modulus_isothermal_300K', 'agl_poisson_ratio_source',
            'agl_vibrational_free_energy_300K_cell', 'agl_vibrational_free_energy_300K_atom',
            'agl_vibrational_entropy_300K_cell', 'agl_vibrational_entropy_300K_atom',
            'ael_poisson_ratio', 'ael_bulk_modulus_voigt', 'ael_bulk_modulus_reuss',
            'ael_shear_modulus_voigt', 'ael_shear_modulus_reuss', 'ael_bulk_modulus_vrh',
            'ael_shear_modulus_vrh', 'ael_elastic_anisotropy', 'ael_youngs_modulus_vrh',
            'ael_speed_sound_transverse', 'ael_speed_sound_longitudinal', 'ael_speed_sound_average',
            'ael_pughs_modulus_ratio', 'ael_debye_temperature', 'ael_applied_pressure',
            'ael_average_external_pressure', 'ael_stiffness_tensor', 'ael_compliance_tensor',
            'bader_net_charges', 'bader_atomic_volumes', 'n_files', 'files', 'node_CPU_Model',
            'node_CPU_Cores', 'node_CPU_MHz', 'node_RAM_GB', 'catalog', 'aflowlib_version',
            'aflowlib_date']
        for key in calculation_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_scc, 'x_aflow_%s' % key, val)

        self.parse_ael()
        self.parse_agl()
