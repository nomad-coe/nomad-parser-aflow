#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

import pytest

from nomad.datamodel import EntryArchive
from aflowparser import AFLOWParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return AFLOWParser()


def test_ael(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Ag1Co1O2_ICSD_246157/aflow.ael.out', archive, None)

    assert archive.run[0].program.version == 'aflow30847'
    assert archive.workflow[0].type == 'elastic'
    sec_elastic = archive.workflow[0].elastic

    assert sec_elastic.n_deformations == 3
    assert sec_elastic.strain_maximum == pytest.approx(0.01)
    assert sec_elastic.n_strains == 8
    assert sec_elastic.elastic_constants_matrix_second_order[0][1].magnitude == approx(7.45333e+10)
    assert sec_elastic.bulk_modulus_voigt.magnitude == approx(1.50939e+11)
    assert sec_elastic.pugh_ratio_hill == approx(0.298965)


def test_agl(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Ag1Co1O2_ICSD_246157/aflow.agl.out', archive, None)

    assert archive.workflow[0].type == 'debye_model'
    sec_debye = archive.workflow[0].debye_model
    sec_thermo = archive.workflow[0].thermodynamics

    assert sec_thermo.temperatures[12].magnitude == 120
    assert sec_debye.thermal_conductivity[18].magnitude == approx(4.924586)
    assert sec_debye.gruneisen_parameter[35] == approx(2.255801)
    assert sec_thermo.heat_capacity_c_p[87].magnitude == approx(3.73438425e-22)
    assert sec_thermo.vibrational_free_energy[93].magnitude == approx(-4.13010555e-19)
